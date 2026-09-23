"""
Pure-geometry tests for the uneven-shard padding scheme from PR #4
("fix(muon): support uneven FSDP shards").

The optimizer pads every local FSDP shard to ceil(N / world_size) along the
sharded dim so NCCL all-to-all transfers stay uniform, reconstructs the full
matrix with cat + narrow, orthogonalizes it, then pads to
ceil(N / world_size) * world_size and splits back with tensor_split before
trimming each rank to its true shard.

These tests emulate the two all-to-all collectives in a single CPU process and
check that the pad/reconstruct/trim round-trip is EXACTLY lossless under the
torch.chunk layout that DTensor's Shard placement produces (every shard is
full size except a trailing short or empty one). They need no GPU and no
torchrun:

    pytest tests/test_muon_uneven_shard_padding.py -q
"""

import pytest
import torch


def chunk_layout_sizes(n: int, world_size: int) -> list[int]:
    """Shard sizes produced by torch.chunk (DTensor Shard), with empty fill."""
    sizes = [t.numel() for t in torch.chunk(torch.arange(n), world_size)]
    sizes += [0] * (world_size - len(sizes))
    return sizes


def pad_to_size(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """Same zero-padding along `dim` as muon_update_batch_async."""
    if tensor.size(dim) == size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - tensor.size(dim)
    return torch.cat((tensor, tensor.new_zeros(pad_shape)), dim=dim)


def emulate_all_to_all(inputs_per_rank: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    """dist.all_to_all(outputs, inputs): inputs[r][i] goes from rank r to rank i."""
    world_size = len(inputs_per_rank)
    return [
        [inputs_per_rank[src][dst] for src in range(world_size)]
        for dst in range(world_size)
    ]


def shard_roundtrip(matrix: torch.Tensor, world_size: int, shard_dim: int):
    """
    Run the PR's communication geometry for one matrix on `world_size` emulated
    ranks and return (reconstructed_matrix, recovered_local_shards).
    """
    n = matrix.size(shard_dim)
    padded_local_size = -(-n // world_size)  # ceil

    # Local FSDP shards under the torch.chunk layout used by DTensor Shard.
    offset = 0
    local_shards = []
    for size in chunk_layout_sizes(n, world_size):
        local_shards.append(matrix.narrow(shard_dim, offset, size))
        offset += size

    # Forward: pad every local shard, all-to-all, concatenate, trim padding.
    # Rank j's input list holds its own padded shard at batch index 0; the other
    # world_size - 1 slots are dummies (the optimizer pads short batches the same
    # way). Rank 0 receives every rank's real shard in rank order.
    padded = [pad_to_size(shard, padded_local_size, shard_dim) for shard in local_shards]
    received = emulate_all_to_all([[p] + [p] * (world_size - 1) for p in padded])
    reconstructed = torch.cat(received[0], dim=shard_dim).narrow(shard_dim, 0, n)

    # Reverse: pad to padded_local_size * world_size, tensor_split, all-to-all, trim.
    padded_full = pad_to_size(matrix, padded_local_size * world_size, shard_dim)
    pieces = [
        x.contiguous()
        for x in torch.tensor_split(padded_full, world_size, dim=shard_dim)
    ]
    # Every emulated rank reconstructed the same matrix, so every rank sends the
    # same `pieces` list; rank r keeps what it received for slot r (any source).
    back = emulate_all_to_all([list(pieces) for _ in range(world_size)])
    recovered = [
        back[r][0].narrow(shard_dim, 0, local_shards[r].size(shard_dim))
        for r in range(world_size)
    ]
    return reconstructed, recovered, local_shards


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 8, 64])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24])
@pytest.mark.parametrize("shard_dim", [0, 1])
def test_uneven_shard_roundtrip_is_lossless(n, world_size, shard_dim):
    shape = [6, 6]
    shape[shard_dim] = n
    matrix = torch.randn(shape)
    reconstructed, recovered, local_shards = shard_roundtrip(matrix, world_size, shard_dim)

    # The reconstructed matrix is exactly the original, padding fully removed.
    torch.testing.assert_close(reconstructed, matrix, rtol=0, atol=0)
    # Every rank recovers exactly its own original local shard.
    for got, want in zip(recovered, local_shards):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_roundtrip_3d_sharded_on_middle_dim():
    matrix = torch.randn(2, 5, 4)  # world_size 3 -> chunk sizes [2, 2, 1] on dim 1
    reconstructed, recovered, local_shards = shard_roundtrip(matrix, 3, shard_dim=1)
    torch.testing.assert_close(reconstructed, matrix, rtol=0, atol=0)
    for got, want in zip(recovered, local_shards):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_padding_uses_zeros_so_reconstruction_is_exact():
    # The padding rows must be zeros; any other value would corrupt the update
    # of the rank that reconstructs a matrix containing real rows after them.
    matrix = torch.randn(3, 4)  # world_size 2 -> chunk sizes [2, 1]
    n, world_size, shard_dim = 3, 2, 0
    padded_local_size = -(-n // world_size)
    shard = matrix.narrow(shard_dim, 2, 1)  # rank 1's one-row shard
    padded = pad_to_size(shard, padded_local_size, shard_dim)
    assert padded.size(shard_dim) == 2
    torch.testing.assert_close(padded[1], torch.zeros(4), rtol=0, atol=0)
