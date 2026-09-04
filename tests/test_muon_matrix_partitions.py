import pytest
import torch

from dion import Muon

DEVICE = "cuda"

# These tests sweep many matrix shapes through the same compiled update kernels,
# far more than a single model ever would.
torch._dynamo.config.recompile_limit = 64


def copy_matrix(matrix):
    """Copy a split view into fresh contiguous storage, sharing nothing with the fused tensor."""
    return torch.empty_like(matrix, memory_format=torch.contiguous_format).copy_(matrix)


def split_matrices(fused, partitions):
    return fused.detach().split(partitions, dim=-2)


def make_muon(params, matrix_partitions=None, adjust_lr="rms_norm"):
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=None,
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr=adjust_lr,
        matrix_partitions=matrix_partitions,
    )


def train_fused_and_independent(shape, partitions, steps=3, adjust_lr="rms_norm", seed=0):
    """
    Optimize one parameter packing `partitions` along dim -2 and, on the same gradients,
    the independent matrices it packs. Returns the fused parameter and the independent ones.
    """
    torch.manual_seed(seed)
    fused = torch.nn.Parameter(torch.randn(shape, device=DEVICE))
    independent = [torch.nn.Parameter(copy_matrix(matrix)) for matrix in split_matrices(fused, partitions)]

    fused_optimizer = make_muon([fused], {fused: partitions}, adjust_lr)
    independent_optimizer = make_muon(independent, None, adjust_lr)

    for _ in range(steps):
        gradient = torch.randn(shape, device=DEVICE)
        fused.grad = gradient.clone()
        for parameter, matrix in zip(independent, gradient.split(partitions, dim=-2)):
            parameter.grad = copy_matrix(matrix)
        fused_optimizer.step()
        independent_optimizer.step()

    return fused, independent


def assert_partitions_match(fused, independent, partitions):
    for matrix, parameter in zip(split_matrices(fused, partitions), independent):
        torch.testing.assert_close(matrix, parameter.data, rtol=0, atol=0)


def test_single_partition_matches_unpartitioned():
    fused, independent = train_fused_and_independent((256, 128), (256,))
    assert_partitions_match(fused, independent, (256,))


def test_equal_partitions_match_independent_matrices():
    partitions = (192, 192)
    fused, independent = train_fused_and_independent((384, 128), partitions)
    assert_partitions_match(fused, independent, partitions)


def test_expert_partitions_match_independent_matrices():
    partitions = (96, 96)
    fused, independent = train_fused_and_independent((4, 192, 64), partitions)
    assert_partitions_match(fused, independent, partitions)


def test_unequal_partitions_match_independent_matrices():
    partitions = (256, 64, 64)
    fused, independent = train_fused_and_independent((384, 128), partitions)
    assert_partitions_match(fused, independent, partitions)


def test_head_partitions_match_independent_matrices():
    partitions = (32,) * 8
    fused, independent = train_fused_and_independent((256, 128), partitions)
    assert_partitions_match(fused, independent, partitions)


@pytest.mark.parametrize("adjust_lr", ["spectral_norm", "rms_norm", "keller_muon", None])
def test_unequal_partitions_match_for_every_lr_adjustment(adjust_lr):
    partitions = (256, 64, 64)
    fused, independent = train_fused_and_independent((384, 128), partitions, adjust_lr=adjust_lr)
    assert_partitions_match(fused, independent, partitions)


def test_partitioned_parameter_keeps_one_physical_state():
    fused, _ = train_fused_and_independent((384, 128), (192, 192), steps=1)
    optimizer = make_muon([fused], {fused: (192, 192)})
    fused.grad = torch.randn_like(fused)
    optimizer.step()

    assert len(optimizer.state) == 1
    assert list(optimizer.state[fused]) == ["momentum"]
    assert optimizer.state[fused]["momentum"].shape == fused.shape


def test_partitions_must_sum_to_the_output_dimension():
    parameter = torch.nn.Parameter(torch.randn(384, 128, device=DEVICE))
    with pytest.raises(ValueError, match="penultimate dimension"):
        make_muon([parameter], {parameter: (192, 100)})


def test_partitions_must_be_positive():
    parameter = torch.nn.Parameter(torch.randn(384, 128, device=DEVICE))
    with pytest.raises(ValueError, match="positive sizes"):
        make_muon([parameter], {parameter: (384, 0)})


def test_partitions_must_belong_to_the_optimizer():
    parameter = torch.nn.Parameter(torch.randn(384, 128, device=DEVICE))
    other = torch.nn.Parameter(torch.randn(384, 128, device=DEVICE))
    with pytest.raises(ValueError, match="not given to the optimizer"):
        make_muon([parameter], {other: (192, 192)})
