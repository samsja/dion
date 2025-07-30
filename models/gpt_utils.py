import os
import glob
import numpy as np
import torch


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, dp_rank, dp_world_size):
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        # ntok_total = 0
        # for fname in self.files:
        #     shard_ntok = _peek_data_shard(fname)
        #     assert shard_ntok >= num_processes * B * T + 1
        #     ntok_total += int(shard_ntok)
        # self.ntok_total = ntok_total
        self.ntok_total = 1300000000000

        # kick things off
        self.reset()

    def _load_tokens_for_current_shard(self):
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def reset(self):
        self.current_shard = 0
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.dp_world_size
        if self.current_position + (B * T * self.dp_world_size + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

    def state_dict(self):
        # return the state dict for the current shard
        return {
            f"current_shard_rank_{self.dp_rank}": self.current_shard,
            f"current_position_rank_{self.dp_rank}": self.current_position,
            "dataloader_world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict):
        # Check that state_dict has the right world size
        state_dict_world_size = state_dict.get("dataloader_world_size")
        if state_dict_world_size != self.dp_world_size:
            raise NotImplementedError(
                f"DistributedDataLoader does not support redistributing checkpoints to a different world size. "
                f"Current process has world size {self.dp_world_size}, but checkpoint has {state_dict_world_size}."
            )

        # Load the state
        self.current_shard = state_dict[f"current_shard_rank_{self.dp_rank}"]
        self.current_position = state_dict[f"current_position_rank_{self.dp_rank}"]
        self._load_tokens_for_current_shard()


class ArithmeticDistributedDataLoader:
    """
    A DDP‑ready loader for raw uint16 .bin files (no header).
    - If the file has at least B*T+1 tokens, we shard it across ranks
      by striding and do circular wrap‑around.
    - Otherwise (e.g. small val.bin), we sample B random windows of length T.
    """

    def __init__(self, data_dir, B, T, rank, world_size, split="train"):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size

        path = os.path.join(data_dir, f"{split}.bin")
        assert os.path.isfile(path), f"didn’t find {path}"
        tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)
        self.tokens = tokens
        self.ntok = len(tokens)
        # only used if we can do striding
        self.pos = self.rank * self.B * self.T

    def reset(self):
        """Restart the circular pointer to the beginning of this split."""
        self.pos = self.rank * self.B * self.T

    def next_batch(self):
        needed = self.B * self.T + 1
        ntok = self.ntok

        # --- CASE 1: big enough to shard+wrap circularly ---
        if ntok >= needed:
            # grab exactly needed tokens, wrapping at the end if necessary
            if self.pos + needed <= ntok:
                buf = self.tokens[self.pos : self.pos + needed]
            else:
                first = self.tokens[self.pos :]
                second = self.tokens[: (needed - len(first))]
                buf = np.concatenate([first, second], axis=0)
            # advance by a full strided chunk, modulo ntok
            self.pos = (self.pos + self.world_size * self.B * self.T) % ntok

            # reshape into (B, T)
            x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
            y = torch.from_numpy(buf[1:]).view(self.B, self.T)
            return x.cuda(), y.cuda()

        # --- CASE 2: too small, fallback to random windows ---
        else:
            # sample B start positions (with replacement)
            max_start = ntok - self.T - 1
            assert (
                max_start > 0
            ), f"split too small: ntok={ntok}, need at least T+1={self.T+1}"
            starts = np.random.randint(0, max_start + 1, size=self.B)
            x_buf = np.stack([self.tokens[s : s + self.T] for s in starts], axis=0)
            y_buf = np.stack(
                [self.tokens[s + 1 : s + self.T + 1] for s in starts], axis=0
            )
            x = torch.from_numpy(x_buf)
            y = torch.from_numpy(y_buf)
            return x.cuda(), y.cuda()
