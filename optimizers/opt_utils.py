import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed.tensor import DTensor
from typing import Generator, List, Optional, Union


def to_local(tensor: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
    """
    Convert a single DTensor or list of DTensors to local tensors.
    This is a no-op for regular tensors.
    """
    if isinstance(tensor, Tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]


def create_param_batches(
    params: List[Tensor], batch_size: int, pad: bool = True
) -> Generator[List[Tensor], None, None]:
    """
    Batch parameters into groups of size `batch_size`.
    Tensors in each batch will have identical shape, sharding, and dtype.
    If pad is True, batches will be padded with empty tensors to ensure uniform size.
    If pad is False, the last batch may be smaller than `batch_size`.
    """
    # Group parameters by shape, sharding, and dtype
    groups = defaultdict(list)
    for p in params:
        sharding = p.placements if isinstance(p, DTensor) else None
        groups[(p.shape, sharding, p.dtype)].append(p)

    # Create batches from grouped parameters
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            while pad and len(batch) < batch_size:
                batch.append(torch.empty_like(batch[0]))
            yield batch


class AsyncTask:
    """
    AsyncTask wraps a Python generator to run until the next yield statement.
    This is used to allow other tasks to run while waiting for distributed operations.
    """

    def __init__(self, generator: Generator[None, None, None]):
        self._generator = generator
        next(self._generator)  # Start running the generator

    def run(self) -> bool:
        # Run the next step of the async task.
        # Returns True if the task is still running and False if completed.
        try:
            next(self._generator)
            return True
        except StopIteration:
            pass
        return False


class AsyncRuntime:
    """
    Event loop for running multiple async tasks concurrently.
    """

    def __init__(
        self, task_gen: Generator["AsyncTask", None, None], max_concurrent_tasks: int
    ):
        # Initialize runtime with a generator that produces AsyncTask objects
        if max_concurrent_tasks <= 0:
            raise ValueError(f"{max_concurrent_tasks=} cannot be <= 0")
        self._task_gen = task_gen
        self._max_concurrent_tasks = max_concurrent_tasks

    def _get_next_task(self) -> Optional["AsyncTask"]:
        try:
            task = next(self._task_gen)
            return task
        except StopIteration:
            return None

    def run(self):
        # Run the event loop until all tasks are completed
        have_new_tasks = True
        previous_tasks = []

        while have_new_tasks or previous_tasks:
            # See if we can add another task
            running_tasks = []
            if have_new_tasks and len(previous_tasks) < self._max_concurrent_tasks:
                new_task = self._get_next_task()
                if new_task is not None:
                    # Add new task to the queue
                    running_tasks.append(new_task)
                else:
                    # No more tasks left
                    have_new_tasks = False

            # Run all previous tasks for one step
            for task in previous_tasks:
                still_running = task.run()
                if still_running:
                    running_tasks.append(task)

            # Update task list for next iteration
            previous_tasks = running_tasks
