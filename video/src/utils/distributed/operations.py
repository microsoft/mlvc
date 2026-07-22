# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import heapq
import time
from typing import Iterable, Callable, Any, List, Optional

__all__ = ["run_distributed"]


def split_task_list_by_index(world_size: int, task_list: List):
    return [idx % world_size for idx in range(len(task_list))]


def split_task_list_by_weight(world_size: int, task_list: List, weight_func: Callable[[Any], float]):
    # heap of groups sorted by accumulated weight
    group_list: list[tuple[float, int]] = [(0, rank) for rank in range(world_size)]
    heapq.heapify(group_list)

    # group index map
    group_map = [0] * len(task_list)

    # reverse sorted list of weighted indices
    index_list = sorted((-weight_func(t), idx) for idx, t in enumerate(task_list))
    for w, idx in index_list:
        # put heaviest item into the lightest group
        g_w, rank = heapq.heappop(group_list)
        group_map[idx] = rank
        g_w -= w
        # push group back into heap
        heapq.heappush(group_list, (g_w, rank))

    return group_map


def run_task_list(task_list: Iterable, task_processor: Callable):
    for task in task_list:
        yield task_processor(task)


def run_with_progress(
    task_list: List,
    list_processor: Optional[Callable[[List], Iterable]],
    progress_callback: Optional[Callable[[int, int, float], Any]],
    progress_interval: float,
):
    assert list_processor is not None
    assert progress_callback is not None
    start_time = last_time = time.time()
    n_tasks = len(task_list)
    for idx, task in enumerate(list_processor(task_list)):
        t = time.time()
        if t - last_time >= progress_interval or idx == n_tasks - 1:
            last_time = t
            progress_callback(idx + 1, n_tasks, t - start_time)

        yield task


def run_distributed(
    rank: int,
    task_list: Iterable,
    *,
    task_processor: Optional[Callable] = None,
    list_processor: Optional[Callable[[List], Iterable]] = None,
    progress_callback: Optional[Callable[[int, int, float], Any]] = None,
    progress_interval: float = 10,
    weight_func: Optional[Callable[[Any], float]] = None,
):
    if task_processor is not None:
        if list_processor is not None:
            raise ValueError("Can not have both task and list processors sepcified")

        list_processor = functools.partial(run_task_list, task_processor=task_processor)

    if progress_callback is not None:
        list_processor = functools.partial(
            run_with_progress,
            list_processor=list_processor,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )

    if not isinstance(task_list, list):
        task_list = list(task_list)
    assert list_processor is not None
    if rank < 0:
        result_list = list(list_processor(task_list))
        if len(result_list) != len(task_list):
            raise ValueError("Task/result count mismatch")
        return result_list
    else:
        import torch.distributed

        world_size = torch.distributed.get_world_size()
        if rank >= world_size:
            raise ValueError(f"Invalid rank: {rank}")

        # split tasks between ranks
        if weight_func is None:
            group_map = split_task_list_by_index(world_size, task_list)
        else:
            group_map = split_task_list_by_weight(world_size, task_list, weight_func=weight_func)

        # run this rank tasks
        rank_task_list = [task_list[idx] for idx, g in enumerate(group_map) if g == rank]
        rank_result = list(list_processor(rank_task_list))
        if len(rank_task_list) != len(rank_result):
            raise ValueError("Task/result count mismatch")

        # collect all results
        group_result_list: List[Optional[List]] = [None] * world_size
        torch.distributed.all_gather_object(group_result_list, rank_result)

        # restore result list
        result_list = [None] * len(group_map)
        for group, result in enumerate(group_result_list):
            # collect group indices
            indices = (idx for idx, g in enumerate(group_map) if g == group)
            # put results into the flat list
            for idx, r in zip(indices, result):  # type: ignore[arg-type]
                result_list[idx] = r

        return result_list
