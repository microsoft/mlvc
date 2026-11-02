# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Faster CUDA graph tree recording for long cascaded clips.

While recording a node, torch looks up each tensor among the outputs of all earlier nodes on the tree path with a
linear scan, so re-recording after a clip length change grows quadratically with the clip length. The patch
replaces the scan with an index keyed by storage data pointer that returns the same results.
"""

import logging
from typing import Any, Optional

_PATCHED = False
_INDEX_ATTR = "_mlvc_path_output_index"


def _get_path_output_index(node: Any) -> dict[int, list[tuple[int, int]]]:
    levels = node.path_weakrefs
    cache = node.__dict__.get(_INDEX_ATTR)
    if cache is None or cache[0] is not levels or len(cache[1]) != len(levels):
        cache = (levels, [0] * len(levels), {})
        node.__dict__[_INDEX_ATTR] = cache

    _, indexed_counts, index = cache
    for depth, level in enumerate(levels):
        start = indexed_counts[depth]
        count = len(level)
        if count < start:
            # a level shrank: rebuild from scratch
            node.__dict__.pop(_INDEX_ATTR, None)
            return _get_path_output_index(node)
        for output_index in range(start, count):
            ref = level[output_index]
            if ref is not None:
                index.setdefault(ref.data_ptr(), []).append((depth, output_index))
        indexed_counts[depth] = count
    return index


def _make_patched_methods(maybe_deref):
    def is_alias_of_live_recorded_tensor(self, t) -> Optional[tuple[int, int]]:
        ptr = t.untyped_storage().data_ptr()
        levels = self.path_weakrefs
        for depth, output_index in _get_path_output_index(self).get(ptr, ()):
            ref = levels[depth][output_index]
            # slots can be re-wrapped in place; the data pointer of a slot does not change
            if ref is None or ref.data_ptr() != ptr:
                return _original_is_alias_of_live_recorded_tensor(self, t)
            if maybe_deref(ref) is not None:
                return depth, output_index
        return None

    def is_cuda_graph_recorded_tensor(self, t) -> bool:
        return t.untyped_storage().data_ptr() in _get_path_output_index(self)

    return is_alias_of_live_recorded_tensor, is_cuda_graph_recorded_tensor


_original_is_alias_of_live_recorded_tensor: Any = None
_original_is_cuda_graph_recorded_tensor: Any = None


def patch_cudagraph_tree_alias_lookup() -> bool:
    """Installs the indexed alias lookup. Returns False if this torch version does not have the expected internals."""
    global _PATCHED, _original_is_alias_of_live_recorded_tensor, _original_is_cuda_graph_recorded_tensor
    if _PATCHED:
        return True

    try:
        from torch._inductor import cudagraph_trees
    except ImportError:
        return False

    node_cls = getattr(cudagraph_trees, "CUDAGraphNode", None)
    maybe_deref = getattr(cudagraph_trees, "maybe_deref", None)
    required = ("_is_alias_of_live_recorded_tensor", "_is_cuda_graph_recorded_tensor")
    if node_cls is None or maybe_deref is None or not all(hasattr(node_cls, n) for n in required):
        logging.warning("CUDA graph tree alias lookup patch is not applicable to this torch version")
        return False

    _original_is_alias_of_live_recorded_tensor = node_cls._is_alias_of_live_recorded_tensor
    _original_is_cuda_graph_recorded_tensor = node_cls._is_cuda_graph_recorded_tensor
    node_cls._is_alias_of_live_recorded_tensor, node_cls._is_cuda_graph_recorded_tensor = _make_patched_methods(
        maybe_deref
    )
    _PATCHED = True
    logging.info("Installed indexed CUDA graph tree alias lookup")
    return True
