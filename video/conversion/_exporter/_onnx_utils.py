# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import onnx
import onnxscript
from onnxscript import ir

DEFAULT_ONNX_PASSES = [
    "onnxscript_optimizations",
    "use_space_to_depth",
    "depth_to_space_crd_to_dcr",
    "replace_reciprocal_op",
    "replace_slice_with_split",
    "split_gated_conv",
]


class OnnxOptimizer:
    def __init__(self, model) -> None:
        self._model = model

    def optimize(self, passes=DEFAULT_ONNX_PASSES):
        for pass_name in passes:
            print(f"Running ONNX optimization pass: {pass_name}")
            if pass_name == "onnxscript_optimizations":
                model_ir = ir.from_proto(self._model)
                onnxscript.optimizer.optimize(model_ir)
                self._model = ir.to_proto(model_ir)
            elif pass_name == "use_space_to_depth":
                self._use_space_to_depth()
                onnxscript.optimizer.remove_unused_nodes(self._model)
            elif pass_name == "depth_to_space_crd_to_dcr":
                self._depth_to_space_crd_to_dcr()
                onnxscript.optimizer.remove_unused_nodes(self._model)
            elif pass_name == "replace_reciprocal_op":
                self._replace_reciprocal_op()
            elif pass_name == "replace_slice_with_split":
                self._replace_slice_with_split()
                onnxscript.optimizer.remove_unused_nodes(self._model)
            elif pass_name == "split_gated_conv" or pass_name.startswith("split_gated_conv:"):
                max_splits = int(pass_name.split(":")[1]) if ":" in pass_name else None
                # Skip splitting these convolutions to keep encoder and decoder consistent.
                # In the encoder the conv output has two consumers so the pattern doesn't match,
                # but in the decoder it does. Skipping avoids a mismatch between the two.
                skip_names = {"y_spatial_prior/conv/conv.2/Conv"}
                self._split_gated_conv(
                    max_splits=max_splits,
                    skip_names=skip_names,
                )
                onnxscript.optimizer.remove_unused_nodes(self._model)
            elif pass_name == "qc_workaround_floor_after_round":
                self._qc_workaround_floor_after_round()
            elif pass_name == "qc_workaround_squeeze_gather_4d":
                self._qc_workaround_squeeze_gather_4d()
            elif pass_name == "qc_workaround_clip_min_only":
                self._qc_workaround_clip_min_only()
            else:
                raise ValueError(f"Unknown pass: {pass_name}")
        return self._model

    def _replace_reciprocal_op(self):
        # Qualcomm HTP does not support Reciprocal op, replace it with Div(1.0, input)
        graph = self._graph
        if len([node for node in list(graph.node) if node.op_type == "Reciprocal"]) == 0:
            return

        # Create a scalar "1.0" initializer if needed
        one_init = onnx.helper.make_tensor(
            name="reciprocal_ones_const",
            data_type=onnx.TensorProto.FLOAT16,
            dims=[1],
            vals=[1.0],
        )
        graph.initializer.append(one_init)

        for node in list(graph.node):
            if node.op_type == "Reciprocal":
                new_div = onnx.helper.make_node(
                    "Div",
                    inputs=["reciprocal_ones_const", node.input[0]],
                    outputs=node.output,
                )
                idx = list(graph.node).index(node)
                graph.node.remove(node)
                graph.node.insert(idx, new_div)

    def _qc_workaround_floor_after_round(self):
        # Qualcomm HTP outputs weird non integer values from Round op, add Floor
        # after Round op to fix this
        graph = self._graph
        for node in list(graph.node):
            if node.op_type == "Round":
                rounded_output = node.output[0]
                temp_output = rounded_output + "_temp"
                node.output[0] = temp_output
                floor_node = onnx.helper.make_node(
                    "Floor",
                    inputs=[temp_output],
                    outputs=[rounded_output],
                )
                idx = list(graph.node).index(node)
                graph.node.insert(idx + 1, floor_node)

    def _qc_workaround_squeeze_gather_4d(self):
        # QNN SDK 2.37+ / onnxruntime 1.23.0+ workaround: Gather on 4D initializers
        # with shape [N, C, 1, 1] crashes. Squeeze initializer to [N, C] and insert
        # Reshape after Gather to restore the original [1, C, 1, 1] output shape.
        graph = self._graph
        count = 0
        for node in list(graph.node):
            if node.op_type != "Gather":
                continue

            try:
                axis = self._get_attr(node, "axis").i
            except ValueError:
                axis = 0
            if axis != 0:
                continue

            data_initializer = self._get_initializer(node.input[0])
            if data_initializer is None:
                continue
            dims = list(data_initializer.dims)
            if len(dims) != 4 or dims[2] != 1 or dims[3] != 1:
                continue

            # Squeeze initializer: [N, C, 1, 1] -> [N, C]
            del data_initializer.dims[2:]

            # Insert Reshape after Gather to restore [1, C, 1, 1] shape
            original_output = node.output[0]
            temp_output = original_output + "_squeezed"
            node.output[0] = temp_output

            # Add value_info for the intermediate tensor so Netron displays shape/dtype
            graph.value_info.append(
                onnx.helper.make_tensor_value_info(
                    temp_output,
                    data_initializer.data_type,
                    [1, dims[1]],
                )
            )

            shape_const_name = original_output + "_reshape_shape"
            graph.initializer.append(
                onnx.helper.make_tensor(
                    name=shape_const_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[4],
                    vals=[1, dims[1], 1, 1],
                )
            )

            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[temp_output, shape_const_name],
                outputs=[original_output],
            )
            idx = list(graph.node).index(node)
            graph.node.insert(idx + 1, reshape_node)

            # print(f"Gather '{node.name}' data squeezed: {dims} -> {list(data_initializer.dims)}")
            count += 1

        if count > 0:
            print(f"qc_workaround_squeeze_gather_4d: transformed {count} Gather node(s)")

    def _qc_workaround_clip_min_only(self):
        # Regression in onnxruntime-qnn==1.23.* (works in 1.22 and 1.24): Clip nodes
        # with only a min bound (optional max left empty) fail QNN EP validation and
        # fall back to CPU, preventing execution on the NPU. This pass replaces such
        # Clip(x, min) nodes with Max(x, min_initializer), which is semantically
        # equivalent and fully supported by QNN HTP.
        graph = self._graph
        count = 0
        for node in list(graph.node):
            if node.op_type != "Clip":
                continue
            has_max = len(node.input) >= 3 and node.input[2] != ""
            if has_max:
                continue
            has_min = len(node.input) >= 2 and node.input[1] != ""
            if not has_min:
                continue

            min_input = node.input[1]

            # Check if the min input comes from a Constant node; if so, convert
            # it to an initializer so Max can consume it directly.
            min_const_node = None
            for n in graph.node:
                if n.op_type == "Constant" and len(n.output) > 0 and n.output[0] == min_input:
                    min_const_node = n
                    break

            if min_const_node is not None:
                # Extract value from the Constant node and create an initializer
                init_name = min_input + "_init"
                value_tensor = self._get_attr(min_const_node, "value").t
                graph.initializer.append(
                    onnx.helper.make_tensor(
                        name=init_name,
                        data_type=value_tensor.data_type,
                        dims=list(value_tensor.dims),
                        vals=value_tensor.raw_data,
                        raw=True,
                    )
                )
                min_input = init_name

            # Replace Clip with Max
            max_node = onnx.helper.make_node(
                "Max",
                inputs=[node.input[0], min_input],
                outputs=list(node.output),
                name=(node.name or "") + "_as_max",
            )
            idx = list(graph.node).index(node)
            graph.node.remove(node)
            graph.node.insert(idx, max_node)
            count += 1

        if count > 0:
            print(f"qc_workaround_clip_min_only: replaced {count} Clip node(s) with Max")
            onnxscript.optimizer.remove_unused_nodes(self._model)

    def _use_space_to_depth(self):
        # Qualcomm HTP does not support PixelUnshuffle, replace it with SpaceToDepth
        while True:
            self._build_lookup_tables()
            model_changed = False
            for node in self._graph.node:
                if self._try_replace_pixel_unshuffle(node):
                    print("PixelUnshuffle node replaced")
                    model_changed = True
                    break
            if not model_changed:
                break

    def _depth_to_space_crd_to_dcr(self):
        while True:
            self._build_lookup_tables()
            model_changed = False
            for node in self._graph.node:
                if self._try_replace_pixel_shuffle(node):
                    print("PixelShuffle node replaced")
                    model_changed = True
                    break
            if not model_changed:
                break

    def _replace_slice_with_split(self):
        while True:
            self._build_lookup_tables()
            model_changed = False
            for node in self._graph.node:
                if self._try_replace_slice_chunking(node):
                    model_changed = True
                    break
            if not model_changed:
                break

    @property
    def _graph(self):
        return self._model.graph

    def _build_lookup_tables(self):
        self._input_to_nodes_map = {}
        for node in self._graph.node:
            for input_name in node.input:
                if input_name not in self._input_to_nodes_map:
                    self._input_to_nodes_map[input_name] = []
                self._input_to_nodes_map[input_name].append(node)

    @staticmethod
    def _get_attr(node, name):
        for attr in node.attribute:
            if attr.name == name:
                return attr
        raise ValueError(f"Attribute {name} not found")

    def _get_initializer(self, name):
        for initializer in self._graph.initializer:
            if initializer.name == name:
                return initializer
        return None

    def _get_constant(self, name):
        for node in self._graph.node:
            if node.op_type == "Constant":
                if node.output[0] == name:
                    return onnx.numpy_helper.to_array(node.attribute[0].t)
        initializer = self._get_initializer(name)
        if initializer is not None:
            return onnx.numpy_helper.to_array(initializer)
        raise ValueError(f"Constant {name} not found")

    def _get_tensor_shape(self, name):
        for value_info in self._graph.value_info:
            if value_info.name == name:
                shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                return shape
        raise ValueError(f"Tensor {name} not found in value_info")

    def _get_child_node(self, node):
        if len(node.output) != 1:
            return None
        output_name = node.output[0]
        if output_name not in self._input_to_nodes_map:
            return None
        child_nodes = self._input_to_nodes_map[output_name]
        if len(child_nodes) != 1:
            return None
        return child_nodes[0]

    # PixelUnshuffle -> SpaceToDepth

    def _match_unshuffle_pattern(self, node):
        # TODO: Add more checks to be sure that the pattern corresponds to PixelUnshuffle
        reshape_node1 = node
        if reshape_node1.op_type != "Reshape":
            return None

        transpose_node = self._get_child_node(reshape_node1)
        if transpose_node is None or transpose_node.op_type != "Transpose":
            return None

        reshape_node2 = self._get_child_node(transpose_node)
        if reshape_node2 is None or reshape_node2.op_type != "Reshape":
            return None

        conv_node = self._get_child_node(reshape_node2)
        if conv_node is None:
            return None

        concat_node = None
        if conv_node.op_type == "Concat":
            concat_node = conv_node
            conv_node = self._get_child_node(conv_node)
            if conv_node is None or conv_node.op_type != "Conv":
                return None
        elif conv_node.op_type != "Conv":
            return None

        return (reshape_node1, transpose_node, reshape_node2, conv_node, concat_node)

    def _try_replace_pixel_unshuffle(self, node):
        # torch exports PixelUnshuffle as Reshape -> Transpose -> Reshape
        # we assume Reshape -> Transpose -> Reshape -> Conv pattern
        # or Reshape -> Transpose -> Reshape -> Concat -> Conv pattern (used in DMC-6.5 encoder)
        matched_nodes = self._match_unshuffle_pattern(node)
        if matched_nodes is None:
            return False

        reshape_node1, transpose_node, reshape_node2, conv_node, concat_node = matched_nodes

        # Create a new SpaceToDepth node
        blocksize = self._get_constant(reshape_node1.input[1])[-1]
        depth_to_space_node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=[reshape_node1.input[0]],
            outputs=[reshape_node2.output[0]],
            blocksize=blocksize,
        )

        # Permute the weights
        weights_initializer = self._get_initializer(conv_node.input[1])
        if weights_initializer is None:
            raise ValueError("Conv weights not found")
        weights = onnx.numpy_helper.to_array(weights_initializer)

        if concat_node is not None:
            num_channels = self._get_tensor_shape(concat_node.input[0])[1]
            weights0 = weights[:, :num_channels]
            weights1 = weights[:, num_channels:]
            weights = weights0

        original_shape = weights.shape
        new_shape = (
            weights.shape[0],
            weights.shape[1] // (blocksize**2),
            blocksize,
            blocksize,
            weights.shape[2],
            weights.shape[3],
        )
        weights = weights.reshape(new_shape)
        weights = weights.transpose(0, 2, 3, 1, 4, 5)
        weights = weights.reshape(original_shape)

        if concat_node is not None:
            weights = np.concatenate([weights, weights1], axis=1)

        weights_initializer.CopyFrom(onnx.numpy_helper.from_array(weights, weights_initializer.name))

        # Modify graph
        graph = self._graph
        graph.node.insert(list(graph.node).index(reshape_node2) + 1, depth_to_space_node)
        graph.node.remove(reshape_node1)
        graph.node.remove(transpose_node)
        graph.node.remove(reshape_node2)

        return True

    # DepthToSpace CRD -> DCR

    def _match_pixel_shuffle_pattern(self, node):
        conv_node = node
        if conv_node.op_type != "Conv":
            return None
        depth_to_space = self._get_child_node(conv_node)
        if depth_to_space is None or depth_to_space.op_type != "DepthToSpace":
            return None
        mode = self._get_attr(depth_to_space, "mode").s.decode("utf-8")
        if mode != "CRD":
            return None
        return (conv_node, depth_to_space)

    def _try_replace_pixel_shuffle(self, node):
        matched_nodes = self._match_pixel_shuffle_pattern(node)
        if matched_nodes is None:
            return False
        conv_node, depth_to_space = matched_nodes
        blocksize = self._get_attr(depth_to_space, "blocksize").i

        weights_initializer = self._get_initializer(conv_node.input[1])
        if weights_initializer is None:
            raise ValueError("Conv weights not found")
        weights = onnx.numpy_helper.to_array(weights_initializer)

        bias_initializer = self._get_initializer(conv_node.input[2])
        if bias_initializer is None:
            raise ValueError("Conv bias not found")
        bias = onnx.numpy_helper.to_array(bias_initializer)

        # Permute the weights
        original_weights_shape = weights.shape
        weights = (
            weights.reshape(
                (
                    weights.shape[0] // (blocksize**2),
                    blocksize,
                    blocksize,
                    weights.shape[1],
                    weights.shape[2],
                    weights.shape[3],
                )
            )
            .transpose(1, 2, 0, 3, 4, 5)
            .reshape(original_weights_shape)
        )
        weights_initializer.CopyFrom(onnx.numpy_helper.from_array(weights, weights_initializer.name))

        # Permute the bias
        original_bias_shape = bias.shape
        bias = (
            bias.reshape((bias.shape[0] // (blocksize**2), blocksize, blocksize))
            .transpose(1, 2, 0)
            .reshape(original_bias_shape)
        )
        bias_initializer.CopyFrom(onnx.numpy_helper.from_array(bias, bias_initializer.name))

        # Modify the existing DepthToSpace node's mode attribute
        self._get_attr(depth_to_space, "mode").s = b"DCR"

        return True

    # Replace 2 x Slice with Split

    def _match_chunk_pattern(self, node):
        if node.op_type == "Constant":
            return None

        for output in node.output:
            slice_nodes = [n for n in self._input_to_nodes_map.get(output, []) if n.op_type == "Slice"]
            if len(slice_nodes) < 2:
                continue

            chunk_axes = []
            chunk_sizes = []
            for slice_node in slice_nodes:
                starts = self._get_constant(slice_node.input[1])[0]
                ends = self._get_constant(slice_node.input[2])[0]
                axis = self._get_constant(slice_node.input[3])[0]
                chunk_axes.append(axis)
                chunk_sizes.append(ends - starts)

            if len(set(chunk_axes)) != 1 or len(set(chunk_sizes)) != 1 or chunk_axes[0] != 1:
                print(f"Invalid chunk pattern for {output}: axes={chunk_axes}, sizes={chunk_sizes}")
                continue

            axis = chunk_axes[0]
            return (node, slice_nodes, axis)

        return None

    def _try_replace_slice_chunking(self, node):
        matched_nodes = self._match_chunk_pattern(node)
        if matched_nodes is None:
            return False
        source_node, slice_nodes, axis = matched_nodes

        split_node = onnx.helper.make_node(
            "Split",
            inputs=[slice_nodes[0].input[0]],
            outputs=[n.output[0] for n in slice_nodes],
            axis=axis,
            num_outputs=len(slice_nodes),
        )

        graph = self._graph
        graph.node.insert(list(graph.node).index(source_node) + 1, split_node)
        for slice_node in slice_nodes:
            graph.node.remove(slice_node)

        return True

    def _split_gated_conv(self, max_splits=None, min_chunk_size=None, skip_names=None):
        """Split conv -> split into N separate convolutions.

        Pattern: conv(in_ch → N*out_ch, kernel=1x1, groups=1) → split(axis=1, num_splits=N)
        Replaced: conv_0(in_ch → out_ch) + ... + conv_N-1(in_ch → out_ch)

        Args:
            max_splits: Maximum number of splits to allow. None means no limit.
            min_chunk_size: Minimum output channels per split. None means no limit.
            skip_names: Set of substrings; skip convolutions whose name contains any of them.
        """
        while True:
            self._build_lookup_tables()
            model_changed = False
            for node in self._graph.node:
                if self._try_split_gated_conv(node, max_splits, min_chunk_size, skip_names):
                    model_changed = True
                    break
            if not model_changed:
                break

    def _try_split_gated_conv(self, conv_node, max_splits=None, min_chunk_size=None, skip_names=None):

        if conv_node.op_type != "Conv":
            return False

        weights_init = self._get_initializer(conv_node.input[1])
        if weights_init is None:
            return False
        weights = onnx.numpy_helper.to_array(weights_init)
        out_ch = weights.shape[0]
        if len(weights.shape) != 4 or weights.shape[2] != 1 or weights.shape[3] != 1:
            return False
        try:
            if self._get_attr(conv_node, "group").i != 1:
                return False
        except ValueError:
            pass  # no group attr means groups=1

        if skip_names and any(s in conv_node.name for s in skip_names):
            return False

        # Match either Conv → Split  OR  Conv → Slice, Slice, ...
        conv_output = conv_node.output[0]
        consumers = self._input_to_nodes_map.get(conv_output, [])

        split_outputs = []  # ordered list of output tensor names
        nodes_to_remove = []  # Split or Slice nodes to delete

        child = self._get_child_node(conv_node)
        if child is not None and child.op_type == "Split":
            # Pattern A: Conv → Split
            num_splits = len(child.output)
            try:
                if self._get_attr(child, "axis").i != 1:
                    return False
            except ValueError:
                return False
            split_outputs = list(child.output)
            nodes_to_remove = [child]
        else:
            # Pattern B: Conv → Slice, Slice, ... (torch.chunk export)
            # This pattern happens when the replace_slice_with_split has not been run
            slice_nodes = [n for n in consumers if n.op_type == "Slice"]
            if len(slice_nodes) < 2:
                return False
            # All consumers of the conv output must be Slices (no extra users)
            if len(consumers) != len(slice_nodes):
                return False
            # Verify equal-sized chunks along axis 1
            slice_info = []
            for sn in slice_nodes:
                try:
                    starts = self._get_constant(sn.input[1])[0]
                    ends = self._get_constant(sn.input[2])[0]
                    axis = self._get_constant(sn.input[3])[0]
                except (ValueError, IndexError):
                    return False
                if axis != 1:
                    return False
                slice_info.append((starts, ends, sn))
            slice_info.sort(key=lambda x: x[0])
            chunk_size = slice_info[0][1] - slice_info[0][0]
            if any(e - s != chunk_size for s, e, _ in slice_info):
                return False
            num_splits = len(slice_info)
            split_outputs = [sn.output[0] for _, _, sn in slice_info]
            nodes_to_remove = [sn for _, _, sn in slice_info]

        if num_splits < 2 or out_ch % num_splits != 0:
            return False
        if max_splits is not None and num_splits > max_splits:
            return False

        chunk = out_ch // num_splits
        if min_chunk_size is not None and chunk < min_chunk_size:
            return False

        graph = self._graph

        bias_init = None
        bias = None
        if len(conv_node.input) >= 3 and conv_node.input[2] != "":
            bias_init = self._get_initializer(conv_node.input[2])
            if bias_init is not None:
                bias = onnx.numpy_helper.to_array(bias_init)

        for i in range(num_splits):
            suffix = f"_{chr(ord('a') + i)}"
            w_name = weights_init.name + suffix
            graph.initializer.append(onnx.numpy_helper.from_array(weights[i * chunk : (i + 1) * chunk], w_name))
            inputs = [conv_node.input[0], w_name]
            if bias is not None:
                assert bias_init is not None
                b_name = bias_init.name + suffix
                graph.initializer.append(onnx.numpy_helper.from_array(bias[i * chunk : (i + 1) * chunk], b_name))
                inputs.append(b_name)

            new_conv = onnx.helper.make_node(
                "Conv",
                inputs=inputs,
                outputs=[split_outputs[i]],
                name=(conv_node.name or "") + f"_split{suffix}",
            )
            for attr in conv_node.attribute:
                new_attr = onnx.AttributeProto()
                new_attr.CopyFrom(attr)
                new_conv.attribute.append(new_attr)
            graph.node.insert(list(graph.node).index(conv_node), new_conv)

        graph.node.remove(conv_node)
        for n in nodes_to_remove:
            graph.node.remove(n)
        print(f"split_gated_conv: split Conv '{conv_node.name}' ({out_ch} ch) into {num_splits} ({chunk} ch each)")
        return True
