# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalOperand=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
import numpy as np
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil import Builder as mb


@register_pass(namespace="mlvc")
class fast_prediction_workaround(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_workaround_in_block(f)

    def _match_and_replace_pattern(self, block, act_op):
        # Workaround for CoreML fast prediction bug (?)
        # Replace "activation -> split -> add" with "split -> 2 x activation -> add"

        if act_op.op_type not in (
            "silu",
            "relu",
            "gelu",
            "leaky_relu",
        ) or not self._check_next_op_type(act_op, "split"):
            return False
        if act_op.x.shape[1] <= 1024:
            # Workaround is only needed if the number of channels is high
            return False
        split_op = list(act_op.outputs[0].child_ops)[0]
        if not self._check_next_op_type(split_op, "add"):
            return False
        add_op = list(split_op.outputs[0].child_ops)[0]

        # Add "split -> 2 x activation -> add" operations
        new_split_x, new_split_y = mb.split(
            x=act_op.x,
            axis=split_op.axis,
            split_sizes=split_op.split_sizes,
            num_splits=split_op.num_splits,
            name=split_op.name + "_workaround",
            before_op=add_op,
        )

        act_mb = getattr(mb, act_op.op_type)
        act_args = {}
        if act_op.op_type == "leaky_relu":
            act_args["alpha"] = act_op.alpha.val
        new_act_1 = act_mb(x=new_split_x, name=act_op.name + "_workaround_1", before_op=add_op, **act_args)
        new_act_2 = act_mb(x=new_split_y, name=act_op.name + "_workaround_2", before_op=add_op, **act_args)
        new_add_op = mb.add(x=new_act_1, y=new_act_2, name=add_op.name + "_workaround", before_op=add_op)

        # Remove old "activation -> split -> add" operations
        if add_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=add_op,
            old_var=add_op.outputs[0],
            new_var=new_add_op,
        ):
            block.remove_ops([act_op, split_op, add_op])
            return True

        return False

    @staticmethod
    def _check_next_op_type(op, child_op_type):
        if len(op.outputs) == 0:
            return False
        child_ops = list(op.outputs[0].child_ops)
        return child_ops[0].op_type == child_op_type

    @block_context_manager
    def _apply_workaround_in_block(self, block):
        def help_apply_workaround(block):
            for op in list(block.operations):
                if self._match_and_replace_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply_workaround(block)


@register_pass(namespace="mlvc", override=True)
class pixel_shuffle_workaround(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_workaround_in_block(f)

    def _match_and_replace_pattern(self, block, op):
        if op.op_type != "pixel_shuffle":
            return False
        if op.x.shape[1] < 1024:
            return False
        upscale_factor = op.upscale_factor.val
        num_splits = upscale_factor
        print(f"Found pixel_shuffle op {op.x.shape} (upscale_factor={upscale_factor}, num_splits={num_splits})")

        splits = mb.split(
            x=op.x,
            axis=1,
            num_splits=num_splits,
            name=op.name + "_split",
            before_op=op,
        )
        if not isinstance(splits, (list, tuple)):
            splits = [splits]

        split_shuffles = []
        for i, split in enumerate(splits):
            s = mb.pixel_shuffle(
                x=split,
                upscale_factor=upscale_factor,
                name=op.name + f"_shuffle{i}",
                before_op=op,
            )
            split_shuffles.append(s)

        if len(split_shuffles) == 1:
            concat_op = split_shuffles[0]
        else:
            concat_op = mb.concat(
                values=tuple(split_shuffles),
                axis=1,
                name=op.name + "_concat",
                before_op=op,
            )

        if op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=concat_op,
        ):
            block.remove_ops([op])
            return True

        return False

    @block_context_manager
    def _apply_workaround_in_block(self, block):
        def help_apply_workaround(block):
            for op in list(block.operations):
                if self._match_and_replace_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply_workaround(block)


def replace_op_var(op, old_var, new_var):
    ops = list(op.enclosing_block.operations)
    op_before = ops[ops.index(op) - 1]
    if not op.enclosing_block.try_replace_uses_of_var_after_op(
        anchor_op=op_before,
        old_var=old_var,
        new_var=new_var,
        end_op=op,
    ):
        raise ValueError(f"Failed to replace {old_var} with {new_var}")


def quantize_op_var(op, input_var, scale=1.0, zero_point=0):
    from coremltools.converters.mil.mil.types import fp16, fp32

    if input_var.dtype not in [fp16, fp32]:
        return False
    if input_var.op.op_type == "dequantize":
        return False
    if input_var.rank == 0:
        return False

    quantize_op = mb.quantize(
        input=input_var,
        zero_point=np.int8(zero_point),
        scale=np.float32(scale),
        output_dtype="int8",
        before_op=op,
        name=f"{input_var.op.name}_quantize",
    )
    dequantize_op = mb.dequantize(
        input=quantize_op,
        zero_point=np.int8(zero_point),
        scale=np.float32(scale),
        before_op=op,
        name=f"{input_var.op.name}_dequantize",
    )

    replace_op_var(op, input_var, dequantize_op)
    return True


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_skip_connections(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_in_block(f)

    def _match_pattern(self, block, op):

        for output in op.outputs:
            child_ops = list(output.child_ops)
            quantize_op = None
            add_ops = []
            multiple_quantize_ops = False
            for child_op in child_ops:
                if child_op.op_type == "quantize":
                    if quantize_op is not None:
                        multiple_quantize_ops = True
                    quantize_op = child_op
                elif child_op.op_type == "add":
                    add_ops.append(child_op)

            if quantize_op is None or len(add_ops) < 1:
                continue

            if multiple_quantize_ops:
                print("Warn: Multiple quantize ops are not supported, skip")
                continue

            for add_op in add_ops:
                scale = 3.0 * 1.0 / 128.0
                zero_point = 0
                dequantize_op = mb.dequantize(
                    input=quantize_op.outputs[0],
                    zero_point=np.int8(zero_point),
                    scale=np.float32(scale),
                    before_op=add_op,
                )
                replace_op_var(add_op, output, dequantize_op)
            return True

        return False

    @block_context_manager
    def _apply_in_block(self, block):
        def help_apply_in_block(block):
            for op in list(block.operations):
                if self._match_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply_in_block(block)


class DummyQuantizeGraphPassBase(AbstractGraphPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = []
        self._inputs_to_quantize = []

    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_in_block(f)

    def _match_pattern(self, block, op):
        if op.op_type not in self._match_ops:
            return False

        for input_name in self._inputs_to_quantize:
            if input_name not in op.inputs:
                continue
            input_var = op.inputs[input_name]
            if isinstance(input_var, (list, tuple)):
                for var in input_var:
                    if quantize_op_var(op, var, scale=4.0 / 128.0):
                        return True
            else:
                if quantize_op_var(op, input_var, scale=4.0 / 128.0):
                    return True
        return False

    @block_context_manager
    def _apply_in_block(self, block):
        def help_apply_in_block(block):
            for op in list(block.operations):
                if self._match_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply_in_block(block)


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_conv_weight(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["conv"]
        self._inputs_to_quantize = ["weight"]


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_conv_activations(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["conv"]
        self._inputs_to_quantize = ["x"]


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_activation_functions(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["relu", "silu"]
        self._inputs_to_quantize = ["x"]

    def _match_pattern(self, block, op):
        if op.op_type == "relu" and op.x.op.op_type == "conv":
            return False
        return super()._match_pattern(block, op)


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_concat_split(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["split", "concat", "slice_by_index"]
        self._inputs_to_quantize = ["x", "values"]


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_pixel_shuffle(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["pixel_shuffle"]
        self._inputs_to_quantize = ["x"]


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_arithmetics(DummyQuantizeGraphPassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._match_ops = ["add", "mul", "sub", "round"]
        self._inputs_to_quantize = ["x", "y"]


@register_pass(namespace="mlvc", override=True)
class dummy_quantize_outputs(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_in_block(f)

    def _match_pattern(self, block, op):
        num_child_ops = sum([len(output.child_ops) for output in op.outputs])
        if num_child_ops != 0:
            return False
        if op.op_type == "dequantize":
            return False

        for output in op.outputs:
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["dummy_output_quant"]),
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["dummy_output_quant"]),
            ):
                quantize_op = mb.quantize(
                    input=output,
                    zero_point=np.int8(0),
                    scale=np.float32(1.0 / 128),
                    output_dtype="int8",
                    name=f"{output.op.name}_quantize",
                    scopes=op.scopes,
                )
                dequantize_op = mb.dequantize(
                    input=quantize_op,
                    zero_point=np.int8(0),
                    scale=np.float32(1.0 / 128),
                    name=f"{output.op.name}_dequantize",
                )
                op.enclosing_block.replace_block_output_var(
                    old_var=output,
                    new_var=dequantize_op,
                )
                output.name = f"{output.name}_tmp"
                return True
        return False

    @block_context_manager
    def _apply_in_block(self, block):
        def help_apply_in_block(block):
            for op in list(block.operations):
                if self._match_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply_in_block(block)


@register_pass(namespace="mlvc", override=True)
class gather_first(AbstractGraphPass):
    # Moves gather ops (usually requires CPU) to the beginning of the block,
    # so there is higher chance to run model on NPU

    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_in_block(f)

    def _find_parent_ops(self, block, op):
        if op is None:
            return []
        res = [op]
        for name in op.inputs:
            input_var = getattr(op, name)
            input_op = input_var.op
            res.extend(self._find_parent_ops(block, input_op))
        return res

    @block_context_manager
    def _apply_in_block(self, block):
        priority_ops = []
        ops = list(block.operations)
        for op in ops:
            if op.op_type == "gather":
                priority_ops.extend(self._find_parent_ops(block, op))
        priority_ops = set(priority_ops)

        sorted_ids = sorted(range(len(ops)), key=lambda i: (ops[i] not in priority_ops, i))
        if sorted_ids == list(range(len(ops))):
            return

        for op in ops:
            block.operations.remove(op)
        for i in sorted_ids:
            block.operations.insert_op_before(ops[i])


@register_pass(namespace="mlvc", override=True)
class split_gated_conv(AbstractGraphPass):
    """Split conv -> split patterns into N separate convolutions.

    This eliminates the split op and reduces peak tensor size.

    Pattern matched:
        conv(in_ch → N*out_ch, kernel=1x1) → split(axis=1, N outputs)
    Replaced with:
        conv_0 ... conv_N-1 (in_ch → out_ch, kernel=1x1 each)
    """

    max_splits = 2
    min_chunk_size = None

    # Skip splitting these convolutions to keep encoder and decoder consistent.
    # In the encoder the conv output has two consumers so the pattern doesn't match,
    # but in the decoder it does. Skipping avoids a mismatch between the two.
    skip_names = {"means"}

    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_in_block(f)

    def _match_and_replace_pattern(self, block, conv_op):
        if conv_op.op_type != "conv":
            return False

        # Must be a 1x1 pointwise convolution (no groups)
        weight = conv_op.weight.val
        if weight is None or len(weight.shape) != 4:
            return False
        if weight.shape[2] != 1 or weight.shape[3] != 1:
            return False
        out_channels = weight.shape[0]
        groups_val = conv_op.groups.val if conv_op.groups is not None else 1
        if groups_val != 1:
            return False

        if self.skip_names is not None:
            if any(substr in conv_op.name for substr in self.skip_names):
                return False

        # Must have exactly one consumer: a split along channels
        if len(conv_op.outputs) != 1:
            return False

        child_ops = list(conv_op.outputs[0].child_ops)
        if len(child_ops) != 1 or child_ops[0].op_type != "split":
            return False

        split_op = child_ops[0]
        if split_op.axis is not None and split_op.axis.val != 1:
            return False

        num_splits = len(split_op.outputs)
        if num_splits < 2 or out_channels % num_splits != 0:
            return False

        chunk = out_channels // num_splits
        if self.min_chunk_size is not None and chunk < self.min_chunk_size:
            return False

        if split_op.split_sizes is not None and split_op.split_sizes.val is not None:
            if any(s != chunk for s in split_op.split_sizes.val):
                return False

        max_splits = int(self.max_splits) if self.max_splits is not None else None
        if max_splits is not None and num_splits > max_splits:
            return False

        bias = conv_op.bias.val if conv_op.bias is not None and conv_op.bias.val is not None else None

        # Build N new conv ops
        def make_conv(i):
            w = weight[i * chunk : (i + 1) * chunk]
            kwargs = dict(x=conv_op.x, weight=w, name=f"{conv_op.name}_split_{chr(ord('a') + i)}", before_op=split_op)
            if bias is not None:
                kwargs["bias"] = bias[i * chunk : (i + 1) * chunk]
            return mb.conv(**kwargs)

        new_convs = [make_conv(i) for i in range(num_splits)]

        # Replace uses of split outputs with the new conv outputs
        for old_var, new_var in zip(split_op.outputs, new_convs):
            if not split_op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=split_op,
                old_var=old_var,
                new_var=new_var,
            ):
                return False

        block.remove_ops([conv_op, split_op])
        print(f"split_gated_conv: split conv '{conv_op.name}' ({out_channels} ch) into {num_splits} ({chunk} ch each)")
        return True

    @block_context_manager
    def _apply_in_block(self, block):
        def help_apply(block):
            for op in list(block.operations):
                if self._match_and_replace_pattern(block, op):
                    return True
            return False

        block_changed = True
        while block_changed:
            block_changed = help_apply(block)
