# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses
from .types import ModelOpProfile, ModelOpStats, ModelProfile, CoremlComputeUnits


def profile_coreml_model(
    model,
    coreml_compute_units: CoremlComputeUnits = CoremlComputeUnits.ALL,
) -> ModelProfile:
    import coremltools as ct

    def get_device_name(device):
        if isinstance(device, ct.models.compute_device.MLNeuralEngineComputeDevice):
            return "NPU"
        elif isinstance(device, ct.models.compute_device.MLCPUComputeDevice):
            return "CPU"
        elif isinstance(device, ct.models.compute_device.MLGPUComputeDevice):
            return "GPU"
        return type(device).__name__

    function_name = model.function_name or "main"
    compute_units = {
        CoremlComputeUnits.CPU: ct.ComputeUnit.CPU_ONLY,
        CoremlComputeUnits.GPU: ct.ComputeUnit.CPU_AND_GPU,
        CoremlComputeUnits.NPU: ct.ComputeUnit.CPU_AND_NE,
        CoremlComputeUnits.ALL: ct.ComputeUnit.ALL,
    }[coreml_compute_units]

    compute_plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=compute_units
    )
    if compute_plan.model_structure.program is None:
        raise ValueError("Unexpected model type.")

    res_operations: list[ModelOpProfile] = []
    program = compute_plan.model_structure.program
    operations = program.functions[function_name].block.operations
    for operation in operations:
        compute_device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
        estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(operation)
        output_name = operation.outputs[0].name

        op_profile = ModelOpProfile(
            operator_type=operation.operator_name,
            output_name=output_name,
            cost=estimated_cost.weight if estimated_cost is not None else float("nan"),
            preferred_compute_device="unknown",
        )
        if compute_device_usage is not None:
            op_profile = dataclasses.replace(
                op_profile,
                preferred_compute_device=get_device_name(compute_device_usage.preferred_compute_device),
                supported_compute_devices=[get_device_name(d) for d in compute_device_usage.supported_compute_devices],
            )
        res_operations.append(op_profile)

    return ModelProfile(
        operations=res_operations,
        op_stats=aggregate_op_profile(res_operations),
    )


def aggregate_op_profile(op_profile: list[ModelOpProfile]) -> ModelOpStats:
    num_ops_cpu = 0
    num_ops_npu = 0
    num_ops_gpu = 0
    cost_cpu = 0
    cost_npu = 0
    cost_gpu = 0
    for op in op_profile:
        if op.preferred_compute_device == "CPU":
            num_ops_cpu += 1
            cost_cpu += op.cost or 0
        elif op.preferred_compute_device == "NPU":
            num_ops_npu += 1
            cost_npu += op.cost or 0
        elif op.preferred_compute_device == "GPU":
            num_ops_gpu += 1
            cost_gpu += op.cost or 0
    return ModelOpStats(
        num_ops_cpu=num_ops_cpu,
        num_ops_npu=num_ops_npu,
        num_ops_gpu=num_ops_gpu,
        cost_cpu=cost_cpu,
        cost_npu=cost_npu,
        cost_gpu=cost_gpu,
    )
