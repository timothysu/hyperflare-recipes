# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared tensor-scaled FP8 projections with checkpoint activation calibration."""

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.rebuild_manifest import weight_container
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare


_SCALED_MM_MIN_ROWS = 128


@torch.library.custom_op("flare::tensor_fp8_scaled_mm_out", mutates_args=("out",))
def _scaled_mm_out(source: torch.Tensor, weight: torch.Tensor, unit_scale: torch.Tensor,
                   output_scale: torch.Tensor, out: torch.Tensor) -> None:
    torch._scaled_mm(source, weight.t(), scale_a=unit_scale, scale_b=output_scale,
                     out_dtype=out.dtype, out=out)


@_scaled_mm_out.register_fake
def _fake_scaled_mm_out(source, weight, unit_scale, output_scale, out):
    pass


@weight_container(('packed', 'input_scale'))
@dataclass(frozen=True)
class TensorFP8Weight:
    packed: object
    input_scale: torch.Tensor

    @property
    def in_features(self) -> int:
        return self.packed.in_features

    @property
    def out_features(self) -> int:
        return self.packed.out_features


class TensorFP8Projection:
    def __init__(self, source, weight, output, *, session, name, staging=None):
        if (source.ndim != 2 or source.dtype != torch.bfloat16
                or source.shape[1] != weight.packed.in_features
                or output.shape != (source.shape[0], weight.packed.out_features)
                or output.dtype != source.dtype or output.device != source.device):
            raise ValueError("tensor-FP8 projection requires matching BF16 input and output matrices")
        self.weight, self.output = weight, output
        for tensor in (weight.packed.values, weight.packed.scale_mma, weight.packed.block_scale,
                       weight.packed.output_scale, weight.input_scale):
            record_numeric_tensor(tensor)
        if staging is None:
            self.quantized = torch.empty(source.shape, dtype=torch.float8_e4m3fn, device=source.device)
        else:
            if (staging.dtype != torch.uint8 or staging.device != source.device
                    or not staging.is_contiguous() or staging.numel() < source.numel()):
                raise ValueError("tensor-FP8 staging requires sufficient contiguous device bytes")
            self.quantized = staging.flatten()[:source.numel()].view(torch.float8_e4m3fn).view(source.shape)
        rows = source.shape[0]
        self.use_scaled_mm = (
            rows >= _SCALED_MM_MIN_ROWS
            and rows % 16 == 0
            and weight.packed.in_features == weight.packed.padded_in_features
            and weight.packed.in_features % 16 == 0
            and weight.packed.out_features % 16 == 0
            and self.quantized.is_contiguous()
            and output.is_contiguous()
        )
        if self.use_scaled_mm:
            self.unit_scale = persistent_copy(torch.ones((1,), dtype=torch.float32, device=source.device))
            record_numeric_tensor(self.unit_scale)
            self.plan = None
            return
        self.unit_scale = None
        self.plan = blockscaled.plan(blockscaled.query_from_call(self.quantized, weight.packed))
        quantized = self.quantized

        def prepare(state):
            quantized.zero_()
            w = weight.packed
            return PreparedCall(run=lambda: state.run_tensor_fp8(
                quantized, w.values, w.scale_mma, w.block_scale, w.output_scale,
                out_dtype=output.dtype, stream=None), owners=(weight,))

        def benchmark(state):
            inputs = benchmark_resource(session, ("tensor_fp8", name), lambda:
                torch.randn(source.shape, device=source.device).clamp(-4, 4).to(torch.float8_e4m3fn))
            w = weight.packed
            return PreparedCall(run=lambda: state.run_tensor_fp8(
                inputs, w.values, w.scale_mma, w.block_scale, w.output_scale,
                out_dtype=output.dtype, stream=None), owners=(weight,))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))

    def functional(self, *, session):
        """Expose calibrated activation and output storage to forward compilation."""
        del session
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.quantized_layout = (tuple(self.quantized.shape), self.quantized.stride(), self.quantized.dtype)
            self.output = self.quantized = None
        return self

    def __call__(self, source):
        output, quantized = self.output, self.quantized
        if output is None:
            quantized = (source.float() / self.weight.input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            if self.use_scaled_mm:
                shape, stride, dtype = self.output_layout
                output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
                _scaled_mm_out(quantized, self.weight.packed.values, self.unit_scale,
                               self.weight.packed.output_scale, output)
                return output
            return blockscaled.mm(quantized, self.weight.packed, plan=self.plan)
        quantized.copy_((source.float() / self.weight.input_scale).clamp(-448, 448))
        if self.use_scaled_mm:
            _scaled_mm_out(quantized, self.weight.packed.values, self.unit_scale,
                           self.weight.packed.output_scale, output)
        else:
            output.copy_(blockscaled.mm(quantized, self.weight.packed, plan=self.plan))
        return output
