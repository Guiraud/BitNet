# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._inductor.runtime.runtime_utils import compile_mps_shader


# Topologically Sorted Source Nodes: [activation, abs_1, max_1, clamp_, scale, mul, round_1, clamp, activation_1, to], Original ATen: [aten._to_copy, aten.abs, aten.max, aten.clamp, aten.reciprocal, aten.mul, aten.round, aten.div]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   activation => convert_element_type
#   activation_1 => div
#   clamp => clamp_max, clamp_min_1
#   clamp_ => clamp_min
#   max_1 => max_1
#   mul => mul_1
#   round_1 => round_1
#   scale => mul, reciprocal
#   to => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type,), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%abs_1, -1, True), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%getitem, 1e-05), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 127), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %mul), kwargs = {})
#   %round_1 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_1,), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%round_1, -128), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 127), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max, %mul), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
mps_lib_0 = compile_mps_shader("""
    #include <c10/metal/random.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/utils.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device half* out_ptr1,
        constant half* in_ptr0,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int x0 = xindex;
        threadgroup float tmp_acc_0[1024];
        tmp_acc_0[r0_index] = ::metal::numeric_limits<float>::lowest();
        for(auto r0_1_cnt = 0; r0_1_cnt < 3; ++r0_1_cnt) {
            int r0_1 = 3 * r0_index + r0_1_cnt;
            if (r0_1 >= 2560) break;
            auto tmp0 = in_ptr0[r0_1 + 2560*x0];
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = metal::abs(tmp1);
            tmp_acc_0[r0_index] = ::c10::metal::max(tmp_acc_0[r0_index], tmp2);
        }
        auto tmp3 = c10::metal::threadgroup_max(tmp_acc_0, 1024);
        auto tmp4 = 1e-05;
        auto tmp5 = c10::metal::max(static_cast<decltype(tmp3+tmp4)>(tmp3), static_cast<decltype(tmp3+tmp4)>(tmp4));
        auto tmp6 = 1;
        auto tmp7 = tmp6 / tmp5;
        auto tmp8 = 127.0;
        auto tmp9 = tmp7 * tmp8;
        auto tmp10 = tmp1 * tmp9;
        auto tmp11 = metal::round(tmp10);
        auto tmp12 = -128.0;
        auto tmp13 = c10::metal::max(static_cast<decltype(tmp11+tmp12)>(tmp11), static_cast<decltype(tmp11+tmp12)>(tmp12));
        auto tmp14 = c10::metal::min(static_cast<decltype(tmp13+tmp8)>(tmp13), static_cast<decltype(tmp13+tmp8)>(tmp8));
        auto tmp15 = tmp14 / tmp9;
        auto tmp16 = static_cast<half>(tmp15);
        out_ptr1[r0_1 + 2560*x0] = static_cast<half>(tmp16);
    }
""")


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 41, 2560), (104960, 2560, 1))
    with torch._ops.contextlib.nullcontext():
        # MPS set device
        buf2 = empty_strided((1, 41, 2560), (104960, 2560, 1), device='mps', dtype=torch.float16)
        mps_lib_0.generated_kernel(buf2, arg0_1, threads=[41, 1024], group_size=[1, 1024])
        del arg0_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 41, 2560), (104960, 2560, 1), device='mps:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
