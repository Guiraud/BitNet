# AOT ID: ['4_inference']
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


cpp_fused_abs_clamp_div_max_mul_reciprocal_round_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'const int64_t'], '''
#include "/Users/mguiraud/Documents/gitlab/BitNet/inductor-cache/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       const int64_t ks0)
{
    {
        {
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp1 = tmp0.abs();
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp1);
                    }
                    if(C10_UNLIKELY(x0 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x0 < static_cast<int64_t>(ks0)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                        auto tmp1 = tmp0.abs();
                        tmp_acc0_vec = max_masked_reduce(tmp_acc0_vec, tmp1, static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                    }
                }
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr0[static_cast<int64_t>(0LL)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = out_ptr0[static_cast<int64_t>(0LL)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = max_propagate_nan(tmp1, tmp2);
                    auto tmp4 = static_cast<int32_t>(1);
                    auto tmp5 = tmp4 / tmp3;
                    auto tmp6 = static_cast<float>(127.0);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp10 = tmp9.round();
                    auto tmp11 = static_cast<float>(-128.0);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = at::vec::maximum(tmp10, tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp6);
                    auto tmp15 = at::vec::minimum(tmp13, tmp14);
                    auto tmp16 = tmp15 / tmp8;
                    tmp16.store(out_ptr1 + static_cast<int64_t>(x0));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x0 < static_cast<int64_t>(ks0)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                    auto tmp1 = out_ptr0[static_cast<int64_t>(0LL)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = max_propagate_nan(tmp1, tmp2);
                    auto tmp4 = static_cast<int32_t>(1);
                    auto tmp5 = tmp4 / tmp3;
                    auto tmp6 = static_cast<float>(127.0);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp10 = tmp9.round();
                    auto tmp11 = static_cast<float>(-128.0);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = at::vec::maximum(tmp10, tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp6);
                    auto tmp15 = at::vec::minimum(tmp13, tmp14);
                    auto tmp16 = tmp15 / tmp8;
                    tmp16.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, 1, s0), (s0, s0, 1))
    buf0 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.float32)
    buf2 = empty_strided_cpu((1, 1, s0), (s0, s0, 1), torch.float32)
    cpp_fused_abs_clamp_div_max_mul_reciprocal_round_0(arg1_1, buf0, buf2, s0)
    del arg1_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2560
    arg1_1 = rand_strided((1, 1, 2560), (2560, 2560, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
