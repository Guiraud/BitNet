# AOT ID: ['0_inference']
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


cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0 = async_compile.cpp_pybinding(['const uint8_t*', 'half*'], '''
#include "/Users/mguiraud/Documents/gitlab/BitNet/inductor-cache/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const uint8_t* in_ptr0,
                       half* out_ptr1)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2560LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(6912LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(6912LL)))
                        {
                            auto tmp0 = x0;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(1280);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<int64_t>(1920);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = tmp3 & tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = at::vec::VecMask<float,1>::from(tmp6).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>((-8847360LL) + x1 + 6912LL*x0));
                                auto tmp9 = static_cast<uint8_t>(48);
                                auto tmp10 = at::vec::Vectorized<uint8_t>(tmp9);
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = static_cast<uint8_t>(4);
                                auto tmp13 = at::vec::Vectorized<uint8_t>(tmp12);
                                auto tmp14 = tmp11 >> tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp6 ? tmp7() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp16 = static_cast<int64_t>(640);
                            auto tmp17 = tmp1 >= tmp16;
                            auto tmp18 = tmp1 < tmp2;
                            auto tmp19 = tmp17 & tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = at::vec::VecMask<float,1>::from(tmp19).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>((-4423680LL) + x1 + 6912LL*x0));
                                auto tmp22 = static_cast<uint8_t>(12);
                                auto tmp23 = at::vec::Vectorized<uint8_t>(tmp22);
                                auto tmp24 = tmp21 & tmp23;
                                auto tmp25 = static_cast<uint8_t>(2);
                                auto tmp26 = at::vec::Vectorized<uint8_t>(tmp25);
                                auto tmp27 = tmp24 >> tmp26;
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp19 ? tmp20() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp29 = tmp1 < tmp16;
                            auto tmp30 = [&]
                            {
                                auto tmp31 = at::vec::VecMask<float,1>::from(tmp29).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>(x1 + 6912LL*x0));
                                auto tmp32 = static_cast<uint8_t>(3);
                                auto tmp33 = at::vec::Vectorized<uint8_t>(tmp32);
                                auto tmp34 = tmp31 & tmp33;
                                auto tmp35 = static_cast<uint8_t>(0);
                                auto tmp36 = at::vec::Vectorized<uint8_t>(tmp35);
                                auto tmp37 = tmp34 >> tmp36;
                                return tmp37;
                            }
                            ;
                            auto tmp38 = tmp29 ? tmp30() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp39 = static_cast<uint8_t>(0);
                            auto tmp40 = at::vec::VecMask<float,1>::from(tmp29);
                            auto tmp41 = at::vec::Vectorized<uint8_t>(tmp39);
                            auto tmp42 = decltype(tmp38)::blendv(tmp41, tmp38, tmp40.template cast<uint8_t,1>());
                            auto tmp43 = at::vec::VecMask<float,1>::from(tmp19);
                            auto tmp44 = decltype(tmp28)::blendv(tmp42, tmp28, tmp43.template cast<uint8_t,1>());
                            auto tmp45 = at::vec::VecMask<float,1>::from(tmp6);
                            auto tmp46 = decltype(tmp15)::blendv(tmp44, tmp15, tmp45.template cast<uint8_t,1>());
                            auto tmp47 = tmp1 >= tmp4;
                            auto tmp48 = [&]
                            {
                                auto tmp49 = at::vec::VecMask<float,1>::from(tmp47).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>((-13271040LL) + x1 + 6912LL*x0));
                                auto tmp50 = static_cast<uint8_t>(192);
                                auto tmp51 = at::vec::Vectorized<uint8_t>(tmp50);
                                auto tmp52 = tmp49 & tmp51;
                                auto tmp53 = static_cast<uint8_t>(6);
                                auto tmp54 = at::vec::Vectorized<uint8_t>(tmp53);
                                auto tmp55 = tmp52 >> tmp54;
                                return tmp55;
                            }
                            ;
                            auto tmp56 = tmp47 ? tmp48() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp57 = at::vec::VecMask<float,1>::from(tmp47);
                            auto tmp58 = decltype(tmp56)::blendv(tmp46, tmp56, tmp57.template cast<uint8_t,1>());
                            auto tmp59 = at::vec::convert<float>(tmp58);
                            auto tmp60 = static_cast<float>(1.0);
                            auto tmp61 = at::vec::Vectorized<float>(tmp60);
                            auto tmp62 = tmp59 - tmp61;
                            auto tmp63 = at::vec::convert<half>(tmp62);
                            tmp63.store(out_ptr1 + static_cast<int64_t>(x1 + 6912LL*x0), static_cast<int64_t>(4));
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (640, 6912), (6912, 1))
    buf1 = empty_strided_cpu((2560, 6912), (6912, 1), torch.float16)
    cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0(arg0_1, buf1)
    del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((640, 6912), (6912, 1), device='cpu', dtype=torch.uint8)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
