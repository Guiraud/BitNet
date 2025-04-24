# AOT ID: ['3_inference']
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


cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0 = async_compile.cpp_pybinding(['const uint8_t*', 'float*', 'const int64_t', 'const int64_t'], '''
#include "/Users/mguiraud/Documents/gitlab/BitNet/inductor-cache/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const uint8_t* in_ptr0,
                       float* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4LL*ks0); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(ks1); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4LL))))))
                        {
                            auto tmp0 = x0;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = 2LL*ks0;
                            auto tmp3 = c10::convert<int64_t>(tmp2);
                            auto tmp4 = tmp1 >= tmp3;
                            auto tmp5 = 3LL*ks0;
                            auto tmp6 = c10::convert<int64_t>(tmp5);
                            auto tmp7 = tmp1 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = [&]
                            {
                                auto tmp10 = at::vec::VecMask<float,1>::from(tmp8).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>(x1 + ks1*x0 + ((-2LL)*ks0*ks1)));
                                auto tmp11 = static_cast<uint8_t>(48);
                                auto tmp12 = at::vec::Vectorized<uint8_t>(tmp11);
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = static_cast<uint8_t>(4);
                                auto tmp15 = at::vec::Vectorized<uint8_t>(tmp14);
                                auto tmp16 = tmp13 >> tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp8 ? tmp9() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp18 = ks0;
                            auto tmp19 = c10::convert<int64_t>(tmp18);
                            auto tmp20 = tmp1 >= tmp19;
                            auto tmp21 = tmp1 < tmp3;
                            auto tmp22 = tmp20 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = at::vec::VecMask<float,1>::from(tmp22).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>(x1 + ks1*x0 + ((-1LL)*ks0*ks1)));
                                auto tmp25 = static_cast<uint8_t>(12);
                                auto tmp26 = at::vec::Vectorized<uint8_t>(tmp25);
                                auto tmp27 = tmp24 & tmp26;
                                auto tmp28 = static_cast<uint8_t>(2);
                                auto tmp29 = at::vec::Vectorized<uint8_t>(tmp28);
                                auto tmp30 = tmp27 >> tmp29;
                                return tmp30;
                            }
                            ;
                            auto tmp31 = tmp22 ? tmp23() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp32 = tmp1 < tmp19;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = at::vec::VecMask<float,1>::from(tmp32).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>(x1 + ks1*x0));
                                auto tmp35 = static_cast<uint8_t>(3);
                                auto tmp36 = at::vec::Vectorized<uint8_t>(tmp35);
                                auto tmp37 = tmp34 & tmp36;
                                auto tmp38 = static_cast<uint8_t>(0);
                                auto tmp39 = at::vec::Vectorized<uint8_t>(tmp38);
                                auto tmp40 = tmp37 >> tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp32 ? tmp33() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp42 = static_cast<uint8_t>(0);
                            auto tmp43 = at::vec::VecMask<float,1>::from(tmp32);
                            auto tmp44 = at::vec::Vectorized<uint8_t>(tmp42);
                            auto tmp45 = decltype(tmp41)::blendv(tmp44, tmp41, tmp43.template cast<uint8_t,1>());
                            auto tmp46 = at::vec::VecMask<float,1>::from(tmp22);
                            auto tmp47 = decltype(tmp31)::blendv(tmp45, tmp31, tmp46.template cast<uint8_t,1>());
                            auto tmp48 = at::vec::VecMask<float,1>::from(tmp8);
                            auto tmp49 = decltype(tmp17)::blendv(tmp47, tmp17, tmp48.template cast<uint8_t,1>());
                            auto tmp50 = tmp1 >= tmp6;
                            auto tmp51 = [&]
                            {
                                auto tmp52 = at::vec::VecMask<float,1>::from(tmp50).template loadu<uint8_t,1>(in_ptr0 + static_cast<int64_t>(x1 + ks1*x0 + ((-3LL)*ks0*ks1)));
                                auto tmp53 = static_cast<uint8_t>(192);
                                auto tmp54 = at::vec::Vectorized<uint8_t>(tmp53);
                                auto tmp55 = tmp52 & tmp54;
                                auto tmp56 = static_cast<uint8_t>(6);
                                auto tmp57 = at::vec::Vectorized<uint8_t>(tmp56);
                                auto tmp58 = tmp55 >> tmp57;
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp50 ? tmp51() : at::vec::Vectorized<uint8_t>(static_cast<uint8_t>(0));
                            auto tmp60 = at::vec::VecMask<float,1>::from(tmp50);
                            auto tmp61 = decltype(tmp59)::blendv(tmp49, tmp59, tmp60.template cast<uint8_t,1>());
                            auto tmp62 = at::vec::convert<float>(tmp61);
                            auto tmp63 = static_cast<float>(1.0);
                            auto tmp64 = at::vec::Vectorized<float>(tmp63);
                            auto tmp65 = tmp62 - tmp64;
                            tmp65.store(out_ptr1 + static_cast<int64_t>(x1 + ks1*x0));
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4LL)))) && x1 < static_cast<int64_t>(ks1)))
                        {
                            for (int64_t x1_tail = static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4LL))));x1_tail < static_cast<int64_t>(ks1); x1_tail++)
                            {
                                auto tmp0 = x0;
                                auto tmp1 = c10::convert<int64_t>(tmp0);
                                auto tmp2 = 2LL*ks0;
                                auto tmp3 = c10::convert<int64_t>(tmp2);
                                auto tmp4 = tmp1 >= tmp3;
                                auto tmp5 = 3LL*ks0;
                                auto tmp6 = c10::convert<int64_t>(tmp5);
                                auto tmp7 = tmp1 < tmp6;
                                auto tmp8 = tmp4 & tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = in_ptr0[static_cast<int64_t>(x1_tail + ks1*x0 + ((-2LL)*ks0*ks1))];
                                    auto tmp11 = static_cast<uint8_t>(48);
                                    auto tmp12 = decltype(tmp10)(tmp10 & tmp11);
                                    auto tmp13 = static_cast<uint8_t>(4);
                                    auto tmp14 =
                                    [&]()
                                    {
                                        constexpr decltype(tmp13) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                        if ((static_cast<std::make_signed_t<uint8_t>>(tmp13) < 0) || (tmp13 >= max_shift))
                                        {
                                            return decltype(tmp12)(tmp12 >> max_shift);
                                        }
                                        return decltype(tmp12)(tmp12 >> tmp13);
                                    }
                                    ()
                                    ;
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0);
                                auto tmp16 = ks0;
                                auto tmp17 = c10::convert<int64_t>(tmp16);
                                auto tmp18 = tmp1 >= tmp17;
                                auto tmp19 = tmp1 < tmp3;
                                auto tmp20 = tmp18 & tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<int64_t>(x1_tail + ks1*x0 + ((-1LL)*ks0*ks1))];
                                    auto tmp23 = static_cast<uint8_t>(12);
                                    auto tmp24 = decltype(tmp22)(tmp22 & tmp23);
                                    auto tmp25 = static_cast<uint8_t>(2);
                                    auto tmp26 =
                                    [&]()
                                    {
                                        constexpr decltype(tmp25) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                        if ((static_cast<std::make_signed_t<uint8_t>>(tmp25) < 0) || (tmp25 >= max_shift))
                                        {
                                            return decltype(tmp24)(tmp24 >> max_shift);
                                        }
                                        return decltype(tmp24)(tmp24 >> tmp25);
                                    }
                                    ()
                                    ;
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0);
                                auto tmp28 = tmp1 < tmp17;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = in_ptr0[static_cast<int64_t>(x1_tail + ks1*x0)];
                                    auto tmp31 = static_cast<uint8_t>(3);
                                    auto tmp32 = decltype(tmp30)(tmp30 & tmp31);
                                    auto tmp33 = static_cast<uint8_t>(0);
                                    auto tmp34 =
                                    [&]()
                                    {
                                        constexpr decltype(tmp33) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                        if ((static_cast<std::make_signed_t<uint8_t>>(tmp33) < 0) || (tmp33 >= max_shift))
                                        {
                                            return decltype(tmp32)(tmp32 >> max_shift);
                                        }
                                        return decltype(tmp32)(tmp32 >> tmp33);
                                    }
                                    ()
                                    ;
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0);
                                auto tmp36 = static_cast<uint8_t>(0);
                                auto tmp37 = tmp28 ? tmp35 : tmp36;
                                auto tmp38 = tmp20 ? tmp27 : tmp37;
                                auto tmp39 = tmp8 ? tmp15 : tmp38;
                                auto tmp40 = tmp1 >= tmp6;
                                auto tmp41 = [&]
                                {
                                    auto tmp42 = in_ptr0[static_cast<int64_t>(x1_tail + ks1*x0 + ((-3LL)*ks0*ks1))];
                                    auto tmp43 = static_cast<uint8_t>(192);
                                    auto tmp44 = decltype(tmp42)(tmp42 & tmp43);
                                    auto tmp45 = static_cast<uint8_t>(6);
                                    auto tmp46 =
                                    [&]()
                                    {
                                        constexpr decltype(tmp45) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                        if ((static_cast<std::make_signed_t<uint8_t>>(tmp45) < 0) || (tmp45 >= max_shift))
                                        {
                                            return decltype(tmp44)(tmp44 >> max_shift);
                                        }
                                        return decltype(tmp44)(tmp44 >> tmp45);
                                    }
                                    ()
                                    ;
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0);
                                auto tmp48 = tmp40 ? tmp47 : tmp39;
                                auto tmp49 = c10::convert<float>(tmp48);
                                auto tmp50 = static_cast<float>(1.0);
                                auto tmp51 = decltype(tmp49)(tmp49 - tmp50);
                                out_ptr1[static_cast<int64_t>(x1_tail + ks1*x0)] = tmp51;
                            }
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (s0, s1), (s1, 1))
    buf1 = empty_strided_cpu((4*s0, s1), (s1, 1), torch.float32)
    cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0(arg2_1, buf1, s0, s1)
    del arg2_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 640
    arg1_1 = 6912
    arg2_1 = rand_strided((640, 6912), (6912, 1), device='cpu', dtype=torch.uint8)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
