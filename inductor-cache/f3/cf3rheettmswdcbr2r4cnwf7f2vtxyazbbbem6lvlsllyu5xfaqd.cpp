
#include "/Users/mguiraud/Documents/gitlab/BitNet/inductor-cache/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const uint8_t* in_ptr0,
                       float* out_ptr1)
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
                            tmp62.store(out_ptr1 + static_cast<int64_t>(x1 + 6912LL*x0));
                        }
                    }
                }
            }
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 2))
            throw std::runtime_error("requires 2 args");
        kernel(parse_arg<uint8_t*>(args, 0), parse_arg<float*>(args, 1)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
