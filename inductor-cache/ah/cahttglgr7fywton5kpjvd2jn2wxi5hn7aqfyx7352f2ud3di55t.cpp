
#include "/Users/mguiraud/Documents/gitlab/BitNet/inductor-cache/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       const int64_t ks0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(34LL); x0+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(ks0); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + ks0*x0), static_cast<int64_t>(4));
                                auto tmp1 = tmp0.abs();
                                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp1);
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x1 < static_cast<int64_t>(ks0)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + ks0*x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                                auto tmp1 = tmp0.abs();
                                tmp_acc0_vec = max_masked_reduce(tmp_acc0_vec, tmp1, static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                            }
                        }
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(ks0); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + ks0*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
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
                            tmp16.store(out_ptr1 + static_cast<int64_t>(x1 + ks0*x0));
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x1 < static_cast<int64_t>(ks0)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + ks0*x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
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
                            tmp16.store(out_ptr1 + static_cast<int64_t>(x1 + ks0*x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
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
        if(unlikely(PyTuple_GET_SIZE(args) != 4))
            throw std::runtime_error("requires 4 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<int64_t>(args, 3)); Py_RETURN_NONE;
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
