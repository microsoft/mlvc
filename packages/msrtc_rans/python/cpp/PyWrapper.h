// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @file python/cpp/PyWrapper.h
 * @brief Simplistic python extension API wrapper
 */

#pragma once

// ensure that assert is defined
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma push_macro("_DEBUG")
#undef _DEBUG
#include <Python.h>
#pragma pop_macro("_DEBUG")

#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef _MSC_VER
    #define PyW_MODINIT_FUNC PyMODINIT_FUNC
#else
// python headers assume that symbols are visible by default, correct them
    #define PyW_MODINIT_FUNC extern "C" __attribute__((visibility("default"))) PyObject*
#endif  // _MSC_VER

namespace PyWrapper {

// Marker to contruct a reference to new object
struct NewRef {};

// Simplistic implementation of reference counting pointer to python object
class PyWPtr {
public:
    PyWPtr(std::nullptr_t = nullptr) noexcept : m_ptr(nullptr) {}

    /// Instantiate pointer from \a ptr and retain it using AddRef()
    explicit PyWPtr(PyObject* ptr) : m_ptr(ptr) { Py_XINCREF(m_ptr); }

    /// Instantiate intrusive pointer from \a newly created ptr. Ownership goes to \e this.
    explicit PyWPtr(NewRef, PyObject* ptr) : m_ptr(ptr) {}

    /// Create PyWPtr from \a that. Ownership is shared
    PyWPtr(const PyWPtr& that) : m_ptr(that.Get()) { Py_XINCREF(m_ptr); }

    /// Move PyWPtr from \a that.
    PyWPtr(PyWPtr&& that) noexcept : m_ptr(that.Detach()) {}

    /// Destroy PyWPtr. If it is last owner, pointee will be deleted also
    ~PyWPtr() { Py_CLEAR(m_ptr); }

    /// Construct pointer to new object (stealing reference)
    static PyWPtr New(PyObject* ptr) { return PyWPtr{ NewRef(), ptr }; }

    /// Assigns nullptr
    PyWPtr& operator=(std::nullptr_t)
    {
        Reset();
        return *this;
    }

    /// Share pointee with \a that
    PyWPtr& operator=(const PyWPtr& that)
    {
        Reset(that.Get());
        return *this;
    }

    /// Share pointee with \a that
    PyWPtr& operator=(PyWPtr&& that) noexcept
    {
        if (this != &that) {
            Reset();
            m_ptr = that.Detach();
        }
        return *this;
    }

    /// Dereference old pointee and set new one \a ptr
    void Reset(PyObject* ptr = nullptr)
    {
        Py_XINCREF(ptr);
        auto* oldPtr = m_ptr;
        m_ptr = ptr;
        Py_XDECREF(oldPtr);
    }

    /// Dereference pointee
    void Release() { Reset(nullptr); }

    /// Disassociate this PyWPtr from the object it holds.
    /// Effectively, returns a raw pointer with an outstanding reference.
    PyObject* Detach()
    {
        auto ptr = m_ptr;
        m_ptr = nullptr;
        return ptr;
    }

    /// Return reference to pointee
    PyObject& operator*() const
    {
        assert(m_ptr != nullptr);
        return *m_ptr;
    }

    /// Return pointer to pointee
    PyObject* operator->() const
    {
        assert(m_ptr != nullptr);
        return m_ptr;
    }

    /// Return pointer to pointee
    PyObject* Get() const { return m_ptr; }
    /// Pointer conversion operator
    operator PyObject*() const { return Get(); }

    /// Is there a pointee?
    explicit operator bool() const { return m_ptr != nullptr; }

    /// Swap this with that. NOTE: NOT THREADSAFE.
    void swap(PyWPtr& that) noexcept { std::swap(m_ptr, that.m_ptr); }

private:
    PyObject* m_ptr;
};

bool operator==(const PyWPtr& a, const PyWPtr& b)
{
    return a.Get() == b.Get();
}

bool operator!=(const PyWPtr& a, const PyWPtr& b)
{
    return a.Get() != b.Get();
}

bool operator==(const PyWPtr& a, const std::nullptr_t& b)
{
    return a.Get() == b;
}

template <typename PyObject>
bool operator==(const std::nullptr_t& a, const PyWPtr& b)
{
    return b == a;
}

bool operator!=(const PyWPtr& a, const std::nullptr_t& b)
{
    return a.Get() != b;
}

bool operator!=(const std::nullptr_t& a, const PyWPtr& b)
{
    return b != a;
}

template <typename PyObject>
bool operator==(const PyWPtr& a, PyObject* b)
{
    return a.Get() == b;
}

template <typename PyObject>
bool operator!=(const PyWPtr& a, PyObject* b)
{
    return a.Get() != b;
}

template <typename PyObject>
bool operator==(PyObject* a, const PyWPtr& b)
{
    return a == b.Get();
}

bool operator!=(PyObject* a, const PyWPtr& b)
{
    return a != b.Get();
}

void swap(PyWPtr& a, PyWPtr& b)
{
    a.swap(b);
}

// Add a strong reference and return object itself
inline PyObject* PyW_NewRef(PyObject* obj)
{
    Py_XINCREF(obj);
    return obj;
}

// Wrapper around Py_buffer structure
class PyWBuffer {
public:
    PyWBuffer() { memset(&m_buffer, 0, sizeof(m_buffer)); }
    ~PyWBuffer() { Release(); }

    PyWBuffer(const PyWBuffer&) = delete;
    PyWBuffer(PyWBuffer&& other) noexcept : m_buffer(other.m_buffer) { memset(&other.m_buffer, 0, sizeof(m_buffer)); }

    bool IsNull() const { return !m_buffer.buf; }
    const Py_buffer& Get() const
    {
        assert(!IsNull());
        return m_buffer;
    }

    const Py_buffer* operator->() const { return &Get(); }
    const Py_buffer& operator*() const { return Get(); }
    operator bool() const { return !IsNull(); }

    const PyWBuffer& operator=(PyWBuffer&& other) noexcept
    {
        Release();

        m_buffer = other.m_buffer;
        memset(&other.m_buffer, 0, sizeof(m_buffer));
        return *this;
    }

    bool GetBuffer(PyObject* object, int flags)
    {
        Release();
        if (PyObject_GetBuffer(object, &m_buffer, flags) < 0) {
            m_buffer.buf = nullptr;
            return false;
        }
        return true;
    }

    void Release()
    {
        if (!IsNull()) {
            PyBuffer_Release(&m_buffer);
            m_buffer.buf = nullptr;
        }
    }

private:
    Py_buffer m_buffer;
};

class PyWRebindThreadGuard;
// Guard to save python thread state and release/reacquire GIL (global interpreter lock)
class PyWSavedThreadContext {
public:
    PyWSavedThreadContext() { m_state = PyEval_SaveThread(); }
    ~PyWSavedThreadContext()
    {
        assert(m_state);
        if (m_state) {
            PyEval_RestoreThread(m_state);
        }
    }

private:
    PyThreadState* m_state;

    friend class PyWRebindThreadGuard;
};

// Guard to temporary reacquire GIL released PyWAllowThreadsGuard
class PyWRebindThreadGuard {
public:
    PyWRebindThreadGuard(PyWSavedThreadContext* threadContext) : m_threadContext(nullptr)
    {
        if (threadContext) {
            auto state = threadContext->m_state;
            if (!state) {
                assert(state);
                throw std::runtime_error("GIL is already reacquired");
            }

            m_threadContext = threadContext;
            threadContext->m_state = nullptr;

            PyEval_RestoreThread(state);
        }
    }
    ~PyWRebindThreadGuard()
    {
        if (m_threadContext) {
            assert(!m_threadContext->m_state);
            m_threadContext->m_state = PyEval_SaveThread();
        }
    }

private:
    PyWSavedThreadContext* m_threadContext;
};

// Exception to carry python error
class PyWException : public std::exception {};

// starts handling of currently raised exception if any
void PyW_CatchAndHandleException()
{
    PyObject* type{ nullptr };
    PyObject* exception{ nullptr };
    PyObject* traceback{ nullptr };
    PyErr_Fetch(&type, &exception, &traceback);
    if (!type) {
        return;
    }
    PyErr_NormalizeException(&type, &exception, &traceback);
    if (exception && traceback) {
        PyException_SetTraceback(exception, traceback);
    }
    PyErr_SetExcInfo(type, exception, traceback);
}

namespace detail {

template <typename T>
struct MethodTrait {
    using ThisClass = void;
};

template <typename T, typename R, typename... Args>
struct MethodTrait<R (T::*)(Args... args)> {
    using ThisClass = T;
    using ReturnType = R;
    using ParameterTuple = std::tuple<Args...>;
};

template <typename T>
struct ExceptionReturn {};

template <>
struct ExceptionReturn<int> {
    static constexpr inline int value = -1;
};

template <>
struct ExceptionReturn<PyObject*> {
    static constexpr inline PyObject* value = nullptr;
};

template <typename R, typename F, R D = ExceptionReturn<R>::value>
R tryCatch(F&& f)
{
    try {
        return f();
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return D;
    } catch (const PyWException&) {
        // error is already set
        return D;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return D;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected c++ exception");
        return D;
    }
}

}  // namespace detail

// PyObject wrapper around C++ class
template <typename T>
class PyWBox {
    PyObject_HEAD
    std::aligned_storage_t<sizeof(T), std::alignment_of_v<T>> m_storage;

public:
    // Check that object wraps this C++ type
    static bool TypeCheck(PyObject* obj)
    {
        if (!obj) {
            return false;
        }
        if (!obj->ob_type) {
            return false;
        }
        return obj->ob_type->tp_new == tp_new;
    }

    /// Unwrap wrapped type
    T& Unwrap() noexcept { return *reinterpret_cast<T*>(&m_storage); }

    static T& Unwrap(PyObject* obj) noexcept
    {
        assert(TypeCheck(obj));
        return reinterpret_cast<PyWBox*>(obj)->Unwrap();
    }

    /// cast to python object
    PyObject* AsObject() noexcept { return &ob_base; }
    operator PyObject*() noexcept { return AsObject(); }

    /// Get box from wrapped object
    static PyWBox& FromWrapped(T& wrapped) noexcept
    {
        return *reinterpret_cast<PyWBox*>(reinterpret_cast<char*>(&wrapped) - offsetof(PyWBox, m_storage));
    }

    // Make tp_new type object slot
    static constexpr PyType_Slot MakeTypeNewSlot() { return { Py_tp_new, reinterpret_cast<void*>(tp_new) }; }
    // Make tp_dealloc type object slot
    static constexpr PyType_Slot MakeTypeDeallocSlot()
    {
        return { Py_tp_dealloc, reinterpret_cast<void*>(tp_dealloc) };
    }
    // Make tp_init type object slot
    template <int (T::*Method)(PyObject* args, PyObject* kwargs)>
    static constexpr PyType_Slot MakeTypeInitSlot()
    {
        return { Py_tp_init, reinterpret_cast<void*>(WrapMethod<Method>()) };
    }

    // Make python callable wrapper around C++ method
    template <auto Method>
    static constexpr auto WrapMethod()
    {
        using MethodTrait = detail::MethodTrait<decltype(Method)>;
        static_assert(std::is_same_v<typename MethodTrait::ThisClass, T>, "invalid method class");
        return wrapMethodImpl<Method, typename MethodTrait::ReturnType, typename MethodTrait::ParameterTuple>(
            std::make_index_sequence<std::tuple_size<typename MethodTrait::ParameterTuple>::value>());
    }

    // Make python method definition from C++ method without args
    template <PyObject* (T::*Method)()>
    static constexpr PyMethodDef MakeMethodDef(const char* name, const char* doc = nullptr)
    {
        return { name, reinterpret_cast<PyCFunction>(WrapMethod<Method>()), METH_NOARGS, doc };
    }

    // Make python method definition from C++ method using positional args only
    template <PyObject* (T::*Method)(PyObject* args)>
    static constexpr PyMethodDef MakeMethodDef(const char* name, const char* doc = nullptr)
    {
        return { name, WrapMethod<Method>(), METH_VARARGS, doc };
    }

    // Make python method definition from C++ method using positional and keyword args
    template <PyObject* (T::*Method)(PyObject* args, PyObject* kwargs)>
    static constexpr PyMethodDef MakeMethodDef(const char* name, const char* doc = nullptr)
    {
        return { name, reinterpret_cast<PyCFunction>(WrapMethod<Method>()), METH_VARARGS | METH_KEYWORDS, doc };
    }

private:
    PyWBox() = delete;
    PyWBox(const PyWBox&) = delete;
    ~PyWBox() = delete;

    PyWBox& operator=(const PyWBox&) = delete;

    static PyObject* tp_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds)
    {
        (void)args;
        (void)kwds;

        auto self = subtype->tp_alloc(subtype, 0);
        if (!self) {
            return nullptr;
        }
        auto& box = *reinterpret_cast<PyWBox*>(self);
        if constexpr (noexcept(new (&box.Unwrap()) T())) {
            new (&box.Unwrap()) T();
            return self;
        } else {
            return detail::tryCatch<PyObject*>([self, &box]() {
                new (&box.Unwrap()) T();
                return self;
            });
        }
    }

    static void tp_dealloc(PyObject* self)
    {
        auto& box = *reinterpret_cast<PyWBox*>(self);
        static_assert(noexcept(box.Unwrap().~T()), "Object destructor must be noexcept");
        box.Unwrap().T::~T();
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    template <auto Method, typename R, typename P, size_t... S>
    static constexpr auto wrapMethodImpl(std::index_sequence<S...>)
    {
        return function<Method, R, std::tuple_element_t<S, P>...>;
    }

    template <auto Method, typename R, typename... Args>
    static R function(PyObject* self, Args... args)
    {
        auto& box = *reinterpret_cast<PyWBox*>(self);
        if constexpr (noexcept((box.Unwrap().*Method)(args...))) {
            return (box.Unwrap().*Method)(args...);
        } else {
            return detail::tryCatch<R>([&box, &args...]() { return (box.Unwrap().*Method)(args...); });
        }
    }
};

// Make python callable wrapper around C++ method
template <auto Method>
constexpr inline auto WrapMethod()
{
    using MethodClass = typename detail::MethodTrait<decltype(Method)>::ThisClass;
    static_assert(!std::is_same_v<MethodClass, void>, "invalid method signature");
    return PyWBox<MethodClass>::template WrapMethod<Method>();
}

// Make tp_new type object slot
template <typename T>
static constexpr PyType_Slot MakeTypeNewSlot()
{
    return PyWBox<T>::MakeTypeNewSlot();
}

// Make tp_dealloc type object slot
template <typename T>
static constexpr PyType_Slot MakeTypeDeallocSlot()
{
    return PyWBox<T>::MakeTypeDeallocSlot();
}

// Make tp_init type object slot
template <auto Method>
static constexpr PyType_Slot MakeTypeInitSlot()
{
    using MethodClass = typename detail::MethodTrait<decltype(Method)>::ThisClass;
    static_assert(!std::is_same_v<MethodClass, void>, "invalid method signature");
    return PyWBox<MethodClass>::template MakeTypeInitSlot<Method>();
}

// Make python method definition from C++ method
template <auto Method>
constexpr inline PyMethodDef MakeMethodDef(const char* name, const char* doc = nullptr)
{
    using MethodClass = typename detail::MethodTrait<decltype(Method)>::ThisClass;
    static_assert(!std::is_same_v<MethodClass, void>, "invalid method signature");
    return PyWBox<MethodClass>::template MakeMethodDef<Method>(name, doc);
}

// Add extension type to the module
inline bool AddPyType(PyObject* module, PyType_Spec& typeSpec)
{
    auto typeObject = PyWPtr::New(PyType_FromSpec(&typeSpec));
    if (!typeObject) {
        return false;
    }
    if (PyType_Ready(reinterpret_cast<PyTypeObject*>(typeObject.Get())) < 0) {
        return false;
    }
    auto name = strrchr(typeSpec.name, '.');
    if (name) {
        name++;
    } else {
        name = typeSpec.name;
    }
    if (PyModule_AddObject(module, name, typeObject) < 0) {
        return false;
    }
    typeObject.Detach();
    return true;
}

}  // namespace PyWrapper
