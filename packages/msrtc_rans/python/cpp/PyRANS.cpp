// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "PyWrapper.h"
#include <atomic>
#include <cstdio>
#include <msrtc_rans/EntropyCoder.h>

using namespace PyWrapper;

// translate rANS variant argument
static bool checkVariant(int value, msrtc_rans::RansVariant& variant)
{
    switch (value) {
    case static_cast<int>(msrtc_rans::RansVariant::RansByte):
    case static_cast<int>(msrtc_rans::RansVariant::Rans64):
        variant = static_cast<msrtc_rans::RansVariant>(value);
        return true;
    default:
        return false;
    }
}

// rANS encoder stream
class RansEncoderStream : private msrtc_rans::IResizableBuffer {
public:
    int Init(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "variant", "initialSize", "maxSizeStep", nullptr };

        int variantArg = static_cast<int>(msrtc_rans::RansVariant::RansByte);
        Py_ssize_t initialSizeArg = 4096;
        Py_ssize_t maxSizeStepArg = 1024 * 1024;

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "|inn", const_cast<char**>(keywords),  //
                                              &variantArg, &initialSizeArg, &maxSizeStepArg);
        if (!rc) {
            return -1;
        }

        msrtc_rans::RansVariant variant;
        if (!checkVariant(variantArg, variant)) {
            PyErr_SetString(PyExc_ValueError, "unknown rANS variant value");
            return -1;
        }

        size_t initialSize = static_cast<size_t>(std::max<Py_ssize_t>(initialSizeArg, 0));
        size_t maxSizeStep = static_cast<size_t>(std::max<Py_ssize_t>(maxSizeStepArg, 0));

        initialSize = std::max(AlignSize(initialSize, true), s_MinBufferSize);
        maxSizeStep = std::max(AlignSize(maxSizeStep, false), s_MinBufferSize);

        m_bufferView = createBuffer(initialSize, m_buffer);
        if (!m_bufferView) {
            return -1;
        }
        m_maxSizeStep = maxSizeStep;
        auto e = m_impl.Initialize(variant, *this);
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return -1;
        }
        return 0;
    }

    PyObject* Flush()
    {
        auto data = GetImpl().Flush();
        if (data.is_empty()) {
            auto e = msrtc_rans::make_error_code(msrtc_rans::error::invalid_state);
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return nullptr;
        }
        return getSpan(data).Detach();
    }

    PyObject* Reset()
    {
        GetImpl().Flush(true);
        Py_RETURN_NONE;
    }

    // Associate thread context (PyWAllowThreadsGuard) to handle python callbacks
    class ThreadContextGuard {
    public:
        ThreadContextGuard(RansEncoderStream& owner, PyWSavedThreadContext& currentThread)
            : m_owner(owner), m_currentThread(&currentThread)
        {
            PyWSavedThreadContext* expected = nullptr;
            if (!m_owner.m_currentThread.compare_exchange_strong(expected, m_currentThread)) {
                throw std::runtime_error("concurrent usage of Resizable buffer is detected");
            }
        }

        ~ThreadContextGuard()
        {
            PyWSavedThreadContext* expected = m_currentThread;
            if (!m_owner.m_currentThread.compare_exchange_strong(expected, nullptr)) {
                assert(false);
            }
        }

    private:
        RansEncoderStream& m_owner;
        PyWSavedThreadContext* m_currentThread;
    };

    msrtc_rans::RansEncoderStream& GetImpl() { return m_impl; }

private:
    msrtc_rans::RansEncoderStream m_impl;

    PyWPtr m_bufferView;
    PyWBuffer m_buffer;
    PyWPtr m_newBufferView;
    PyWBuffer m_newBuffer;
    size_t m_maxSizeStep;

    std::atomic<PyWSavedThreadContext*> m_currentThread;

    // Get current buffer
    virtual msrtc_rans::span<std::byte> GetBuffer() override
    {
        assert(!m_buffer.IsNull());
        return { reinterpret_cast<std::byte*>(m_buffer->buf), static_cast<size_t>(m_buffer->len) };
    }
    // Begin grow operation and return a new buffer
    virtual msrtc_rans::span<std::byte> BeginToGrow() override
    {
        PyWRebindThreadGuard guard(m_currentThread);

        auto currentSize = static_cast<size_t>(m_buffer->len);
        auto newSize = currentSize + std::min(currentSize, m_maxSizeStep);
        m_newBufferView = createBuffer(newSize, m_newBuffer);
        if (!m_newBufferView) {
            throw PyWException();
        }
        assert(!m_newBuffer.IsNull());
        return { reinterpret_cast<std::byte*>(m_newBuffer->buf), static_cast<size_t>(m_newBuffer->len) };
    }
    // Complete active change (i.e. grow operation)
    virtual void Commit() override
    {
        PyWRebindThreadGuard guard(m_currentThread);

        if (m_newBufferView) {
            m_bufferView = std::move(m_newBufferView);
            m_buffer = std::move(m_newBuffer);
        }
    }
    // Rollback active change (i.e. grow operation)
    virtual void Rollback() override
    {
        PyWRebindThreadGuard guard(m_currentThread);

        if (m_newBufferView) {
            m_newBuffer.Release();
            m_newBufferView.Release();
        }
    }

    PyWPtr createBuffer(size_t size, PyWBuffer& buffer)
    {
        buffer.Release();

        if (size > static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max())) {
            PyErr_NoMemory();
            return {};
        }
        auto args = PyWPtr::New(Py_BuildValue("(n)", static_cast<Py_ssize_t>(size)));
        if (!args) {
            return {};
        }
        auto byteArray = PyWPtr::New(PyObject_CallObject(reinterpret_cast<PyObject*>(&PyByteArray_Type), args));
        if (!byteArray) {
            return {};
        }
        auto bufferView = PyWPtr::New(PyMemoryView_FromObject(byteArray));
        if (!buffer.GetBuffer(bufferView, PyBUF_SIMPLE | PyBUF_WRITABLE)) {
            return {};
        }
        assert(static_cast<size_t>(buffer->len) == size);
        return bufferView;
    }

    // Get memory view for resulting span object
    PyWPtr getSpan(const msrtc_rans::span<const std::byte>& value)
    {
        if (value.is_empty()) {
            return PyWPtr(Py_None);
        }
        assert(!m_buffer.IsNull());

        auto offset = value.data() - reinterpret_cast<std::byte*>(m_buffer->buf);
        assert(offset >= 0 && offset < m_buffer->len);

        assert(value.size() <= static_cast<size_t>(m_buffer->len - offset));
        return PyWPtr::New(PySequence_GetSlice(m_bufferView, offset, offset + static_cast<Py_ssize_t>(value.size())));
    }
};

static PyMethodDef s_ransEncoderStreamMethods[] = {  //
    MakeMethodDef<&RansEncoderStream::Flush>("flush"),
    MakeMethodDef<&RansEncoderStream::Reset>("reset"),
    { 0 }
};

static PyType_Slot s_ransEncoderStreamSlots[] = {  //
    MakeTypeNewSlot<RansEncoderStream>(),
    MakeTypeDeallocSlot<RansEncoderStream>(),
    MakeTypeInitSlot<&RansEncoderStream::Init>(),
    { Py_tp_methods, s_ransEncoderStreamMethods },
    { 0 }
};

static PyType_Spec s_ransEncoderStreamSpec = {  //
    "msrtc.rans.RansEncoderStream", sizeof(PyWBox<RansEncoderStream>), 0, Py_TPFLAGS_DEFAULT, s_ransEncoderStreamSlots
};

static bool isInt32Array(const Py_buffer& buffer, bool isUnsigned)
{
    if (buffer.ndim != 1) {
        return false;
    }
    if (!buffer.format || !buffer.shape) {
        return false;
    }
    if (isUnsigned) {
        if (buffer.itemsize != static_cast<Py_ssize_t>(sizeof(uint32_t))
            || (strcmp(buffer.format, "I") && strcmp(buffer.format, "L"))) {
            return false;
        }
    } else {
        if (buffer.itemsize != static_cast<Py_ssize_t>(sizeof(int32_t))
            || (strcmp(buffer.format, "i") && strcmp(buffer.format, "l"))) {
            return false;
        }
    }
    return true;
}

class EntropyEncoder {
public:
    int Init(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "pmfLengths", "pmfOffsets", "pmfTable", "variant",
                                          "symbolBits", "bypassBits", nullptr };

        PyObject* pmfLengths{ nullptr };
        PyObject* pmfOffsets{ nullptr };
        PyObject* pmfTable{ nullptr };
        int variantArg = 0;
        unsigned symbolBits = 0;
        unsigned bypassBits = 0;

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiII", const_cast<char**>(keywords),  //
                                              &pmfLengths, &pmfOffsets, &pmfTable, &variantArg, &symbolBits, &bypassBits);
        if (!rc) {
            return -1;
        }
        PyWBuffer pmfLengthsBuf;
        if (!pmfLengthsBuf.GetBuffer(pmfLengths, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfLengths must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfLengthsBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfLengths shape or data type, expected int32 1-d array");
            return -1;
        }
        PyWBuffer pmfOffsetsBuf;
        if (!pmfOffsetsBuf.GetBuffer(pmfOffsets, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfOffsets must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfOffsetsBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfOffsets shape or data type, expected int32 1-d array");
            return -1;
        }
        PyWBuffer pmfTableBuf;
        if (!pmfTableBuf.GetBuffer(pmfTable, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfTable must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfTableBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfTable shape or data type, expected int32 1-d array");
            return -1;
        }
        msrtc_rans::RansVariant variant;
        if (!checkVariant(variantArg, variant)) {
            PyErr_SetString(PyExc_ValueError, "unknown rANS variant value");
            return -1;
        }
        std::error_code e;
        {
            PyWSavedThreadContext threadsGuard;
            e = m_encoder.Initialize(
                variant,
                { reinterpret_cast<const int32_t*>(pmfLengthsBuf->buf), static_cast<size_t>(pmfLengthsBuf->shape[0]) },
                { reinterpret_cast<const int32_t*>(pmfOffsetsBuf->buf), static_cast<size_t>(pmfOffsetsBuf->shape[0]) },
                { reinterpret_cast<const int32_t*>(pmfTableBuf->buf), static_cast<size_t>(pmfTableBuf->shape[0]) },
                symbolBits, bypassBits);
        }
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return -1;
        }
        return 0;
    }

    PyObject* Encode(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "stream", "indices", "values", nullptr };

        PyObject* stream{ nullptr };
        PyObject* indices{ nullptr };
        PyObject* values{ nullptr };

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", const_cast<char**>(keywords),  //
                                              &stream, &indices, &values);
        if (!rc) {
            return nullptr;
        }
        if (!PyWBox<RansEncoderStream>::TypeCheck(stream)) {
            PyErr_SetString(PyExc_TypeError, "argument stream must be RansEncoderStream");
            return nullptr;
        }
        PyWBuffer indicesBuf;
        if (!indicesBuf.GetBuffer(indices, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "indices must be an int32 1-d array");
            return nullptr;
        }
        if (!isInt32Array(*indicesBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid indices shape or data type, expected int32 1-d array");
            return nullptr;
        }
        PyWBuffer valuesBuf;
        if (!valuesBuf.GetBuffer(values, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "values must be an int32 1-d array");
            return nullptr;
        }
        if (!isInt32Array(*valuesBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid values shape or data type, expected int32 1-d array");
            return nullptr;
        }
        auto& unwrappedStream = PyWBox<RansEncoderStream>::Unwrap(stream);
        std::error_code e;
        {
            // Allow threads
            PyWSavedThreadContext savedThreadContext;
            // Setup current thread pointer for stream
            RansEncoderStream::ThreadContextGuard bindThreadToStreamGuard(unwrappedStream, savedThreadContext);

            e = m_encoder.Encode(
                unwrappedStream.GetImpl(),
                { reinterpret_cast<const int32_t*>(indicesBuf->buf), static_cast<size_t>(indicesBuf->shape[0]) },
                { reinterpret_cast<const int32_t*>(valuesBuf->buf), static_cast<size_t>(valuesBuf->shape[0]) });
        }
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return nullptr;
        }
        Py_RETURN_NONE;
    }

private:
    msrtc_rans::EntropyEncoder m_encoder;
};

static PyMethodDef s_entropyEncoderMethods[] = {  //
    MakeMethodDef<&EntropyEncoder::Encode>("encode"),
    { 0 }
};

static PyType_Slot s_entropyEncoderSlots[] = {  //
    MakeTypeNewSlot<EntropyEncoder>(),
    MakeTypeDeallocSlot<EntropyEncoder>(),
    MakeTypeInitSlot<&EntropyEncoder::Init>(),
    { Py_tp_methods, s_entropyEncoderMethods },
    { 0 }
};

static PyType_Spec s_entropyEncoderSpec = {  //
    "msrtc.rans.EntropyEncoder", sizeof(PyWBox<EntropyEncoder>), 0, Py_TPFLAGS_DEFAULT, s_entropyEncoderSlots
};

// rANS decoder stream
class RansDecoderStream {
public:
    int Init(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "data", "variant", nullptr };

        int variantArg = static_cast<int>(msrtc_rans::RansVariant::RansByte);
        PyObject* data{ nullptr };

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", const_cast<char**>(keywords),  //
                                              &data, &variantArg);
        if (!rc) {
            return -1;
        }

        msrtc_rans::RansVariant variant;
        if (!checkVariant(variantArg, variant)) {
            PyErr_SetString(PyExc_ValueError, "unknown rANS variant value");
            return -1;
        }

        auto e = m_impl.Initialize(variant);
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return -1;
        }
        if (data && !open(data)) {
            return -1;
        }
        return 0;
    }

    PyObject* Open(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "data", nullptr };

        PyObject* data{ nullptr };

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords), &data);
        if (!rc) {
            return nullptr;
        }
        if (!open(data)) {
            return nullptr;
        }
        Py_RETURN_NONE;
    }

    PyObject* Close()
    {
        GetImpl().Close();
        m_dataBuf.Release();
        Py_RETURN_NONE;
    }

    PyObject* IsOpen()
    {
        if (GetImpl().IsOpen()) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    }

    PyObject* DecodeEOF()
    {
        if (!GetImpl().CheckEOF()) {
            auto e = msrtc_rans::make_error_code(msrtc_rans::error::invalid_stream);
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return nullptr;
        }

        return Close();
    }

    msrtc_rans::RansDecoderStream& GetImpl() { return m_impl; }

private:
    msrtc_rans::RansDecoderStream m_impl;
    PyWBuffer m_dataBuf;

    bool open(PyObject* data)
    {
        PyWBuffer dataBuf;
        if (!dataBuf.GetBuffer(data, PyBUF_SIMPLE)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "data must be a contiguous buffer");
            return false;
        }
        auto e = GetImpl().Open({ reinterpret_cast<const std::byte*>(dataBuf->buf), static_cast<size_t>(dataBuf->len) });
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return false;
        }
        m_dataBuf = std::move(dataBuf);
        return true;
    }
};

static PyMethodDef s_ransDecoderStreamMethods[] = {  //
    MakeMethodDef<&RansDecoderStream::Open>("open"),
    MakeMethodDef<&RansDecoderStream::Close>("close"),
    MakeMethodDef<&RansDecoderStream::IsOpen>("isOpen"),
    MakeMethodDef<&RansDecoderStream::DecodeEOF>("decodeEOF"),
    { 0 }
};

static PyType_Slot s_ransDecoderStreamSlots[] = {  //
    MakeTypeNewSlot<RansDecoderStream>(),
    MakeTypeDeallocSlot<RansDecoderStream>(),
    MakeTypeInitSlot<&RansDecoderStream::Init>(),
    { Py_tp_methods, s_ransDecoderStreamMethods },
    { 0 }
};

static PyType_Spec s_ransDecoderStreamSpec = {  //
    "msrtc.rans.RansDecoderStream", sizeof(PyWBox<RansDecoderStream>), 0, Py_TPFLAGS_DEFAULT, s_ransDecoderStreamSlots
};

class EntropyDecoder {
public:
    int Init(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "pmfLengths", "pmfOffsets", "pmfTable", "variant",
                                          "symbolBits", "bypassBits", nullptr };

        PyObject* pmfLengths{ nullptr };
        PyObject* pmfOffsets{ nullptr };
        PyObject* pmfTable{ nullptr };
        int variantArg = 0;
        unsigned symbolBits = 0;
        unsigned bypassBits = 0;

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiII|", const_cast<char**>(keywords),  //
                                              &pmfLengths, &pmfOffsets, &pmfTable, &variantArg, &symbolBits, &bypassBits);
        if (!rc) {
            return -1;
        }
        PyWBuffer pmfLengthsBuf;
        if (!pmfLengthsBuf.GetBuffer(pmfLengths, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfLengths must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfLengthsBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfLengths shape or data type, expected int32 1-d array");
            return -1;
        }
        PyWBuffer pmfOffsetsBuf;
        if (!pmfOffsetsBuf.GetBuffer(pmfOffsets, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfOffsets must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfOffsetsBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfOffsets shape or data type, expected int32 1-d array");
            return -1;
        }
        PyWBuffer pmfTableBuf;
        if (!pmfTableBuf.GetBuffer(pmfTable, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "pmfTable must be an int32 1-d array");
            return -1;
        }
        if (!isInt32Array(*pmfTableBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid pmfTable shape or data type, expected int32 1-d array");
            return -1;
        }
        msrtc_rans::RansVariant variant;
        if (!checkVariant(variantArg, variant)) {
            PyErr_SetString(PyExc_ValueError, "unknown rANS variant value");
            return -1;
        }
        std::error_code e;
        {
            PyWSavedThreadContext savedThreadContext;
            e = m_decoder.Initialize(
                variant,
                { reinterpret_cast<const int32_t*>(pmfLengthsBuf->buf), static_cast<size_t>(pmfLengthsBuf->shape[0]) },
                { reinterpret_cast<const int32_t*>(pmfOffsetsBuf->buf), static_cast<size_t>(pmfOffsetsBuf->shape[0]) },
                { reinterpret_cast<const int32_t*>(pmfTableBuf->buf), static_cast<size_t>(pmfTableBuf->shape[0]) },
                symbolBits, bypassBits);
        }
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return -1;
        }
        return 0;
    }

    PyObject* Decode(PyObject* args, PyObject* kwargs)
    {
        static const char* keywords[] = { "values", "indices", "stream", nullptr };

        PyObject* values{ nullptr };
        PyObject* indices{ nullptr };
        PyObject* stream{ nullptr };

        auto rc = PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", const_cast<char**>(keywords),  //
                                              &values, &indices, &stream);
        if (!rc) {
            return nullptr;
        }
        PyWBuffer valuesBuf;
        if (!valuesBuf.GetBuffer(values, PyBUF_CONTIG | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "values must be a writable int32 1-d array");
            return nullptr;
        }
        if (!isInt32Array(*valuesBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid values shape or data type, expected int32 1-d array");
            return nullptr;
        }
        PyWBuffer indicesBuf;
        if (!indicesBuf.GetBuffer(indices, PyBUF_CONTIG_RO | PyBUF_FORMAT)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "indices must be an int32 1-d array");
            return nullptr;
        }
        if (!isInt32Array(*indicesBuf, false)) {
            PyErr_SetString(PyExc_ValueError, "invalid indices shape or data type, expected int32 1-d array");
            return nullptr;
        }
        PyWBuffer dataBuf;
        bool isStream = PyWBox<RansDecoderStream>::TypeCheck(stream);
        if (!isStream && !dataBuf.GetBuffer(stream, PyBUF_SIMPLE)) {
            PyW_CatchAndHandleException();
            PyErr_SetString(PyExc_ValueError, "stream must by RansDecoderStream or a buffer");
            return nullptr;
        }
        std::error_code e;
        {
            PyWSavedThreadContext savedThreadContext;
            if (isStream) {
                e = m_decoder.Decode(
                    { reinterpret_cast<int32_t*>(valuesBuf->buf), static_cast<size_t>(valuesBuf->shape[0]) },
                    { reinterpret_cast<const int32_t*>(indicesBuf->buf), static_cast<size_t>(indicesBuf->shape[0]) },
                    PyWBox<RansDecoderStream>::Unwrap(stream).GetImpl());
            } else {
                e = m_decoder.Decode(
                    { reinterpret_cast<int32_t*>(valuesBuf->buf), static_cast<size_t>(valuesBuf->shape[0]) },
                    { reinterpret_cast<const int32_t*>(indicesBuf->buf), static_cast<size_t>(indicesBuf->shape[0]) },
                    { reinterpret_cast<const std::byte*>(dataBuf->buf), static_cast<size_t>(dataBuf->len) });
            }
        }
        if (e) {
            PyErr_SetString(PyExc_ValueError, e.message().c_str());
            return nullptr;
        }
        Py_RETURN_NONE;
    }

private:
    msrtc_rans::EntropyDecoder m_decoder;
};

static PyMethodDef s_entropyDecoderMethods[] = {  //
    MakeMethodDef<&EntropyDecoder::Decode>("decode"),
    { 0 }
};

static PyType_Slot s_entropyDecoderSlots[] = {  //
    MakeTypeNewSlot<EntropyDecoder>(),
    MakeTypeDeallocSlot<EntropyDecoder>(),
    MakeTypeInitSlot<&EntropyDecoder::Init>(),
    { Py_tp_methods, s_entropyDecoderMethods },
    { 0 }
};

static PyType_Spec s_entropyDecoderSpec = {  //
    "msrtc.rans.EntropyDecoder", sizeof(PyWBox<EntropyDecoder>), 0, Py_TPFLAGS_DEFAULT, s_entropyDecoderSlots
};

// A struct contains the definition of a module
static PyModuleDef s_rANSModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_msrtc_rans",  // Module name
    "Bindings to C++ rANS implementation",
    -1,    // Optional size of the module state memory
    NULL,  // Optional method table
    NULL,  // Optional slot definitions
    NULL,  // Optional traversal function
    NULL,  // Optional clear function
    NULL   // Optional module deallocation function
};

// The module init function
PyW_MODINIT_FUNC PyInit__msrtc_rans(void)
{
    auto module = PyWPtr::New(PyModule_Create(&s_rANSModuleDef));
    if (!module) {
        return nullptr;
    }
    if (!AddPyType(module, s_ransEncoderStreamSpec)) {
        return nullptr;
    }
    if (!AddPyType(module, s_entropyEncoderSpec)) {
        return nullptr;
    }
    if (!AddPyType(module, s_ransDecoderStreamSpec)) {
        return nullptr;
    }
    if (!AddPyType(module, s_entropyDecoderSpec)) {
        return nullptr;
    }
    if (PyModule_AddIntConstant(module, "RansByte", static_cast<int>(msrtc_rans::RansVariant::RansByte)) < 0) {
        return nullptr;
    }
    if (PyModule_AddIntConstant(module, "Rans64", static_cast<int>(msrtc_rans::RansVariant::Rans64)) < 0) {
        return nullptr;
    }
    return module.Detach();
}
