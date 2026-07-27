// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @file msrtc_rans/entropy_coder.h
 * @brief Defines msrtc_rans::EntropyEncoder and msrtc_rans::EntropyDecoder - entropy encoder and decoder using rANS alogirthm
 */

#pragma once

#if !defined(__cplusplus)
    #error This header is for C++ only
#endif

#include <cstdint>
#include <memory>
#include <system_error>

#include <msrtc_rans/span.h>

namespace msrtc_rans {

// Variants of rANS algorithm
enum class RansVariant {
    Rans64,
    RansByte,
};

// Error codes
enum class error : int {
    general_failure,
    invalid_pmf,
    invalid_params,
    invalid_state,
    invalid_stream,
};

// msrtc_rans error category
const std::error_category& error_category() noexcept;

// make error code from error enum
inline std::error_code make_error_code(error e) noexcept
{
    return std::error_code(static_cast<int>(e), error_category());
}

/// Resizable buffer interface to allow different buffer allocation strategies
struct IResizableBuffer {
    // Minimal buffer size
    static constexpr inline size_t s_MinBufferSize = 512;
    // Minimal alignment
    static constexpr inline size_t s_MinAlignment = sizeof(uint32_t);

    // Adjust size to alignment requirement
    static constexpr size_t AlignSize(size_t size, bool up)
    {
        if (up) {
            size += s_MinAlignment - 1;
        }
        return size & ~static_cast<size_t>((1 << s_MinAlignment) - 1);
    }

    // Get current buffer
    virtual span<std::byte> GetBuffer() = 0;
    // Begin grow operation and return a new buffer
    virtual span<std::byte> BeginToGrow() = 0;
    // Complete active change (i.e. grow operation)
    virtual void Commit() = 0;
    // Rollback active change (i.e. grow operation)
    virtual void Rollback() = 0;
};

// Resizable buffer implemented with new [] operator
class HeapResizableBuffer final : public IResizableBuffer {
public:
    HeapResizableBuffer(size_t initialSize = 4096, size_t maxSizeStep = 1024 * 1024);
    ~HeapResizableBuffer() = default;

    // Get current buffer
    virtual span<std::byte> GetBuffer() override { return { m_buffer.get(), m_bufferSize }; }
    // Begin grow operation and return a new buffer
    virtual span<std::byte> BeginToGrow() override;
    // Complete active change (i.e. grow operation)
    virtual void Commit() override;
    // Rollback active change (i.e. grow operation)
    virtual void Rollback() override;

private:
    // Current buffer
    std::unique_ptr<std::byte[]> m_buffer;
    // Size of buffer
    size_t m_bufferSize;
    // Growing buffer
    std::unique_ptr<std::byte[]> m_newBuffer;
    // Size of growing buffer
    size_t m_newBufferSize;
    // Max buffer growth
    size_t m_maxSizeStep;
};

namespace details {

// rANS encoder stream interface
struct IRansEncoderStreamImpl {
    virtual ~IRansEncoderStreamImpl() = default;

    // minimalistic downcast query
    virtual void* QueryDowncast(const void* tag) = 0;
    // flush current encoding session and yield output
    virtual span<const std::byte> Flush(bool abort) = 0;
};

// rANS decoder stream interface
struct IRansDecoderStreamImpl {
    virtual ~IRansDecoderStreamImpl() = default;

    // minimalistic downcast query
    virtual void* QueryDowncast(const void* tag) = 0;
    // open message
    virtual std::error_code Open(const span<const std::byte>& data) = 0;
    // close current message
    virtual void Close() = 0;
    // check whether stream is open
    virtual bool IsOpen() const = 0;
    // check whether stream can be at EOF
    virtual bool CheckEOF() const = 0;
};

// entropy encoder interface
struct IEntropyEncoderImpl {
    // initialization
    virtual std::error_code Initialize(const span<const int32_t>& pmfLengths, const span<const int32_t>& pmfOffsets,
                                       const span<const int32_t>& pmfTable, int symbolBits, int bypassBits) = 0;
    // encoding
    virtual span<const std::byte> Encode(IResizableBuffer& buffer, const span<const int32_t>& indices,
                                         const span<const int32_t>& values) const = 0;
    // encoding with stream
    virtual std::error_code Encode(IRansEncoderStreamImpl& stream, const span<const int32_t>& indices,
                                   const span<const int32_t>& values) const = 0;
};

// entropy decoder interface
struct IEntropyDecoderImpl {
    // initialization
    virtual std::error_code Initialize(const span<const int32_t>& pmfLengths, const span<const int32_t>& pmfOffsets,
                                       const span<const int32_t>& pmfTable, int symbolBits, int bypassBits) = 0;
    // decoding
    virtual std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                                   const span<const std::byte>& data) const = 0;
    // decoding with stream
    virtual std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                                   IRansDecoderStreamImpl& stream) const = 0;
};

}  // namespace details

class EntropyEncoder;
class EntropyDecoder;

// rANS encoder stream
class RansEncoderStream {
public:
    RansEncoderStream() = default;
    ~RansEncoderStream() = default;

    bool IsInitialized() const { return m_impl != nullptr; }

    std::error_code Initialize(RansVariant variant, IResizableBuffer& buffer);

    // flush current encoding session and yield output
    span<const std::byte> Flush(bool abort = false)
    {

        if (!m_impl) {
            assert(false);
            return {};
        }
        return m_impl->Flush(abort);
    }

private:
    std::unique_ptr<details::IRansEncoderStreamImpl> m_impl;

    friend class EntropyEncoder;
};

// rANS decoder stream
class RansDecoderStream {
public:
    RansDecoderStream() = default;
    ~RansDecoderStream() = default;

    bool IsInitialized() const { return m_impl != nullptr; }

    std::error_code Initialize(RansVariant variant);

    // open message
    std::error_code Open(const span<const std::byte>& data)
    {
        if (!m_impl) {
            return make_error_code(error::invalid_state);
        }
        return m_impl->Open(data);
    }
    // close current message
    void Close()
    {
        if (m_impl) {
            m_impl->Close();
        }
    }
    // check whether stream is open
    bool IsOpen() const { return m_impl && m_impl->IsOpen(); }
    // check whether stream can be at EOF
    bool CheckEOF() const { return m_impl && m_impl->CheckEOF(); }

private:
    std::unique_ptr<details::IRansDecoderStreamImpl> m_impl;

    friend class EntropyDecoder;
};

// Entropy encoder
class EntropyEncoder {
public:
    EntropyEncoder() = default;
    ~EntropyEncoder() = default;

    bool IsInitialized() const { return m_impl != nullptr; }

    // Initialize encoder
    std::error_code Initialize(RansVariant variant, const span<const int32_t>& pmfLengths,
                               const span<const int32_t>& pmfOffsets, const span<const int32_t>& pmfTable,
                               int symbolBits = 16, int bypassBits = 4);
    // Encode
    span<const std::byte> Encode(IResizableBuffer& buffer, const span<const int32_t>& indices,
                                 const span<const int32_t>& values) const
    {
        if (!m_impl) {
            return {};
        }
        return m_impl->Encode(buffer, indices, values);
    }
    // Encode with stream
    std::error_code Encode(RansEncoderStream& stream, const span<const int32_t>& indices,
                           const span<const int32_t>& values) const
    {
        if (!m_impl || !stream.m_impl) {
            return make_error_code(error::invalid_state);
        }
        return m_impl->Encode(*stream.m_impl, indices, values);
    }

private:
    std::shared_ptr<const details::IEntropyEncoderImpl> m_impl;
};

// Entropy decoder
class EntropyDecoder {
public:
    EntropyDecoder() = default;
    ~EntropyDecoder() = default;

    bool IsInitialized() const { return m_impl != nullptr; }

    // Initialize encoder
    std::error_code Initialize(RansVariant variant, const span<const int32_t>& pmfLengths,
                               const span<const int32_t>& pmfOffsets, const span<const int32_t>& pmfTable,
                               int symbolBits = 16, int bypassBits = 4);
    // Decode
    std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                           const span<const std::byte>& data) const
    {
        if (!m_impl) {
            return make_error_code(error::invalid_state);
        }
        return m_impl->Decode(values, indices, data);
    }

    // Decode with stream
    virtual std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                                   RansDecoderStream& stream) const
    {
        if (!m_impl || !stream.m_impl) {
            return make_error_code(error::invalid_state);
        }
        return m_impl->Decode(values, indices, *stream.m_impl);
    }

private:
    std::shared_ptr<const details::IEntropyDecoderImpl> m_impl;
};

}  // namespace msrtc_rans

template <>
struct std::is_error_code_enum<msrtc_rans::error> : public std::true_type {};
