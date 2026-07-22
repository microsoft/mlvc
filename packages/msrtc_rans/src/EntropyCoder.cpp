// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "msrtc_rans/EntropyCoder.h"
#include "msrtc_rans/rans.h"
#include <algorithm>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
    #define MSRTC_RANS_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define MSRTC_RANS_NOINLINE __declspec(noinline)
#else
    #define MSRTC_RANS_NOINLINE
#endif

namespace msrtc_rans {

// ----- error category
class msrtc_rans_error_category final : public std::error_category {
    const char* name() const noexcept override { return "msrtc_rans"; }

    std::string message(int ev) const noexcept override
    {
        switch (static_cast<error>(ev)) {
        case error::general_failure:
            return "general failure";
        case error::invalid_pmf:
            return "invalid PMF data";
        case error::invalid_params:
            return "invalid parameter value";
        case error::invalid_stream:
            return "invalid stream";
        case error::invalid_state:
            return "invalid state";
        }
        assert(false);
        return "unknown error code";
    }
};

const std::error_category& error_category() noexcept
{
    static const msrtc_rans_error_category category;
    return category;
}

// ----- HeapResizableBuffer

HeapResizableBuffer::HeapResizableBuffer(size_t initialSize, size_t maxSizeStep) : m_newBufferSize(0)
{
    initialSize = std::max(AlignSize(initialSize, true), s_MinBufferSize);
    m_buffer = std::make_unique<std::byte[]>(initialSize);
    m_bufferSize = initialSize;
    m_maxSizeStep = std::max(AlignSize(maxSizeStep, false), s_MinBufferSize);
}

span<std::byte> HeapResizableBuffer::BeginToGrow()
{
    auto newSize = m_bufferSize + std::min(m_bufferSize, m_maxSizeStep);
    m_newBuffer = std::make_unique<std::byte[]>(newSize);
    m_newBufferSize = newSize;
    return { m_newBuffer.get(), m_newBufferSize };
}

void HeapResizableBuffer::Commit()
{
    if (m_newBuffer) {
        m_buffer = std::move(m_newBuffer);
        m_bufferSize = m_newBufferSize;
        m_newBufferSize = 0;
        m_newBuffer.release();
    }
}

void HeapResizableBuffer::Rollback()
{
    if (m_newBuffer) {
        m_newBufferSize = 0;
        m_newBuffer.release();
    }
}

// ----- helper functions and types

// Probability distribution descriptor
struct DistributionDesc {
    // offset of zero value symbol
    int32_t m_ValueOffset;
    // last encodable value used as bypass sentinel
    int32_t m_BypassSentinel;
    // rANS symbol offset in global symbol table
    size_t m_SymbolOffset;
};

static std::error_code intializeDistributionDesc(std::vector<DistributionDesc>& distributionDescs,
                                                 const span<const int32_t>& pmfLengths,
                                                 const span<const int32_t>& pmfOffsets, size_t pmfTableSize)
{
    auto distributionCount = pmfLengths.size();
    if (pmfOffsets.size() != distributionCount) {
        return make_error_code(error::invalid_pmf);
    }
    distributionDescs.resize(distributionCount);

    size_t symbolCursor = 0;
    for (size_t i = 0; i < distributionCount; i++) {
        // last PMF element is tail mass, to be used by bypass symbol
        auto length = pmfLengths[i];
        if (length <= 1 || pmfTableSize - symbolCursor < static_cast<size_t>(length)) {
            return make_error_code(error::invalid_pmf);
        }
        auto& desc = distributionDescs[i];
        desc.m_ValueOffset = pmfOffsets[i];
        desc.m_BypassSentinel = length - 1;
        desc.m_SymbolOffset = symbolCursor;

        symbolCursor += static_cast<size_t>(length);
    }

    if (symbolCursor != pmfTableSize) {
        return make_error_code(error::invalid_pmf);
    }
    return {};
}

template <typename StateType, typename UnitType>
static std::error_code checkBits(int probabilityBits)
{
    constexpr auto MaxScaleBits = details::RansConstants<StateType, UnitType>::MaxScaleBits;
    if (probabilityBits < 2 || static_cast<size_t>(probabilityBits) > MaxScaleBits) {
        return make_error_code(error::invalid_params);
    }
    return {};
}

template <typename T>
static bool isSame(const span<T>& s1, const span<T>& s2)
{
    return s1.data() == s2.data() && s1.size() == s2.size();
}

template <typename F, typename T>
static std::error_code intializeSymbols(F&& makeSymbol, std::vector<T>& symbols,
                                        const std::vector<DistributionDesc>& distributionDescs,
                                        const span<const int32_t>& pmfTable, rans_freq_t symbolBits)
{
    auto totalSymbolCount = pmfTable.size();
    symbols.reserve(totalSymbolCount);

    auto maxFreq = static_cast<int32_t>(1) << symbolBits;

    auto pmfTableIt = pmfTable.begin();
    for (auto& desc : distributionDescs) {
        int32_t start = 0;
        for (size_t i = 0; i <= static_cast<size_t>(desc.m_BypassSentinel); i++) {
            auto freq = *pmfTableIt;
            ++pmfTableIt;
            if (!(freq > 0 && freq <= maxFreq - start)) {
                return make_error_code(error::invalid_pmf);
            }
            symbols.push_back(makeSymbol(static_cast<rans_freq_t>(start),  //
                                         static_cast<rans_freq_t>(freq), symbolBits));
            start += freq;
        }
    }

    return {};
}

// ----- rANS encoder stream

namespace details {

template <typename UnitType>
class ResizableBufferSink {
public:
    using unit_t = UnitType;

    ResizableBufferSink(IResizableBuffer& buffer);

    static span<unit_t> ToUnit(const span<std::byte>& span)
    {
        return { reinterpret_cast<unit_t*>(span.data()), span.size() / sizeof(unit_t) };
    }

    void operator()(unit_t value)
    {
        if (m_encodePtr <= m_bufferPtr) {
            enlargeBuffer();
        }
        *(--m_encodePtr) = value;
    }

    span<const std::byte> EncodedSpan() const
    {
        return { reinterpret_cast<const std::byte*>(m_encodePtr), reinterpret_cast<const std::byte*>(m_endPtr) };
    }

    void Reset() { m_encodePtr = m_endPtr; }

private:
    // Buffer
    IResizableBuffer& m_buffer;
    // Start of current buffer
    unit_t* m_bufferPtr;
    // Encoded data pointer
    unit_t* m_encodePtr;
    // Buffer end pointer
    unit_t* m_endPtr;

    void enlargeBuffer();
};

template <typename UnitType>
inline ResizableBufferSink<UnitType>::ResizableBufferSink(IResizableBuffer& buffer) : m_buffer(buffer)
{
    auto bufferSpan = ToUnit(m_buffer.GetBuffer());
    m_bufferPtr = bufferSpan.data();
    // rANS message is created from end to start
    m_encodePtr = m_endPtr = bufferSpan.data() + bufferSpan.size();
}

template <typename UnitType>
MSRTC_RANS_NOINLINE void ResizableBufferSink<UnitType>::enlargeBuffer()
{
    auto newBufferSpan = ToUnit(m_buffer.BeginToGrow());

    auto content = span{ m_encodePtr, m_endPtr };
    assert(content.size() < newBufferSpan.size());

    auto newContent = newBufferSpan.last(content.size());
    std::copy(content.begin(), content.end(), newContent.begin());

    m_buffer.Commit();
    assert(isSame(newBufferSpan, ToUnit(m_buffer.GetBuffer())));

    m_bufferPtr = newBufferSpan.data();
    m_endPtr = newBufferSpan.data() + newBufferSpan.size();
    m_encodePtr = newContent.data();

    assert(m_bufferPtr < m_encodePtr);
}

template <typename StateType, typename UnitType>
using RawRansEncoderStream = RansEncoder<StateType, UnitType, ResizableBufferSink<UnitType>>;

template <typename StateType, typename UnitType>
class RansEncoderStreamImpl final : public IRansEncoderStreamImpl {
public:
    using state_t = StateType;
    using unit_t = UnitType;

    using RawStreamType = RawRansEncoderStream<state_t, unit_t>;

    RansEncoderStreamImpl(IResizableBuffer& buffer) : m_rawStream(buffer) {}
    ~RansEncoderStreamImpl() = default;

    RawStreamType& RawStream() { return m_rawStream; }

    static RansEncoderStreamImpl* Downcast(IRansEncoderStreamImpl& baseRef)
    {
        return reinterpret_cast<RansEncoderStreamImpl*>(baseRef.QueryDowncast(&s_tag));
    }

    virtual void* QueryDowncast(const void* tag) override;
    virtual span<const std::byte> Flush(bool abort) override;

private:
    RawStreamType m_rawStream;

    // Tag for downcast
    static const std::byte s_tag;
};

template <typename StateType, typename UnitType>
const std::byte RansEncoderStreamImpl<StateType, UnitType>::s_tag{};

template <typename StateType, typename UnitType>
void* RansEncoderStreamImpl<StateType, UnitType>::QueryDowncast(const void* tag)
{
    if (tag == &s_tag) {
        return this;
    }
    return nullptr;
}

template <typename StateType, typename UnitType>
span<const std::byte> RansEncoderStreamImpl<StateType, UnitType>::Flush(bool abort)
{
    span<const std::byte> data;
    if (!abort) {
        RawStream().Flush();
        data = RawStream().GetSink().EncodedSpan();
    }
    RawStream().Reset();
    RawStream().GetSink().Reset();
    return data;
}

}  // namespace details

std::error_code RansEncoderStream::Initialize(RansVariant variant, IResizableBuffer& buffer)
{
    std::unique_ptr<details::IRansEncoderStreamImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_unique<details::RansEncoderStreamImpl<uint32_t, uint8_t>>(buffer);
        break;
    case RansVariant::Rans64:
        impl = std::make_unique<details::RansEncoderStreamImpl<uint64_t, uint32_t>>(buffer);
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    m_impl = std::move(impl);
    return {};
}

// ----- entropy encoder

namespace details {

template <typename StateType, typename UnitType>
class EntropyEncoderImpl final : public IEntropyEncoderImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    virtual std::error_code Initialize(const span<const int32_t>& pmfLengths, const span<const int32_t>& pmfOffsets,
                                       const span<const int32_t>& pmfTable, int symbolBits, int bypassBits) override;
    virtual span<const std::byte> Encode(IResizableBuffer& buffer, const span<const int32_t>& indices,
                                         const span<const int32_t>& values) const override;
    virtual std::error_code Encode(IRansEncoderStreamImpl& stream, const span<const int32_t>& indices,
                                   const span<const int32_t>& values) const override;

private:
    using symbol_t = RansEncSymbol<state_t, unit_t>;
    using RansEncoder = RawRansEncoderStream<state_t, unit_t>;

    // minimal bypass bits value is 2
    static constexpr inline size_t s_MaxBypassParts = sizeof(freq_t) * CHAR_BIT / 2;

    // Bits used for symbol encoding
    freq_t m_symbolBits = 0;
    // distribution descriptors
    std::vector<DistributionDesc> m_distributionDescs;
    // Concatenated symbol table for all distributions
    std::vector<symbol_t> m_symbols;
    // Bits used for bypass encoding
    freq_t m_bypassBits = 0;
    // Max bypass value
    freq_t m_bypassMaxValue = 0;

    void encode(RansEncoder& encoder, const span<const int32_t>& indices, const span<const int32_t>& values) const;
    void encodeBypassValue(RansEncoder& encoder, freq_t bypassValue) const;
};

template <typename StateType, typename UnitType>
std::error_code EntropyEncoderImpl<StateType, UnitType>::Initialize(const span<const int32_t>& pmfLengths,
                                                                    const span<const int32_t>& pmfOffsets,
                                                                    const span<const int32_t>& pmfTable, int symbolBits,
                                                                    int bypassBits)
{
    auto e = checkBits<state_t, unit_t>(symbolBits);
    if (e) {
        return e;
    }
    e = checkBits<state_t, unit_t>(bypassBits);
    if (e) {
        return e;
    }
    std::vector<DistributionDesc> distributionDescs;
    e = intializeDistributionDesc(distributionDescs, pmfLengths, pmfOffsets, pmfTable.size());
    if (e) {
        return e;
    }
    std::vector<symbol_t> symbols;
    e = intializeSymbols(
        [](freq_t start, freq_t freq, freq_t symbolBits) {  //
            return symbol_t(start, freq, symbolBits);
        },
        symbols, distributionDescs, pmfTable, static_cast<freq_t>(symbolBits));
    if (e) {
        return e;
    }
    m_distributionDescs = std::move(distributionDescs);
    m_symbols = std::move(symbols);
    m_symbolBits = static_cast<freq_t>(symbolBits);
    m_bypassBits = static_cast<freq_t>(bypassBits);
    m_bypassMaxValue = static_cast<freq_t>((1U << bypassBits) - 1);
    return {};
}

template <typename StateType, typename UnitType>
span<const std::byte> EntropyEncoderImpl<StateType, UnitType>::Encode(IResizableBuffer& buffer,
                                                                      const span<const int32_t>& indices,
                                                                      const span<const int32_t>& values) const
{
    if (!m_symbolBits) {
        // not initialized
        assert(false);
        return {};
    }
    RansEncoder encoder(buffer);
    encode(encoder, indices, values);
    encoder.Flush();
    return encoder.GetSink().EncodedSpan();
}

template <typename StateType, typename UnitType>
std::error_code EntropyEncoderImpl<StateType, UnitType>::Encode(IRansEncoderStreamImpl& stream,
                                                                const span<const int32_t>& indices,
                                                                const span<const int32_t>& values) const
{
    if (!m_symbolBits) {
        // not initialized
        assert(false);
        return make_error_code(error::invalid_state);
    }
    auto* streamImpl = RansEncoderStreamImpl<state_t, unit_t>::Downcast(stream);
    if (!streamImpl) {
        return make_error_code(error::invalid_params);
    }
    encode(streamImpl->RawStream(), indices, values);
    return {};
}

template <typename StateType, typename UnitType>
void EntropyEncoderImpl<StateType, UnitType>::encode(RansEncoder& encoder, const span<const int32_t>& indices,
                                                     const span<const int32_t>& values) const
{
    assert(m_symbolBits);

    auto dataSize = indices.size();
    assert(dataSize == values.size());

    // data is encoded in reverse order
    auto* indicesStart = indices.data();
    auto* indexPtr = indicesStart + dataSize - 1;
    auto* valuesPtr = values.data() + dataSize - 1;

    for (; indexPtr >= indicesStart; --indexPtr, --valuesPtr) {
        auto index = *indexPtr;
        if (index < 0) {
            // skip encoding, return 0 on decoding
            assert(!*valuesPtr);
            continue;
        }
        assert(static_cast<size_t>(index) < m_distributionDescs.size());
        // clamp distribution index to a valid value
        index = std::min(index, static_cast<int32_t>(m_distributionDescs.size()) - 1);
        const auto& distributionDesc = m_distributionDescs[index];

        auto value = *valuesPtr + distributionDesc.m_ValueOffset;
        if (value < 0 || value >= distributionDesc.m_BypassSentinel) {
            // Values out of symbol (PDF) range are encoded using bypass, which adds a lot of overhead
            freq_t bypassValue;
            if (value < 0) {
                bypassValue = 2 * static_cast<freq_t>(-value) - 1;
            } else {
                bypassValue = 2 * static_cast<freq_t>(value - distributionDesc.m_BypassSentinel);
            }
            encodeBypassValue(encoder, bypassValue);

            value = distributionDesc.m_BypassSentinel;
        }

        auto symbol = distributionDesc.m_SymbolOffset + static_cast<freq_t>(value);
        encoder.Put(m_symbols[symbol]);
    }
}

template <typename StateType, typename UnitType>
inline void EntropyEncoderImpl<StateType, UnitType>::encodeBypassValue(RansEncoder& encoder, freq_t bypassValue) const
{
    // Buffer for bypass values
    std::array<freq_t, s_MaxBypassParts> bypassBuffer;

    // Split bypassValue into bypassBits parts
    auto cursor = bypassBuffer.begin();
    for (; bypassValue != 0; ++cursor, bypassValue >>= m_bypassBits) {
        *cursor = bypassValue & m_bypassMaxValue;
    }

    auto bypassCount = static_cast<freq_t>(cursor - bypassBuffer.begin());

    // Put parts in reverse order
    while (cursor > bypassBuffer.begin()) {
        encoder.Put(*(--cursor), 1, m_bypassBits);
    }

    auto bypassPrefixCount = 0;
    // expected to be faster than division
    for (; bypassCount >= m_bypassMaxValue; bypassCount -= m_bypassMaxValue) {
        ++bypassPrefixCount;
    }
    // Put bypassCount remainder
    encoder.Put(bypassCount, 1, m_bypassBits);
    // Put bypassCount prefix
    for (; bypassPrefixCount > 0; --bypassPrefixCount) {
        encoder.Put(m_bypassMaxValue, 1, m_bypassBits);
    }
}

}  // namespace details

std::error_code EntropyEncoder::Initialize(RansVariant variant, const span<const int32_t>& pmfLengths,
                                           const span<const int32_t>& pmfOffsets, const span<const int32_t>& pmfTable,
                                           int symbolBits, int bypassBits)
{
    std::shared_ptr<details::IEntropyEncoderImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_shared<details::EntropyEncoderImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_shared<details::EntropyEncoderImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    auto ec = impl->Initialize(pmfLengths, pmfOffsets, pmfTable, symbolBits, bypassBits);
    if (ec) {
        return ec;
    }

    m_impl = std::move(impl);
    return {};
}

// ----- rANS decoder stream

namespace details {

template <typename UnitType>
class ByteSpanSource {
public:
    using unit_t = UnitType;

    ByteSpanSource(const span<const unit_t>& data) : m_decodePtr(data.data()), m_endPtr(data.data() + data.size()) {}

    bool IsOpen() const { return m_decodePtr != nullptr; }

    bool operator()(unit_t& value)
    {
        if (m_decodePtr == m_endPtr) {
            return false;
        }
        value = *(m_decodePtr++);
        return true;
    }

    bool OnOK() { return true; }
    bool OnInvalidStream() { return false; }

    bool IsEOF() const { return m_decodePtr == m_endPtr; }

private:
    const unit_t* m_decodePtr;
    const unit_t* m_endPtr;
};

template <typename StateType, typename UnitType>
using RawRansDecoderStream = RansDecoder<StateType, UnitType, ByteSpanSource<UnitType>>;

template <typename StateType, typename UnitType>
class RansDecoderStreamImpl final : public IRansDecoderStreamImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    using RawStreamType = RawRansDecoderStream<StateType, UnitType>;

    RansDecoderStreamImpl() : m_rawStream(span<const unit_t>{}) {}
    ~RansDecoderStreamImpl() = default;

    RawStreamType& RawStream() { return m_rawStream; }
    const RawStreamType& RawStream() const { return m_rawStream; }

    static RansDecoderStreamImpl* Downcast(IRansDecoderStreamImpl& baseRef)
    {
        return reinterpret_cast<RansDecoderStreamImpl*>(baseRef.QueryDowncast(&s_tag));
    }

    virtual void* QueryDowncast(const void* tag) override;
    virtual std::error_code Open(const span<const std::byte>& data) override;
    virtual void Close() override;
    virtual bool IsOpen() const override;
    virtual bool CheckEOF() const override;

private:
    RawStreamType m_rawStream;

    // Tag for downcast
    static const std::byte s_tag;
};

template <typename StateType, typename UnitType>
const std::byte RansDecoderStreamImpl<StateType, UnitType>::s_tag{};

template <typename StateType, typename UnitType>
void* RansDecoderStreamImpl<StateType, UnitType>::QueryDowncast(const void* tag)
{
    if (tag == &s_tag) {
        return this;
    }
    return nullptr;
}

template <typename StateType, typename UnitType>
std::error_code RansDecoderStreamImpl<StateType, UnitType>::Open(const span<const std::byte>& data)
{
    if (data.size() % sizeof(unit_t)) {
        return make_error_code(error::invalid_stream);
    }
    auto unitData = span{ reinterpret_cast<const unit_t*>(data.data()), data.size() / sizeof(unit_t) };

    auto stream = RawStreamType(unitData);
    if (!stream.Init()) {
        return make_error_code(error::invalid_stream);
    }
    m_rawStream = std::move(stream);
    return {};
}

template <typename StateType, typename UnitType>
void RansDecoderStreamImpl<StateType, UnitType>::Close()
{
    m_rawStream = RawStreamType(span<const unit_t>{});
}

template <typename StateType, typename UnitType>
bool RansDecoderStreamImpl<StateType, UnitType>::IsOpen() const
{
    return RawStream().GetSource().IsOpen();
}

template <typename StateType, typename UnitType>
bool RansDecoderStreamImpl<StateType, UnitType>::CheckEOF() const
{
    return RawStream().GetSource().IsOpen() && RawStream().GetSource().IsEOF() && RawStream().CheckEOF();
}

}  // namespace details

std::error_code RansDecoderStream::Initialize(RansVariant variant)
{
    std::unique_ptr<details::IRansDecoderStreamImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_unique<details::RansDecoderStreamImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_unique<details::RansDecoderStreamImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    m_impl = std::move(impl);
    return {};
}

// ----- entropy decoder

namespace details {

template <typename StateType, typename UnitType>
class EntropyDecoderImpl final : public IEntropyDecoderImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    virtual std::error_code Initialize(const span<const int32_t>& pmfLengths, const span<const int32_t>& pmfOffsets,
                                       const span<const int32_t>& pmfTable, int symbolBits, int bypassBits) override;

    virtual std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                                   const span<const std::byte>& data) const override;
    virtual std::error_code Decode(const span<int32_t>& values, const span<const int32_t>& indices,
                                   IRansDecoderStreamImpl& stream) const override;

private:
    using RansDecoder = RawRansDecoderStream<state_t, unit_t>;

    // Bits used for symbol encoding
    freq_t m_symbolBits = 0;
    // Bits used for bypass encoding
    freq_t m_bypassBits = 0;
    // Max bypass value
    freq_t m_bypassMaxValue = 0;
    // Distribution descriptors
    std::vector<DistributionDesc> m_distributionDescs;
    // Concatenated CDF table
    std::vector<freq_t> m_cdfTable;

    std::error_code decode(RansDecoder& decoder, const span<int32_t>& values, const span<const int32_t>& indices) const;
    bool decodeBypassValue(RansDecoder& decoder, freq_t& bypassValue) const;
};

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Initialize(const span<const int32_t>& pmfLengths,
                                                                    const span<const int32_t>& pmfOffsets,
                                                                    const span<const int32_t>& pmfTable, int symbolBits,
                                                                    int bypassBits)
{
    auto e = checkBits<state_t, unit_t>(symbolBits);
    if (e) {
        return e;
    }
    e = checkBits<state_t, unit_t>(bypassBits);
    if (e) {
        return e;
    }
    std::vector<DistributionDesc> distributionDescs;
    e = intializeDistributionDesc(distributionDescs, pmfLengths, pmfOffsets, pmfTable.size());
    if (e) {
        return e;
    }

    std::vector<freq_t> cdfTable(pmfTable.size() + distributionDescs.size());

    auto maxFreq = static_cast<int32_t>(1) << symbolBits;
    size_t cursor = 0;
    for (size_t index = 0; index < distributionDescs.size(); index++) {
        auto& desc = distributionDescs[index];

        // set symbol offset to offset in CDF table
        desc.m_SymbolOffset = cursor + index;

        int32_t start = 0;
        for (size_t i = 0; i <= static_cast<size_t>(desc.m_BypassSentinel); i++, cursor++) {
            auto freq = pmfTable[cursor];
            if (!(freq > 0 && freq <= maxFreq - start)) {
                return make_error_code(error::invalid_pmf);
            }
            cdfTable[cursor + index] = static_cast<freq_t>(start);
            start += freq;
        }
        cdfTable[cursor + index] = start;
    }

    m_distributionDescs = std::move(distributionDescs);
    m_cdfTable = std::move(cdfTable);
    m_symbolBits = static_cast<freq_t>(symbolBits);
    m_bypassBits = static_cast<freq_t>(bypassBits);
    m_bypassMaxValue = static_cast<freq_t>((1U << bypassBits) - 1);

    return {};
}

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Decode(const span<int32_t>& values,
                                                                const span<const int32_t>& indices,
                                                                const span<const std::byte>& data) const
{
    if (data.size() % sizeof(unit_t)) {
        return make_error_code(error::invalid_stream);
    }
    auto unitData = span{ reinterpret_cast<const unit_t*>(data.data()), data.size() / sizeof(unit_t) };
    RansDecoder decoder(unitData);
    if (!decoder.Init()) {
        return make_error_code(error::invalid_stream);
    }
    auto ec = decode(decoder, values, indices);
    if (ec) {
        return ec;
    }
    if (!decoder.GetSource().IsEOF() || !decoder.CheckEOF()) {
        return make_error_code(error::invalid_stream);
    }
    return {};
}

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Decode(const span<int32_t>& values,
                                                                const span<const int32_t>& indices,
                                                                IRansDecoderStreamImpl& stream) const
{
    auto* streamImpl = RansDecoderStreamImpl<state_t, unit_t>::Downcast(stream);
    if (!streamImpl) {
        return make_error_code(error::invalid_params);
    }
    return decode(streamImpl->RawStream(), values, indices);
}

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::decode(RansDecoder& decoder, const span<int32_t>& values,
                                                                const span<const int32_t>& indices) const
{
    if (!m_symbolBits) {
        assert(false);
        return make_error_code(error::invalid_state);
    }

    if (values.size() != indices.size()) {
        return make_error_code(error::invalid_params);
    }

    auto indicesIt = indices.begin();
    for (auto& value : values) {
        auto index = *indicesIt;
        ++indicesIt;

        if (index < 0) {
            // Skipped value
            value = 0;
            continue;
        }

        assert(static_cast<size_t>(index) < m_distributionDescs.size());
        // clamp distribution index to a valid value
        index = std::min(index, static_cast<int32_t>(m_distributionDescs.size()) - 1);
        const auto& distributionDesc = m_distributionDescs[index];

        freq_t cumFreq = decoder.Get(m_symbolBits);
        assert(cumFreq < static_cast<size_t>(1) << m_symbolBits);

        const auto* basePtr = m_cdfTable.data() + distributionDesc.m_SymbolOffset;
        const auto* startPtr = std::upper_bound(basePtr + 1, basePtr + distributionDesc.m_BypassSentinel + 1, cumFreq) - 1;
        if (!decoder.Advance(startPtr[0], startPtr[1] - startPtr[0], m_symbolBits)) {
            return make_error_code(error::invalid_stream);
        }

        auto symbol = static_cast<int32_t>(startPtr - basePtr);
        if (symbol == distributionDesc.m_BypassSentinel) {
            freq_t bypassValue;
            if (!decodeBypassValue(decoder, bypassValue)) {
                return make_error_code(error::invalid_stream);
            }
            if (bypassValue & 1) {
                symbol = -static_cast<int32_t>(bypassValue >> 1) - 1;
            } else {
                symbol = static_cast<int32_t>(bypassValue >> 1) + distributionDesc.m_BypassSentinel;
            }
        }
        value = symbol - distributionDesc.m_ValueOffset;
    }
    return {};
}

template <typename StateType, typename UnitType>
bool EntropyDecoderImpl<StateType, UnitType>::decodeBypassValue(RansDecoder& decoder, freq_t& bypassValue) const
{
    // Step 1 : Read bypass count.
    freq_t value = decoder.Get(m_bypassBits);
    if (!decoder.Advance(value, 1, m_bypassBits)) {
        return false;
    }
    freq_t bypassCount = value;
    while (value == m_bypassMaxValue) {
        value = decoder.Get(m_bypassBits);
        if (!decoder.Advance(value, 1, m_bypassBits)) {
            return false;
        }
        bypassCount += value;
        if (bypassCount > sizeof(freq_t) * CHAR_BIT) {
            return false;
        }
    }

    // Step 2 : Read bypass value.
    freq_t encodedValue = 0;
    bypassCount *= m_bypassBits;
    for (freq_t bypassShift = 0; bypassShift < bypassCount; bypassShift += m_bypassBits) {
        value = decoder.Get(m_bypassBits);
        if (!decoder.Advance(value, 1, m_bypassBits)) {
            return false;
        }
        encodedValue |= value << bypassShift;
    }
    bypassValue = encodedValue;
    return true;
}

}  // namespace details

std::error_code EntropyDecoder::Initialize(RansVariant variant, const span<const int32_t>& pmfLengths,
                                           const span<const int32_t>& pmfOffsets, const span<const int32_t>& pmfTable,
                                           int symbolBits, int bypassBits)
{
    std::shared_ptr<details::IEntropyDecoderImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_shared<details::EntropyDecoderImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_shared<details::EntropyDecoderImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    auto ec = impl->Initialize(pmfLengths, pmfOffsets, pmfTable, symbolBits, bypassBits);
    if (ec) {
        return ec;
    }

    m_impl = std::move(impl);
    return {};
}

}  // namespace msrtc_rans
