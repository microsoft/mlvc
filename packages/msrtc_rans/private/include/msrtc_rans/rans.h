// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @file msrtc_rans/rans.h
 * @brief Defines raw primitives of rANS alogirthm
 */

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <tuple>

#if defined(_MSC_VER)
// __umulh
    #include <intrin.h>
#endif

namespace msrtc_rans {

// type holding symbol frequencies
using rans_freq_t = uint32_t;

namespace details {

// 64-bit numbers multiplication outputing high 64-bit
#if defined(_MSC_VER)
static inline uint64_t Mul64Hi(uint64_t a, uint64_t b)
{
    return __umulh(a, b);
}

#elif defined(__SIZEOF_INT128__)
static inline uint64_t Mul64Hi(uint64_t a, uint64_t b)
{
    auto prod = a * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod >> 64);
}

#else
    #error Unknown/unsupported compiler!
#endif

template <typename StateType, typename UnitType>
struct RansConstants {
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    // only N-1 bits are used as exact reciprocal of (N-1)-bit integer takes N bits
    static constexpr inline size_t StateBits = CHAR_BIT * sizeof(state_t) - 1;
    // Maximum possible scale_bits
    static constexpr inline size_t MaxScaleBits = std::min(StateBits - 1, sizeof(freq_t) * CHAR_BIT);
    // The normalization interval lower bound ('l' in the paper): renormalized maximum value
    static constexpr inline state_t LowerBound = static_cast<state_t>(1) << (StateBits - CHAR_BIT * sizeof(unit_t));
};

}  // namespace details

// Prepared encoder symbol description (allows somewhat faster output)
template <typename StateType, typename UnitType>
struct RansEncSymbol {
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    // Max state >> 32
    freq_t m_x_max_hi;
    // Reciprocal shift
    freq_t m_freq_rcp_shift;
    // Fixed-point reciprocal frequency
    StateType m_freq_rcp;
    // Complement of frequency: (1 << scale_bits) - freq
    freq_t m_freq_cmpl;
    // Bias
    freq_t m_bias;

    RansEncSymbol(freq_t start, freq_t freq, freq_t scale_bits);
};

// Raw rANS alogirthm encoder
// Sink need to define:
//  - operator(unit_t) for writing next unit
template <typename StateType, typename UnitType, typename Sink>
class RansEncoder {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    using constants = details::RansConstants<state_t, unit_t>;
    using symbol_t = RansEncSymbol<state_t, unit_t>;

    RansEncoder(Sink&& sink) : m_pair(std::move(sink), constants::LowerBound) {}

    // get sink to fetch result
    Sink& GetSink() { return sink(); }

    // put symbol to stream
    void Put(freq_t start, freq_t freq, freq_t scale_bits);
    void Put(const symbol_t& symbol);

    // flush state to stream
    void Flush();
    // reset state
    void Reset() { state() = constants::LowerBound; }

private:
    // Sink and state
    std::tuple<Sink, state_t> m_pair;

    Sink& sink() { return std::get<0>(m_pair); }
    state_t& state() { return std::get<1>(m_pair); }

    // State renormalization
    state_t Renormalize(state_t x_max);
    // Fast division by frequency
    static state_t Quotient(state_t x, const symbol_t& symbol);
};

// raw put method as defined in the paper
template <typename StateType, typename UnitType, typename Sink>
inline void RansEncoder<StateType, UnitType, Sink>::Put(freq_t start, freq_t freq, freq_t scale_bits)
{
    assert(0 < scale_bits && scale_bits <= constants::MaxScaleBits);
    assert(start < (static_cast<freq_t>(1) << scale_bits));
    assert(freq > 0 && freq <= (static_cast<freq_t>(1) << scale_bits) - start);

    // renormalize if needed
    auto x_max = static_cast<state_t>(freq) << (constants::StateBits - scale_bits);
    auto x = Renormalize(x_max);
    // x = C(s,x)
    x = ((x / freq) << scale_bits) + start + (x % freq);
    state() = x;
}

template <typename StateType, typename UnitType, typename Sink>
inline StateType RansEncoder<StateType, UnitType, Sink>::Renormalize(state_t x_max)
{
    auto x = state();

    while (x >= x_max) {
        sink()(static_cast<unit_t>(x));
        x >>= CHAR_BIT * sizeof(unit_t);

        if constexpr (constants::MaxScaleBits <= CHAR_BIT * sizeof(unit_t)) {
            // one iteration must be enough for such a unit
            assert(x < x_max);
            break;
        }
    }
    return x;
}

// precompile encoder symbol to eliminate division in put operation
template <typename StateType, typename UnitType>
inline RansEncSymbol<StateType, UnitType>::RansEncSymbol(freq_t start, freq_t freq, freq_t scale_bits)
{
    using constants = details::RansConstants<state_t, unit_t>;

    assert(0 < scale_bits && scale_bits <= constants::MaxScaleBits);
    auto scale = static_cast<freq_t>(1) << scale_bits;
    assert(start < scale);
    assert(freq > 0 && freq <= scale - start);

    // high bits of x_max when does not fit into freq_t
    m_x_max_hi = freq << (std::min(constants::StateBits, CHAR_BIT * sizeof(freq_t) - 1) - scale_bits);

    // The original encoder does:
    //   x_new = (x/freq)*scale + start + (x%freq)
    //
    // The fast encoder does (schematically):
    //   q     = mul_hi(x, rcp_freq) >> rcp_shift (division)
    //   r     = x - q*freq                       (remainder)
    //   x_new = q*scale + start + r              (new x)
    //         = x + q*(scale - freq) + start     (substitute r)
    // or defining bias = start, freq_cmpl= scale - freq:
    //         = x + q*freq_cmpl + bias           (*)

    if (freq > 1) {
        // Alverson, "Integer Division using reciprocals"
        // shift=ceil(log2(freq))
        freq_t shift = 1;
        while (freq > (static_cast<freq_t>(1) << shift)) {
            shift++;
        }

        // long divide: ((1 << (shift + bits - 1)) + freq-1) / freq
        if constexpr (std::is_same_v<state_t, uint64_t>) {
            auto x0 = static_cast<state_t>(freq - 1);
            // 32 bits are skipped
            auto x1 = static_cast<state_t>(1) << (shift + 31);

            auto t1 = x1 / freq;
            // and restored
            x0 += (x1 % freq) << 32;
            auto t0 = x0 / freq;

            m_freq_rcp = t0 + (t1 << 32);
        } else {
            static_assert(sizeof(state_t) <= sizeof(uint32_t), "unsupported state size");
            constexpr auto bits = sizeof(state_t) * CHAR_BIT;
            auto nom = (static_cast<uint64_t>(1) << (shift + bits - 1)) + (freq - 1);
            m_freq_rcp = static_cast<state_t>(nom / freq);
        }
        m_freq_rcp_shift = shift - 1;
        m_bias = start;
    } else {
        // freq=0 symbols are never valid to encode, so it doesn't matter what we set our values to.
        //
        // freq=1 is tricky, since the reciprocal of 1 is 1
        // "next best thing": rcp_freq=~0, rcp_shift=0:
        //   q = mul_hi(x, rcp_freq) >> rcp_shift
        //     = mul_hi(x, (1<<64) - 1)) >> 0
        //     = floor(x - x/(2^64))
        //     = x - 1 if 1 <= x < 2^64
        // and we know that x>0 (x=0 is not valid normalization interval).
        //
        // lets choose the other parameters such that
        //   x_new = x*scale + start
        //         = x + q*freq_cmpl + bias       (*)
        // substitute values:
        //         = x + bias + (x - 1)*(scale - 1)
        //         = (x - 1)*scale + bias + 1
        //         = x*scale + (bias + 1 - scale)
        // meaning:
        //   bias = start + scale - 1.
        m_freq_rcp = ~static_cast<state_t>(0);
        m_freq_rcp_shift = 0;
        m_bias = start + scale - 1;
    }
    if constexpr (!std::is_same_v<state_t, uint64_t>) {
        // as we are not using Mul64Hi, we need an additionally shift to get high part of product
        m_freq_rcp_shift += CHAR_BIT * sizeof(state_t);
    }
    m_freq_cmpl = scale - freq;
}

template <typename StateType, typename UnitType, typename Sink>
inline void RansEncoder<StateType, UnitType, Sink>::Put(const symbol_t& symbol)
{
    // renormalize if needed
    auto x_max = static_cast<state_t>(symbol.m_x_max_hi);
    if constexpr (constants::StateBits > CHAR_BIT * sizeof(freq_t) - 1) {
        // restore low bits if state_t is bigger than freq_t
        x_max <<= constants::StateBits - CHAR_BIT * sizeof(freq_t) + 1;
    }
    auto x = Renormalize(x_max);

    // x = C(s,x)
    x += Quotient(x, symbol) * symbol.m_freq_cmpl + symbol.m_bias;
    state() = x;
}

template <typename StateType, typename UnitType, typename Sink>
inline StateType RansEncoder<StateType, UnitType, Sink>::Quotient(state_t x, const symbol_t& symbol)
{
    if constexpr (std::is_same_v<state_t, uint64_t>) {
        return details::Mul64Hi(x, symbol.m_freq_rcp) >> symbol.m_freq_rcp_shift;
    } else {
        static_assert(sizeof(state_t) <= sizeof(uint32_t), "unsupported state size");
        // using 64-bit multiplication
        return static_cast<state_t>((static_cast<uint64_t>(x) * symbol.m_freq_rcp) >> symbol.m_freq_rcp_shift);
    }
}

template <typename StateType, typename UnitType, typename Sink>
inline void RansEncoder<StateType, UnitType, Sink>::Flush()
{
    auto x = state();

    static_assert(sizeof(state_t) % sizeof(unit_t) == 0);

    for (int i = sizeof(state_t) / sizeof(unit_t) - 1; i > 0; i--) {
        sink()(static_cast<unit_t>(x >> i * CHAR_BIT * sizeof(unit_t)));
    }
    sink()(static_cast<unit_t>(x));
}

// Prepared decoder symbol description (no optimization provided)
struct RansDecSymbol {
    using freq_t = rans_freq_t;

    // symbol frequency
    freq_t m_freq;
    // start frequency
    freq_t m_start;

    RansDecSymbol(freq_t start, freq_t freq) : m_freq(freq), m_start(start) { assert(freq > 0); }
};

// Raw rANS alogirthm decoder
// Source needs to define:
//  - operator(unit_t&) for reading next unit
//  - OnOK() for reporting advancing to next symbol
//  - OnInvalidStream() for reporting invalid stream state
template <typename StateType, typename UnitType, typename Source>
class RansDecoder {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    using constants = details::RansConstants<state_t, unit_t>;
    using symbol_t = RansDecSymbol;

    using return_t = decltype(std::declval<Source>().OnOK());

    RansDecoder(Source&& source) : m_pair(std::move(source), constants::LowerBound) {}

    // Get bundled source
    Source& GetSource() { return source(); }
    const Source& GetSource() const { return source(); }

    // Initialize decoder from source
    return_t Init();

    // Get next symbol frequency without advancing over it
    // Frequency needs to be mapped to a symbol
    freq_t Get(freq_t scale_bits);

    // Advance stream by a symbol
    return_t Advance(freq_t start, freq_t freq, freq_t scale_bits);
    return_t Advance(const symbol_t& symbol, freq_t scale_bits)
    {
        return Advance(symbol.m_start, symbol.m_freq, scale_bits);
    }

    // Check whether current state can be EOF state
    bool CheckEOF() const { return state() == constants::LowerBound; }

private:
    // Source and 64-bit state
    std::tuple<Source, state_t> m_pair;

    Source& source() { return std::get<0>(m_pair); }
    const Source& source() const { return std::get<0>(m_pair); }
    state_t& state() { return std::get<1>(m_pair); }
    const state_t& state() const { return std::get<1>(m_pair); }
};

template <typename StateType, typename UnitType, typename Source>
inline typename RansDecoder<StateType, UnitType, Source>::return_t RansDecoder<StateType, UnitType, Source>::Init()
{
    unit_t unit;
    auto rc = source()(unit);
    if (!rc) {
        return rc;
    }
    auto x = static_cast<state_t>(unit);

    static_assert(sizeof(state_t) % sizeof(unit_t) == 0);
    for (size_t i = 1; i < sizeof(state_t) / sizeof(unit_t); i++) {
        rc = source()(unit);
        if (!rc) {
            return rc;
        }
        x += static_cast<state_t>(unit) << i * CHAR_BIT * sizeof(unit_t);
    }
    if (x < constants::LowerBound) {
        return source().OnInvalidStream();
    }
    state() = x;
    return rc;
}

template <typename StateType, typename UnitType, typename Source>
inline rans_freq_t RansDecoder<StateType, UnitType, Source>::Get(freq_t scale_bits)
{
    auto mask = (static_cast<freq_t>(1) << scale_bits) - 1;
    return static_cast<freq_t>(state()) & mask;
}

template <typename StateType, typename UnitType, typename Source>
inline typename RansDecoder<StateType, UnitType, Source>::return_t RansDecoder<StateType, UnitType, Source>::Advance(  //
    freq_t start, freq_t freq, freq_t scale_bits)
{
    auto scale = static_cast<freq_t>(1) << scale_bits;
    assert(start < scale);
    assert(0 < freq && freq <= scale - start);

    auto x = state();
    // s, x = D(x)
    auto value = x & (scale - 1);
    assert(value >= start);
    x = freq * (x >> scale_bits) + value - start;

    // renormalize
    while (x < constants::LowerBound) {
        unit_t unit;
        auto rc = source()(unit);
        if (!rc) {
            return rc;
        }

        x = (x << CHAR_BIT * sizeof(unit_t)) + unit;

        if constexpr (constants::MaxScaleBits <= CHAR_BIT * sizeof(unit_t)) {
            // one iteration must be enough to renormalize
            assert(x >= constants::LowerBound);
            break;
        }
    }
    state() = x;
    return source().OnOK();
}

}  // namespace msrtc_rans
