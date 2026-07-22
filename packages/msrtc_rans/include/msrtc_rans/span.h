// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @file msrtc_rans/span.h
 * @brief Defines msrtc_rans::span class - simplified version of the STL std::span class for C++17.
 */

#pragma once

#if !defined(__cplusplus)
    #error This header is for C++ only
#endif

#include <array>
#include <cassert>
#include <cstddef>

namespace msrtc_rans {

template <class ElementType>
class span;

// implementation details
namespace detail {

template <class T>
struct is_span_oracle : std::false_type {};

template <class ElementType>
struct is_span_oracle<span<ElementType>> : std::true_type {};

template <class T>
struct is_span : public is_span_oracle<std::remove_cv_t<T>> {};
template <class T>
inline constexpr bool is_span_v = is_span<T>::value;

template <class From, class To>
struct is_allowed_element_type_conversion : std::integral_constant<bool, std::is_convertible_v<From (*)[], To (*)[]>> {};
template <class From, class To>
inline constexpr bool is_allowed_element_type_conversion_v = is_allowed_element_type_conversion<From, To>::value;

template <class Type>
class span_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<Type>;
    using difference_type = std::ptrdiff_t;
    using pointer = Type*;
    using reference = Type&;

    constexpr span_iterator() = default;

    constexpr span_iterator(pointer begin, pointer end, pointer current)
        : m_begin(begin), m_end(end), m_current(current)
    {
    }

    constexpr operator span_iterator<const Type>() const noexcept { return { m_begin, m_end, m_current }; }

    constexpr reference operator*() const noexcept
    {
        assert(m_begin && m_end);
        assert(m_begin <= m_current && m_current < m_end);
        return *m_current;
    }

    constexpr pointer operator->() const noexcept
    {
        assert(m_begin && m_end);
        assert(m_begin <= m_current && m_current < m_end);
        return m_current;
    }
    constexpr span_iterator& operator++() noexcept
    {
        assert(m_begin && m_current && m_end);
        assert(m_current < m_end);
        ++m_current;
        return *this;
    }

    constexpr span_iterator operator++(int) noexcept
    {
        span_iterator ret = *this;
        ++*this;
        return ret;
    }

    constexpr span_iterator& operator--() noexcept
    {
        assert(m_begin && m_end);
        assert(m_begin < m_current);
        --m_current;
        return *this;
    }

    constexpr span_iterator operator--(int) noexcept
    {
        span_iterator ret = *this;
        --*this;
        return ret;
    }

    constexpr span_iterator& operator+=(const difference_type n) noexcept
    {
        if (n != 0) assert(m_begin && m_current && m_end);
        if (n > 0) assert(m_end - m_current >= n);
        if (n < 0) assert(m_current - m_begin >= -n);
        m_current += n;
        return *this;
    }

    constexpr span_iterator operator+(const difference_type n) const noexcept
    {
        span_iterator ret = *this;
        ret += n;
        return ret;
    }

    friend constexpr span_iterator operator+(const difference_type n, const span_iterator& rhs) noexcept
    {
        return rhs + n;
    }

    constexpr span_iterator& operator-=(const difference_type n) noexcept
    {
        if (n != 0) assert(m_begin && m_current && m_end);
        if (n > 0) assert(m_current - m_begin >= n);
        if (n < 0) assert(m_end - m_current >= -n);
        m_current -= n;
        return *this;
    }

    constexpr span_iterator operator-(const difference_type n) const noexcept
    {
        span_iterator ret = *this;
        ret -= n;
        return ret;
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, difference_type> operator-(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        assert(m_begin == rhs.m_begin && m_end == rhs.m_end);
        return m_current - rhs.m_current;
    }

    constexpr reference operator[](const difference_type n) const noexcept { return *(*this + n); }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator==(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        assert(m_begin == rhs.m_begin && m_end == rhs.m_end);
        return m_current == rhs.m_current;
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator!=(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator<(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        assert(m_begin == rhs.m_begin && m_end == rhs.m_end);
        return m_current < rhs.m_current;
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator>(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        return rhs < *this;
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator<=(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        return !(rhs < *this);
    }

    template <class Type2>
    constexpr std::enable_if_t<std::is_same_v<std::remove_cv_t<Type2>, value_type>, bool> operator>=(  //
        const span_iterator<Type2>& rhs) const noexcept
    {
        return !(*this < rhs);
    }

#if defined(_MSC_VER)
    // MSVC++ iterator debugging support; allows STL algorithms in 15.8+ to unwrap span_iterator to a pointer type
    // after a range check in STL algorithm calls
    using _Unchecked_type = pointer;

    friend constexpr void _Verify_range([[maybe_unused]] span_iterator lhs, [[maybe_unused]] span_iterator rhs) noexcept
    {
        // test that [lhs, rhs) forms a valid range inside an STL algorithm
        assert(lhs.m_begin == rhs.m_begin && lhs.m_end == rhs.m_end  // range spans have to match
               && lhs.m_current <= rhs.m_current);                   // range must not be transposed
    }

    constexpr void _Verify_offset(const difference_type n) const noexcept
    {
        // test that *this + n is within the range of this call
        if (n != 0) assert(m_begin && m_current && m_end);
        if (n > 0) assert(m_end - m_current >= n);
        if (n < 0) assert(m_current - m_begin >= -n);
    }

    constexpr pointer _Unwrapped() const noexcept
    {
        // after seeking *this to a high water mark, or using one of the _Verify_xxx functions above,
        // unwrap this span_iterator to a raw pointer
        return m_current;
    }

    // Tell the STL that span_iterator should not be unwrapped if it can't validate in advance,
    // even in release / optimized builds:
    static constexpr bool _Unwrap_when_unverified = false;

    constexpr void _Seek_to(const pointer p) noexcept
    {
        // adjust the position of *this to previously verified location p after _Unwrapped
        m_current = p;
    }
#endif  // ^^^ _MSC_VER

    pointer m_begin = nullptr;
    pointer m_end = nullptr;
    pointer m_current = nullptr;
};

}  // namespace detail

// [span], class template span
template <class ElementType>
class span {
public:
    // constants and types
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using pointer = element_type*;
    using const_pointer = const element_type*;
    using reference = element_type&;
    using const_reference = const element_type&;
    using difference_type = std::ptrdiff_t;

    using iterator = detail::span_iterator<ElementType>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    constexpr span() noexcept : m_data(nullptr), m_size(0) {}

    constexpr span(pointer data, size_type count) noexcept : m_data(data), m_size(count) {}

    constexpr span(pointer firstElem, pointer lastElem) noexcept
        : span(firstElem, static_cast<std::size_t>(lastElem - firstElem))
    {
        assert(firstElem <= lastElem);
    }

    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : span(arr, N)
    {
    }

    template <class T, std::size_t N, std::enable_if_t<detail::is_allowed_element_type_conversion_v<const T, element_type>, int> = 0>
    constexpr span(const std::array<T, N>& arr) noexcept : span(arr.data(), N)
    {
    }

    template <class Container,
              std::enable_if_t<!detail::is_span_v<Container> && std::is_pointer_v<decltype(std::declval<Container&>().data())>
                                   && detail::is_allowed_element_type_conversion_v<
                                       std::remove_pointer_t<decltype(std::declval<Container&>().data())>, element_type>,
                               int> = 0>
    constexpr span(Container& cont) noexcept : span(cont.data(), cont.size())
    {
    }

    template <class Container,
              std::enable_if_t<std::is_const_v<element_type> && !detail::is_span_v<Container>
                                   && std::is_pointer_v<decltype(std::declval<const Container&>().data())>
                                   && detail::is_allowed_element_type_conversion_v<
                                       std::remove_pointer_t<decltype(std::declval<const Container&>().data())>, element_type>,
                               int> = 0>
    constexpr span(const Container& cont) noexcept : span(cont.data(), cont.size())
    {
    }

    constexpr span(const span& other) noexcept = default;

    template <class Type2, std::enable_if_t<detail::is_allowed_element_type_conversion_v<Type2, element_type>, int> = 0>
    constexpr span(const span<Type2>& other) noexcept : span(other.data(), other.size())
    {
    }

    ~span() noexcept = default;
    constexpr span& operator=(const span& other) noexcept = default;

    constexpr span<element_type> first(size_type count) const noexcept
    {
        assert(count <= size());
        return { data(), count };
    }

    constexpr span<element_type> last(size_type count) const noexcept
    {
        assert(count <= size());
        return { data() + size() - count, count };
    }

    constexpr span<element_type> subspan(size_type offset) const noexcept
    {
        assert(offset <= size());
        return { data() + offset, size() - offset };
    }

    constexpr span<element_type> subspan(size_type offset, size_type count) const noexcept
    {
        assert(offset <= size() && offset + count <= size());
        return { data() + offset, count };
    }

    // [span.obs], span observers
    constexpr size_type size() const noexcept { return m_size; }

    constexpr bool is_empty() const noexcept { return size() == 0; }

    // [span.elem], span element access
    constexpr reference operator[](size_type idx) const noexcept
    {
        assert(idx < size());
        return data()[idx];
    }

    constexpr reference front() const noexcept
    {
        assert(size() > 0);
        return data()[0];
    }

    constexpr reference back() const noexcept
    {
        assert(size() > 0);
        return data()[size() - 1];
    }

    constexpr pointer data() const noexcept { return m_data; }

    // [span.iter], span iterator support
    constexpr iterator begin() const noexcept
    {
        const auto dataPtr = data();
        const auto endPtr = dataPtr + size();
        return { dataPtr, endPtr, dataPtr };
    }

    constexpr iterator end() const noexcept
    {
        const auto dataPtr = data();
        const auto endPtr = dataPtr + size();
        return { dataPtr, endPtr, endPtr };
    }

    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator{ end() }; }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator{ begin() }; }

#if defined(_MSC_VER)
    // Tell MSVC how to unwrap spans in range-based-for
    constexpr pointer _Unchecked_begin() const noexcept { return data(); }
    constexpr pointer _Unchecked_end() const noexcept { return data() + size(); }
#endif  // ^^^ _MSC_VER

private:
    pointer m_data;
    size_t m_size;
};

// Deduction guides
template <class Container, class Element = std::remove_pointer_t<decltype(std::declval<Container&>().data())>>
span(Container&) -> span<Element>;

template <class Container, class Element = std::remove_pointer_t<decltype(std::declval<const Container&>().data())>>
span(const Container&) -> span<Element>;

}  // namespace msrtc_rans
