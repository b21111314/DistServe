/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>

#include <cute/container/tuple.hpp>
#include <cute/container/array.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/numeric/integral_constant.hpp>

/** IntTuple is an integer or a tuple of IntTuples.
 * This file holds utilities for working with IntTuples,
 * but does not hold a concrete concept or class of IntTuple.
 */ 

namespace cute
{

// Implementation of get<0>(Integral).
//   Even though is_tuple<Integral> is false and tuple_size<Integral> doesn't compile,
//   CuTe defines rank(Integral) as 1, so it's useful for get<0>(Integral) to return its input
template <size_t I, class T, __CUTE_REQUIRES(cute::is_integral<cute::remove_cvref_t<T>>::value)>
CUTE_HOST_DEVICE constexpr 
decltype(auto)
get(T&& t) noexcept
{
  static_assert(I == 0, "Index out of range");
  return static_cast<T&&>(t);
}

// Custom recursive get for anything that implements get<I>(.) (for a single integer I).
template <size_t I0, size_t I1, size_t... Is, class T>
CUTE_HOST_DEVICE constexpr 
decltype(auto)
get(T&& t) noexcept
{
  return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}

//
// rank
//

template <int... Is, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
rank(IntTuple const& t)
{
  if constexpr (sizeof...(Is) == 0) {
    if constexpr (is_tuple<IntTuple>::value) {
      return Int<tuple_size<IntTuple>::value>{};
    } else {
      return Int<1>{};
    }
  } else {
    return rank(get<Is...>(t));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class IntTuple>
using rank_t = decltype(rank(declval<IntTuple>()));

template <class IntTuple>
static constexpr int rank_v = rank_t<IntTuple>::value;

//
// shape
//

template <class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
shape(IntTuple const& s)
{
  if constexpr (is_tuple<IntTuple>::value) {
    return transform(s, [](auto const& a) { return shape(a); });
  } else {
    return s;
  }

  CUTE_GCC_UNREACHABLE;
}

template <int I, int... Is, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
shape(IntTuple const& s)
{
  if constexpr (is_tuple<IntTuple>::value) {
    return shape<Is...>(get<I>(s));
  } else {
    return get<I,Is...>(shape(s));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// max
//

template <class T0, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
max(T0 const& t0, Ts const&... ts)
{
  if constexpr (is_tuple<T0>::value) {
    return cute::max(cute::apply(t0, [](auto const&... a){ return cute::max(a...); }), ts...);
  } else if constexpr (sizeof...(Ts) == 0) {
    return t0;
  } else {
    return cute::max(t0, cute::max(ts...));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// min
//

template <class T0, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
min(T0 const& t0, Ts const&... ts)
{
  if constexpr (is_tuple<T0>::value) {
    return cute::min(cute::apply(t0, [](auto const&... a){ return cute::min(a...); }), ts...);
  } else if constexpr (sizeof...(Ts) == 0) {
    return t0;
  } else {
    return cute::min(t0, cute::min(ts...));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// gcd
//

template <class T0, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
gcd(T0 const& t0, Ts const&... ts)
{
  if constexpr (is_tuple<T0>::value) {
    return cute::gcd(cute::apply(t0, [](auto const&... a){ return cute::gcd(a...); }), ts...);
  } else if constexpr (sizeof...(Ts) == 0) {
    return t0;
  } else {
    return cute::gcd(t0, cute::gcd(ts...));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// depth
//

template <int... Is, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
depth(IntTuple const& t)
{
  if constexpr (sizeof...(Is) == 0) {
    if constexpr (is_tuple<IntTuple>::value) {
      return Int<1>{} + cute::apply(t, [](auto const&... v){ return cute::max(depth(v)...); });
    } else {
      return Int<0>{};
    }
  } else {
    return depth(get<Is...>(t));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Tuple>
using depth_t = decltype(depth(declval<Tuple>()));

template <class Tuple>
static constexpr int depth_v = depth_t<Tuple>::value;

//
// product
//

template <class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
product(IntTuple const& a)
{
  if constexpr (is_tuple<IntTuple>::value) {
    return cute::apply(a, [](auto const&... v){ return (Int<1>{} * ... * product(v)); });
  } else {
    return a;
  }

  CUTE_GCC_UNREACHABLE;
}

// Return a rank(t) tuple @a result such that get<i>(@a result) = product(get<i>(@a t))
template <class Tuple>
CUTE_HOST_DEVICE constexpr
auto
product_each(Tuple const& t)
{
  return transform(wrap(t), [](auto const& x) { return product(x); });
}

// Take the product of Tuple at the leaves of TupleG
template <class Tuple, class TupleG>
CUTE_HOST_DEVICE constexpr
auto
product_like(Tuple const& tuple, TupleG const& guide)
{
  return transform_leaf(guide, tuple, [](auto const& g, auto const& t) { return product(t); });
}

// Return the product of elements in a mode
template <int... Is, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
size(IntTuple const& a)
{
  if constexpr (sizeof...(Is) == 0) {
    return product(a);
  } else {
    return product(get<Is...>(a));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class IntTuple>
static constexpr int size_v = decltype(size(declval<IntTuple>()))::value;

//
// sum
//

template <class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
sum(IntTuple const& a)
{
  if constexpr (is_tuple<IntTuple>::value) {
    return cute::apply(a, [](auto const&... v){ return (Int<0>{} + ... + sum(v)); });
  } else {
    return a;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// inner_product
//

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
inner_product(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    static_assert(tuple_size<IntTupleA>::value == tuple_size<IntTupleB>::value, "Mismatched ranks");
    return transform_apply(a, b, [](auto const& x, auto const& y) { return inner_product(x,y); },
                                 [](auto const&... v) { return (Int<0>{} + ... + v); });
  } else {
    return a * b;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// ceil_div
//

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
ceil_div(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    static_assert(tuple_size<IntTupleA>::value >= tuple_size<IntTupleB>::value, "Mismatched ranks");
    constexpr int R = tuple_size<IntTupleA>::value;        // Missing ranks in TupleB are implictly 1
    return transform(a, append<R>(b,Int<1>{}), [](auto const& x, auto const& y) { return ceil_div(x,y); });
  } else {
    return (a + b - Int<1>{}) / b;
  }

  CUTE_GCC_UNREACHABLE;
}

/** Division for Shapes
 * Case Tuple Tuple:
 *   Perform shape_div element-wise
 * Case Tuple Int:
 *   Fold the division of b across each element of a
 *   Example: shape_div((4,5,6),40) -> shape_div((1,5,6),10) -> shape_div((1,1,6),2) -> (1,1,3)
 * Case Int Tuple:
 *   Return shape_div(a, product(b))
 * Case Int Int:
 *   Enforce the divisibility condition a % b == 0 || b % a == 0 when possible
 *   Return a / b with rounding away from 0 (that is, 1 or -1 when a < b)
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
shape_div(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value) {
    if constexpr (is_tuple<IntTupleB>::value) {  // tuple tuple
      static_assert(tuple_size<IntTupleA>::value == tuple_size<IntTupleB>::value, "Mismatched ranks");
      return transform(a, b, [](auto const& x, auto const& y) { return shape_div(x,y); });
    } else {                                     // tuple int
      auto const [result, rest] = fold(a, cute::make_tuple(cute::make_tuple(), b),
        [] (auto const& init, auto const& ai) {
          return cute::make_tuple(append(get<0>(init), shape_div(ai, get<1>(init))), shape_div(get<1>(init), ai));
        });
      return result;
    }
  } else
  if constexpr (is_tuple<IntTupleB>::value) {    // int tuple
    return shape_div(a, product(b));
  } else
  if constexpr (is_static<IntTupleA>::value && is_static<IntTupleB>::value) {
    static_assert(IntTupleA::value % IntTupleB::value == 0 || IntTupleB::value % IntTupleA::value == 0, "Static shape_div failure");
    return C<shape_div(IntTupleA::value, IntTupleB::value)>{};
  } else {                                       // int int   
    //assert(a % b == 0 || b % a == 0);          // Wave dynamic assertion
    return a / b != 0 ? a / b : signum(a) * signum(b);  // Division with rounding away from zero
  }

  CUTE_GCC_UNREACHABLE;
}

/** Minimum for Shapes
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
shape_min(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value || is_tuple<IntTupleB>::value) {
    static_assert(dependent_false<IntTupleA>, "Not implemented.");
  } else
  if constexpr (is_constant<1, IntTupleA>::value || is_constant<1, IntTupleB>::value) {
    return Int<1>{};            // _1 is less than all other shapes, preserve static
  } else {
    return cute::min(a, b);
  }

  CUTE_GCC_UNREACHABLE;
}

/** Return a tuple the same profile as A scaled by corresponding elements in B
 */
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
elem_scale(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    return transform(a, b, [](auto const& x, auto const& y) { return elem_scale(x,y); });
  } else {
    return a * product(b);
  }

  CUTE_GCC_UNREACHABLE;
}

/** Test if two IntTuple have the same profile (hierarchical rank division)
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
congruent(IntTupleA const& a, IntTupleB const& b)
{
  return bool_constant<is_same<decltype(repeat_like(shape(a),_0{})),
                               decltype(repeat_like(shape(b),_0{}))>::value>{};
}

template <class A, class B>
using is_congruent = decltype(congruent(declval<A>(), declval<B>()));

/** Test if two IntTuple have the similar profiles up to Shape A (hierarchical rank division)
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
weakly_congruent(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    if constexpr (tuple_size<IntTupleA>::value != tuple_size<IntTupleB>::value) {
      return false_type{};
    } else {
      return transform_apply(a, b, [](auto const& x, auto const& y) { return weakly_congruent(x,y); },
                                   [](auto const&... z) { return (true_type{} && ... && z); });
    }
  } else if constexpr (is_integral<IntTupleA>::value) {
    return true_type{};
  } else if constexpr (is_integral<IntTupleB>::value) {
    return false_type{};
  } else {
    return weakly_congruent(shape(a), shape(b));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class A, class B>
using is_weakly_congruent = decltype(weakly_congruent(declval<A>(), declval<B>()));

/** Test if Shape B is compatible with Shape A:
 * Any coordinate into A can also be used as a coordinate into B
 * A <= B is a partially ordered set of factored shapes
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
compatible(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    if constexpr (tuple_size<IntTupleA>::value != tuple_size<IntTupleB>::value) {
      return false_type{};
    } else {
      return transform_apply(a, b, [](auto const& x, auto const& y) { return compatible(x,y); },
                                   [](auto const&... z) { return (true_type{} && ... && z); });
    }
  } else if constexpr (is_integral<IntTupleA>::value) {
    return a == size(b);
  } else if constexpr (is_integral<IntTupleB>::value) {
    return false_type{};
  } else {
    return compatible(shape(a), shape(b));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class A, class B>
using is_compatible = decltype(compatible(declval<A>(), declval<B>()));

/** Test if Shape B is weakly compatible with Shape A:
 * Shape B divides Shape A at some level of refinement
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
weakly_compatible(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    if constexpr (tuple_size<IntTupleA>::value != tuple_size<IntTupleB>::value) {
      return false_type{};
    } else {
      return transform_apply(a, b, [](auto const& x, auto const& y) { return weakly_compatible(x,y); },
                                   [](auto const&... z) { return (true_type{} && ... && z); });
    }
  } else if constexpr (is_integral<IntTupleA>::value) {
    return a % size(b) == Int<0>{};
  } else if constexpr (is_integral<IntTupleB>::value) {
    return false_type{};
  } else {
    return weakly_compatible(shape(a), shape(b));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class A, class B>
using is_weakly_compatible = decltype(weakly_compatible(declval<A>(), declval<B>()));

/** Replace the elements of Tuple B that are paired with an Int<0> with an Int<1>
 */
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value) {
    return transform(a, b, [](auto const& x, auto const& y) { return filter_zeros(x,y); });
  } else if constexpr (is_constant<0, IntTupleA>::value) {
    return Int<1>{};
  } else {
    return b;
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Tuple>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tuple const& t)
{
  return filter_zeros(t, t);
}

//
// Converters and constructors with arrays and params
//

/** Make an IntTuple of rank N from an Indexable array.
 * Access elements up to a dynamic index n, then use init (requires compatible types)
 * Consider cute::take<B,E> if all indexing is known to be valid
 * \code
 *   std::vector<int> a = {6,3,4};
 *   auto tup = make_int_tuple<5>(a, a.size(), 0)            // (6,3,4,0,0)
 * \endcode
 */
template <int N, class Indexable, class T>
CUTE_HOST_DEVICE constexpr
auto
make_int_tuple(Indexable const& t, int n, T const& init)
{
  static_assert(N > 0);
  if constexpr (N == 1) {
    return 0 < n ? t[0] : init;
  } else {
    return transform(make_seq<N>{}, [&](auto i) { return i < n ? t[i] : init; });
  }

  CUTE_GCC_UNREACHABLE;
}

/** Fill the dynamic values of a Tuple with values from another Tuple
 * \code
 *   auto params = make_tuple(6,3,4);
 *   cute::tuple<Int<1>, cute::tuple<int, int, Int<3>>, int, Int<2>> result;
 *   fill_int_tuple_from(result, params);                    // (_1,(6,3,_3),4,_2)
 * \endcode
 */
template <class Tuple, class TupleV>
CUTE_HOST_DEVICE constexpr
auto
fill_int_tuple_from(Tuple& result, TupleV const& vals)
{
  return fold(result, vals, [](auto const& init, auto&& r) {
    if constexpr (is_static<remove_cvref_t<decltype(r)>>::value) {       // Skip static elements of result
      return init;
    } else if constexpr (is_tuple<remove_cvref_t<decltype(r)>>::value) { // Recurse into tuples
      return fill_int_tuple_from(r, init);
    } else {                                                             // Assign and consume arg
      static_assert(tuple_size<remove_cvref_t<decltype(init)>>::value > 0, "Not enough values to fill with!");
      r = get<0>(init);
      return remove<0>(init);
    }

    CUTE_GCC_UNREACHABLE;
  });
}

/** Make a "Tuple" by filling in the dynamic values in order from the arguments
 * \code
 *   using result_t = cute::tuple<Int<1>, cute::tuple<int, int, Int<3>>, int, Int<2>>;
 *   auto result = make_int_tuple_from<result_t>(6,3,4);     // (_1,(6,3,_3),4,_2)
 * \endcode
 */
template <class Tuple, class... Ts>
CUTE_HOST_DEVICE constexpr
Tuple
make_int_tuple_from(Ts const&... ts)
{
  Tuple result = Tuple{};
  fill_int_tuple_from(result, cute::make_tuple(ts...));
  return result;
}

/** Convert a tuple to a flat homogeneous array of type T
 * \code
 *   auto tup = cute::make_tuple(Int<1>{}, cute::make_tuple(6,3,Int<3>{}),4,Int<2>{});
 *   cute::array<uint64_t,6> result = to_array<uint64_t>(tup);   // [1,6,3,3,4,2]
 * \endcode
 */
template <class T = int64_t, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
to_array(IntTuple const& t)
{
  auto flat_t = flatten_to_tuple(t);
  constexpr int N = tuple_size<decltype(flat_t)>::value;
  cute::array<T,N> result;
  for_each(make_seq<N>{}, [&] (auto i) { result[i] = get<i>(flat_t); });
  return result;
}

//
// Comparison operators
//

//
// There are many ways to compare tuple of elements and because CuTe is built
//   on parameterizing layouts of coordinates, some comparisons are appropriate
//   only in certain cases.
//  -- lexicographical comparison [reverse, reflected, revref]   : Correct for coords in RowMajor Layout
//  -- colexicographical comparison [reverse, reflected, revref] : Correct for coords in ColMajor Layout
//  -- element-wise comparison [any,all]                         :
// This can be very confusing. To avoid errors in selecting the appropriate
//   comparison, op<|op<=|op>|op>= are *not* implemented for cute::tuple.
//
// When actually desiring to order coordinates, the user should map them to
//   their indices within the Layout they came from:
//      e.g.  layoutX(coordA) < layoutX(coordB)
// That said, we implement the three most common ways to compare tuples below.
//   These are implemented with slighly more explicit names than op<.
//

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
lex_less(IntTupleA const& a, IntTupleB const& b);

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
colex_less(IntTupleA const& a, IntTupleB const& b);

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
elem_less(IntTupleA const& a, IntTupleB const& b);

namespace detail {

template <size_t I, class TupleA, class TupleB>
CUTE_HOST_DEVICE constexpr
auto
lex_less_impl(TupleA const& a, TupleB const& b)
{
  if constexpr (I == tuple_size<TupleB>::value) {
    return cute::false_type{};    // Terminal: TupleB is exhausted
  } else if constexpr (I == tuple_size<TupleA>::value) {
    return cute::true_type{};     // Terminal: TupleA is exhausted, TupleB is not exhausted
  } else {
    return lex_less(get<I>(a), get<I>(b)) || (get<I>(a) == get<I>(b) && lex_less_impl<I+1>(a,b));
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I, class TupleA, class TupleB>
CUTE_HOST_DEVICE constexpr
auto
colex_less_impl(TupleA const& a, TupleB const& b)
{
  if constexpr (I == tuple_size<TupleB>::value) {
    return cute::false_type{};    // Terminal: TupleB is exhausted
  } else if constexpr (I == tuple_size<TupleA>::value) {
    return cute::true_type{};     // Terminal: TupleA is exhausted, TupleB is not exhausted
  } else {
    constexpr size_t A = tuple_size<TupleA>::value - 1 - I;
    constexpr size_t B = tuple_size<TupleB>::value - 1 - I;
    return colex_less(get<A>(a), get<B>(b)) || (get<A>(a) == get<B>(b) && colex_less_impl<I+1>(a,b));
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I, class TupleA, class TupleB>
CUTE_HOST_DEVICE constexpr
auto
elem_less_impl(TupleA const& a, TupleB const& b)
{
  if constexpr (I == tuple_size<TupleA>::value) {
    return cute::true_type{};     // Terminal: TupleA is exhausted
  } else if constexpr (I == tuple_size<TupleB>::value) {
    return cute::false_type{};    // Terminal: TupleA is not exhausted, TupleB is exhausted
  } else {
    return elem_less(get<I>(a), get<I>(b)) && elem_less_impl<I+1>(a,b);
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Lexicographical comparison

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
lex_less(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    return detail::lex_less_impl<0>(a, b);
  } else {
    return a < b;
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
lex_leq(T const& t, U const& u) {
  return !lex_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
lex_gtr(T const& t, U const& u) {
  return lex_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
lex_geq(T const& t, U const& u) {
  return !lex_less(t, u);
}

// Colexicographical comparison

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
colex_less(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    return detail::colex_less_impl<0>(a, b);
  } else {
    return a < b;
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
colex_leq(T const& t, U const& u) {
  return !colex_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
colex_gtr(T const& t, U const& u) {
  return colex_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
colex_geq(T const& t, U const& u) {
  return !colex_less(t, u);
}

// Elementwise [all] comparison

template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto
elem_less(IntTupleA const& a, IntTupleB const& b)
{
  if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
    return detail::elem_less_impl<0>(a, b);
  } else {
    return a < b;
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
elem_leq(T const& t, U const& u) {
  return !elem_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
elem_gtr(T const& t, U const& u) {
  return elem_less(u, t);
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
auto
elem_geq(T const& t, U const& u) {
  return !elem_less(t, u);
}

/** Increment a (dynamic) coord lexicographically within a shape
 * \code
 *    auto shape = make_shape(1,2,make_shape(2,3),3);
 *
 *   int i = 0;
 *   for (auto coord = repeat_like(shape, 0); back(coord) != back(shape); increment(coord, shape)) {
 *      std::cout << i++ << ": " << coord << std::endl;
 *   }
 *   assert(i == size(shape));
 * \endcode
 */
template <class Coord, class Shape>
CUTE_HOST_DEVICE constexpr
void
increment(Coord& coord, Shape const& shape);

namespace detail {

template <class Coord, class Shape, int I0, int... Is>
CUTE_HOST_DEVICE constexpr
void
increment(Coord& coord, Shape const& shape, seq<I0,Is...>)
{
  cute::increment(get<I0>(coord), get<I0>(shape));
  if constexpr (sizeof...(Is) != 0) {
    if (back(get<I0>(coord)) == back(get<I0>(shape))) {
      back(get<I0>(coord)) = 0;
      increment(coord, shape, seq<Is...>{});
    }
  }
}

} // end namespace detail

template <class Coord, class Shape>
CUTE_HOST_DEVICE constexpr
void
increment(Coord& coord, Shape const& shape)
{
  if constexpr (is_integral<Coord>::value && is_integral<Shape>::value) {
    ++coord;
  } else if constexpr (is_tuple<Coord>::value && is_tuple<Shape>::value) {
    static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched ranks");
    detail::increment(coord, shape, tuple_seq<Coord>{});
  } else {
    static_assert(sizeof(Coord) == 0, "Invalid parameters");
  }
}

struct ForwardCoordIteratorSentinal
{};

// A forward iterator for a starting coordinate in a shape's domain, and a shape.
// The starting coordinate may be zero but need not necessarily be.
template <class Coord, class Shape>
struct ForwardCoordIterator
{
  static_assert(is_congruent<Coord, Shape>::value);

  CUTE_HOST_DEVICE constexpr
  Coord const& operator*() const { return coord; }

  CUTE_HOST_DEVICE constexpr
  ForwardCoordIterator& operator++() { increment(coord, shape); return *this; }

  // Sentinel for the end of the implied range
  CUTE_HOST_DEVICE constexpr
  bool operator< (ForwardCoordIteratorSentinal const&) const { return back(coord) <  back(shape); }
  CUTE_HOST_DEVICE constexpr
  bool operator==(ForwardCoordIteratorSentinal const&) const { return back(coord) == back(shape); }
  CUTE_HOST_DEVICE constexpr
  bool operator!=(ForwardCoordIteratorSentinal const&) const { return back(coord) != back(shape); }
  // NOTE: These are expensive, avoid use
  CUTE_HOST_DEVICE constexpr
  bool operator< (ForwardCoordIterator const& other) const { return colex_less(coord, other.coord); }
  CUTE_HOST_DEVICE constexpr
  bool operator==(ForwardCoordIterator const& other) const { return coord == other.coord; }
  CUTE_HOST_DEVICE constexpr
  bool operator!=(ForwardCoordIterator const& other) const { return coord != other.coord; }

  Coord coord;
  Shape const& shape;
};

// A forward iterator for a coordinate that starts from a provided coordinate
template <class Shape, class Coord>
CUTE_HOST_DEVICE constexpr
auto
make_coord_iterator(Coord const& coord, Shape const& shape)
{
  return ForwardCoordIterator<Coord,Shape>{coord,shape};
}

// A forward iterator for a coordinate that starts from zero
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_coord_iterator(Shape const& shape)
{
  auto coord = repeat_like(shape, int(0));
  return make_coord_iterator(coord, shape);
}

} // end namespace cute
