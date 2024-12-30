#pragma once

template <typename...>
using void_t = void;

template <typename... Types>
struct type_list {
    template <typename T>
    using prepend = type_list<T, Types...>;
};

template <typename List, typename Target, typename Replacement>
struct replace_type;

template <typename First, typename... Rest, typename Target, typename Replacement>
struct replace_type<type_list<First, Rest...>, Target, Replacement> {
    using type = typename replace_type<type_list<Rest...>, Target, Replacement>::type::template prepend<First>;
};

template <typename... Rest, typename Target, typename Replacement>
struct replace_type<type_list<Target, Rest...>, Target, Replacement> {
    using type = typename replace_type<type_list<Rest...>, Target, Replacement>::type::template prepend<Replacement>;
};

template <typename Target, typename Replacement>
struct replace_type<type_list<>, Target, Replacement> {
    using type = type_list<>;
};
