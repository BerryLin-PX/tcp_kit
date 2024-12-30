#pragma once

#include <tuple>
#include <functional>

namespace tcp_kit {

    template<typename T>
    struct func_traits;

    template <typename R, typename... Args>
    struct func_traits<R(*)(Args...)>  {
        using result_type = R;
        using args_type = std::tuple<typename std::decay<Args>::type...>;
    };

    template <typename R, typename... Args>
    struct func_traits<R(Args...)> : func_traits<R(*)(Args...)> {};

    template <typename T>
    struct func_traits : func_traits<decltype(&T::operator())> {};

    template <typename C, typename R, typename... Args>
    struct func_traits<R (C::*)(Args...)> : func_traits<R (*)(Args...)> {};

    template <typename C, typename R, typename... Args>
    struct func_traits<R (C::*)(Args...) const> : func_traits<R (*)(Args...)> {};

    template<size_t... I>
    struct index_seq {};

    // 此模版类移植于 C++14 std::index_sequence, 递归实例化一个 index_seq<1, 2, 3, ...., n> 的类型
    // using seq_t = typename index_seq_h<5>::seq_t;
    // 等同于:
    // using seq_t = index_seq<0, 1, 2, 3, 4, 5>;
    template<size_t N, size_t... I>
    struct index_seq_h {
        using seq_t = typename index_seq_h<N - 1, N, I...>::seq_t;
    };

    template<size_t... I>
    struct index_seq_h<0, I...> {
        using seq_t = index_seq<0, I...>;
    };

    template<typename Tuple>
    auto make_index_seq() {
        using seq_t = typename index_seq_h<std::tuple_size<Tuple>::value - 1>::seq_t;
        return seq_t{};
    }

    template<typename Function,typename... Args, size_t... I>
    decltype(auto) call_helper(Function f, std::tuple<Args...>&& params, index_seq<I...>) {
        return f(move(std::get<I>(params))...);
    }

    template<typename Function, typename... Args>
    decltype(auto) call(Function f, std::tuple<Args...>&& params) {
        return call_helper(f, std::forward<std::tuple<Args...>>(params),
                           make_index_seq<std::tuple<Args...>>());
    }

}
