#ifndef TCP_KIT_FUNC_TRAITS_H
#define TCP_KIT_FUNC_TRAITS_H

#include <tuple>

template <typename R, typename... Args>
struct func_traits;

template <typename R, typename... Args>
struct func_traits<R (*)(Args...)> {
    using result_type = R;
    using args_type = std::tuple<typename std::decay<Args>::type...>;
};

#endif
