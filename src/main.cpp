#include <test/blocking_fifo_test.hpp>
#include <test/interruptible_thread_test.hpp>
#include <test/thread_pool_test.hpp>
#include <test/thread_pool_worker_test.hpp>
#include <test/system_util_test.hpp>
#include <test/server_test.hpp>
#include <test/tcp_util_test.hpp>
#include <test/lock_free_queue_test.hpp>
#include <test/lock_free_queue_nb_test.hpp>
#include <util/func_traits.h>
#include <network/filter_chain.h>
#include <test/func_traits_test.h>
#include <string>
#define init_google_test InitGoogleTest

//int main() {
//    testing::init_google_test();
//    return RUN_ALL_TESTS();
//}

template <typename T>
struct get_arg_type;

template <typename R, typename Arg>
struct get_arg_type<R(*)(Arg)> {
    using type = Arg;
};

template <typename R, typename Arg>
struct get_arg_type<R(Arg)> {
    using type = Arg;
};

template <typename R, typename Arg>
struct get_arg_type<R(&)(Arg)> {
    using type = Arg;
};


template <typename First, typename... Others>
struct filters_caller {
    static decltype(auto) call(typename get_arg_type<decltype(First::filter)>::type arg) {
        return filters_caller<Others...>::call(First::filter(arg));
    }
};

template <typename Last>
struct filters_caller<Last> {
    static decltype(auto) call(typename get_arg_type<decltype(Last::filter)>::type arg) {
        return Last::filter(arg);
    }
};

struct a {
    static std::string filter(int a) {
        printf("%d\n", a);
        return "Hello";
    }
};

struct b {
    static float filter(std::string str) {
        printf("%s\n", str.c_str());
        return 3.14f;
    }
};

struct c {
    static bool filter(float f) {
        printf("%f\n", f);
        return true;
    }
};


int main() {
    auto r = filters_caller<a, b, c>::call(10);
    return 0;
}
