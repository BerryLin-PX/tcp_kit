#pragma once

#include <event2/bufferevent.h>
#include <util/func_traits.h>
#include <util/types.h>
#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <network/ev_context.h>
#include <logger/logger.h>

// 通过 filter 介入 tcp 连接的整个生命周期.
// 过滤器链按照顺序依次调度, 在任何一个钩子函数中抛出异常都会打断拦截链, 使连接以错误断开.
namespace tcp_kit {

    struct ev_context;
    struct msg_context;

    class evbuffer_holder {
    public:
        evbuffer* buffer;
        evbuffer_holder(evbuffer* buffer_);

    };

    class msg_buffer {
    public:
        friend class handler_base;

        char*  ptr;
        size_t size;

        msg_buffer(size_t size_);

    private:
        msg_buffer() = default;
        msg_buffer(char* ptr_, size_t size_);

    };

    // 连接建立后回调
    // 参数
    //   @ctx: 事件上下文
    using connect_filter = void (*)(ev_context* ctx);

    using process_chain  = std::unique_ptr<msg_buffer>(*)(msg_context* ctx, std::unique_ptr<msg_buffer>);

    using close_filter = void (*)(ev_context *);

    class filter_chain {

    public:
        connect_filter                     connects;
        std::vector<bufferevent_filter_cb> reads;
        std::vector<bufferevent_filter_cb> writes;
        process_chain                      process;
        close_filter                       closes;

        // template<typename... F>
        // static std::shared_ptr<filter_chain> make(type_list<F...>);

        filter_chain() = default;

    private:

    };

    // SFINAE

    template<typename Func>
    struct get_arg2_type {
        using tp = typename func_traits<Func>::args_type;
        using type = typename std::tuple_element<1, tp>::type;
    };

    template<typename T, typename = void>
    struct check_connect : std::false_type {};

    // 检查 T 是否有静态函数 void connect(ev_context*);
    template<typename T>
    struct check_connect<T, void_t<decltype(T::connect)>> : std::is_same<decltype(T::connect), void(ev_context*)> {};

    // 检查 T 是否有静态函数 void read(evbuffer*, evbuffer*, ev_ssize_t, bufferevent_flush_mode, ev_context*);
    template<typename T, typename = void>
    struct check_read_filter : std::false_type {};

    template<typename T>
    struct check_read_filter<T, void_t<decltype(T::read)>> : std::is_same<decltype(T::read), bufferevent_filter_result(evbuffer*, evbuffer*, ev_ssize_t, bufferevent_flush_mode, ev_context*)> {};

    // 检查 T 是否有静态函数 void write(evbuffer*, evbuffer*, ev_ssize_t, bufferevent_flush_mode, ev_context*);
    template<typename T, typename = void>
    struct check_write_filter : std::false_type {};

    template<typename T>
    struct check_write_filter<T, void_t<decltype(T::write)>> : std::is_same<decltype(T::write), bufferevent_filter_result(evbuffer*, evbuffer*, ev_ssize_t, bufferevent_flush_mode, ev_context*)> {};

    template<typename T, typename = void>
    struct check_close : std::false_type {};

    // 检查 T 是否有静态函数 void close(ev_context *);
    template<typename T>
    struct check_close<T, void_t<decltype(T::close)>> : std::is_same<decltype(T::close), void(ev_context *)> {};

    // 检查某类型为 std::unique_ptr
    template<typename, typename = void>
    struct check_unique : std::false_type {};

    template<typename T>
    struct check_unique<T, void_t<typename T::element_type, typename T::deleter_type>> : std::is_same<T, std::unique_ptr<typename T::element_type, typename T::deleter_type>> {};

    template<typename T>
    struct process_traits;

    template <typename R, typename... Args>
    struct process_traits<R(*)(Args...)>  {
        using result_type = R;
        using args_type = std::tuple<Args...>;
    };

    template <typename R, typename... Args>
    struct process_traits<R(Args...)> : process_traits<R(*)(Args...)> {};

    template <typename T>
    struct process_traits : process_traits<decltype(&T::operator())> {};

    // 检查 T 是否有静态函数 unique_ptr<?,?> process(ev_context*, unique_ptr<?,?>);
    template<typename, typename = void>
    struct check_process_filter : std::false_type { };

    template<typename T>
    struct check_process_filter<T, void_t<decltype(T::process)>> {
        using result_t = typename process_traits<decltype(T::process)>::result_type;
        using arg_t = typename std::tuple_element<1, typename process_traits<decltype(T::process)>::args_type>::type;
        static constexpr bool value = check_unique<result_t>::value &&
                                      check_unique<arg_t>::value &&
                                      std::is_same<decltype(T::process), result_t(msg_context*, arg_t)>::value;
    };

    // 仅保留有 Connect Filter 的过滤器类型
    template <typename First, typename... Others>
    struct valid_connect_filters {
        using types = typename std::conditional<
                check_connect<First>::value,
                typename valid_connect_filters<Others...>::types::template prepend<First>,
                typename valid_connect_filters<Others...>::types>::type;
    };

    template <typename Last>
    struct valid_connect_filters<Last> {
        using types = std::conditional_t<
                check_connect<Last>::value,
                type_list<Last>,
                type_list<>>;
    };

    // 仅保留有 Read Filter 的过滤器类型
    template <typename First, typename... Others>
    struct valid_read_filters {
        using types = typename std::conditional<
                check_read_filter<First>::value,
                typename valid_read_filters<Others...>::types::template prepend<First>,
                typename valid_read_filters<Others...>::types>::type;
    };

    template <typename Last>
    struct valid_read_filters<Last> {
        using types = std::conditional_t<
                check_read_filter<Last>::value,
                type_list<Last>,
                type_list<>>;
    };

    // 仅保留有 Write Filter 的过滤器类型
    template <typename First, typename... Others>
    struct valid_write_filters {
        using types = typename std::conditional<
                check_write_filter<First>::value,
                typename valid_write_filters<Others...>::types::template prepend<First>,
                typename valid_write_filters<Others...>::types>::type;
    };

    template <typename Last>
    struct valid_write_filters<Last> {
        using types = std::conditional_t<
                check_write_filter<Last>::value,
                type_list<Last>,
                type_list<>>;
    };

    // 仅保留有 Process Filter 的过滤器类型
    template <typename First, typename... Others>
    struct valid_process_filters {
        using types = typename std::conditional<
                check_process_filter<First>::value,
                typename valid_process_filters<Others...>::types::template prepend<First>,
                typename valid_process_filters<Others...>::types>::type;
    };

    template <typename Last>
    struct valid_process_filters<Last> {
        using types = std::conditional_t<
                check_process_filter<Last>::value,
                type_list<Last>,
                type_list<>>;
    };

    // 仅保留有 Connect Filter 的过滤器类型
    template <typename First, typename... Others>
    struct valid_close_filters {
        using types = typename std::conditional<
                check_close<First>::value,
                typename valid_close_filters<Others...>::types::template prepend<First>,
                typename valid_close_filters<Others...>::types>::type;
    };

    template <typename Last>
    struct valid_close_filters<Last> {
        using types = std::conditional_t<
                check_close<Last>::value,
                type_list<Last>,
                type_list<>>;
    };

    void empty_connect_chain(ev_context* ctx);

    template<typename First, typename... Others>
    struct connect_chain_caller {

        static void call(ev_context* ctx) {
            First::connect(ctx);
            connect_chain_caller<Others...>::call(ctx);
        }

    };

    template<typename Last>
    struct connect_chain_caller<Last> {

        static void call(ev_context* ctx) {
            Last::connect(ctx);
        }

    };

    void empty_close_chain(ev_context* ctx);

    template<typename First, typename... Others>
    struct close_chain_caller {

        static void call(ev_context* ctx) {
            First::close(ctx);
            close_chain_caller<Others...>::call(ctx);
        }

    };

    template<typename Last>
    struct close_chain_caller<Last> {

        static void call(ev_context* ctx) {
            Last::close(ctx);
        }

    };

    template<typename F>
    bufferevent_filter_result catchable_read(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t dst_limit,
                                             enum bufferevent_flush_mode mode, void *ctx) {
        try {
            return F::read(src, dst, dst_limit, mode, static_cast<ev_context*>(ctx));
        } catch (...) {

        }
    }

    template<typename F>
    bufferevent_filter_result catchable_write(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t dst_limit,
                                             enum bufferevent_flush_mode mode, void *ctx) {
        try {
            return F::write(src, dst, dst_limit, mode, static_cast<ev_context*>(ctx));
        } catch (...) {

        }
    }

    template<typename... F>
    connect_filter make_connect_chain(type_list<F...>, typename std::enable_if<(sizeof...(F) > 0), void>::type* = nullptr) {
        return &connect_chain_caller<F...>::call;
    }

    template<typename... F>
    connect_filter make_connect_chain(type_list<>, typename std::enable_if<(sizeof...(F) == 0), void>::type* = nullptr) {
        return &empty_connect_chain;
    }

    template<typename... F>
    std::vector<bufferevent_filter_cb> make_reads(type_list<F...>, typename std::enable_if<(sizeof...(F) > 0), void>::type* = nullptr) {
        return {&catchable_read<F>()...};
    }

    template<typename... F>
    std::vector<bufferevent_filter_cb> make_reads(type_list<>, typename std::enable_if<(sizeof...(F) == 0), void>::type* = nullptr) {
        return {};
    }

    template<typename... F>
    std::vector<bufferevent_filter_cb> make_writes(type_list<F...>, typename std::enable_if<(sizeof...(F) > 0), void>::type* = nullptr) {
        return {&catchable_write<F>()...};
    }

    template<typename... F>
    std::vector<bufferevent_filter_cb> make_writes(type_list<>, typename std::enable_if<(sizeof...(F) == 0), void>::type* = nullptr) {
        return {};
    }

    template<typename... F>
    connect_filter make_close_chain(type_list<F...>, typename std::enable_if<(sizeof...(F) > 0), void>::type* = nullptr) {
        return &close_chain_caller<F...>::call;
    }

    template<typename... F>
    connect_filter make_close_chain(type_list<>, typename std::enable_if<(sizeof...(F) == 0), void>::type* = nullptr) {
        return &empty_close_chain;
    }

    // 将 Process Filters 调用链展开, A, B, C  -> return C::process(ctx, B::process(ctx, A::process(ctx, move(input))));
    template<typename First, typename... Others>
    struct process_chain_caller {
        static decltype(auto) call(msg_context* ctx, typename get_arg2_type<decltype(First::process)>::type input) {
            return process_chain_caller<Others...>::call(ctx, First::process(ctx, move(input)));
        }
    };

    template<typename Last>
    struct process_chain_caller<Last> {
        static decltype(auto) call(msg_context* ctx, typename get_arg2_type<decltype(Last::process)>::type input) {
            return Last::process(ctx, move(input));
        }
    };

    std::unique_ptr<msg_buffer> empty_process_chain(msg_context* ctx, std::unique_ptr<msg_buffer> in);

//    template<typename... F>
//    std::unique_ptr<msg_buffer> catchable_process_chain(msg_context* ctx, std::unique_ptr<msg_buffer> input) {
//        try {
//            return process_chain_caller<F...>::call(ctx, move(input));
//        } catch (...) {
//
//        }
//    }

    template<typename... F>
    process_chain make_process_chain(type_list<F...>, typename std::enable_if<(sizeof...(F) > 0), void>::type* = nullptr) {
        return &process_chain_caller<F...>::call;
    }

    template<typename... F>
    process_chain make_process_chain(type_list<>, typename std::enable_if<(sizeof...(F) == 0), void>::type* = nullptr) {
        return &empty_process_chain;
    }

//    template<typename... F>
//    std::shared_ptr<filter_chain> filter_chain::make(type_list<F...>) {
//        auto chain = std::make_shared<filter_chain>();
//        chain->connects = make_connect_chain(typename valid_connect_filters<F...>::types{});
//        chain->reads = make_reads(typename valid_read_filters<F...>::types{});
//        chain->writes = make_reads(typename valid_write_filters<F...>::types{});
//        chain->process = make_process_chain(typename valid_process_filters<F...>::types{});
//        return chain;
//    }

    template<typename... F>
    std::shared_ptr<filter_chain> make_filter_chain(type_list<F...>) {
        auto chain = std::make_shared<filter_chain>();
        chain->connects = make_connect_chain(typename valid_connect_filters<F...>::types{});
        chain->reads = make_reads(typename valid_read_filters<F...>::types{});
        chain->writes = make_reads(typename valid_write_filters<F...>::types{});
        chain->process = make_process_chain(typename valid_process_filters<F...>::types{});
        chain->closes = make_close_chain(typename valid_close_filters<F...>::types{});
        return chain;
    }

}

