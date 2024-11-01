#ifndef TCP_KIT_FILTER_H
#define TCP_KIT_FILTER_H

#include <vector>
#include <event2/bufferevent.h>
#include <network/event_context.h>
#include <util/func_traits.h>
#include <logger/logger.h>

namespace tcp_kit {

    using namespace std;

    class raw_buffer {

    public:
        bufferevent* bev;
        raw_buffer(bufferevent* bev_);

    };

    // 同 libevent 中 bufferevent 声明的回调函数 bufferevent_filter_cb:
    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // A callback function to implement a builtin for a bufferevent.
    //
    // Parameters
    //   @src:   An evbuffer to drain data from.
    //   @dst:   An evbuffer to add data to.
    //   @limit: A suggested upper bound of bytes to write to dst. The builtin may ignore this value, but doing so means that it will overflow the high-water mark associated with dst. -1 means "no limit".
    //   @mode:  Whether we should write data as may be convenient (BEV_NORMAL), or flush as much data as we can (BEV_FLUSH), or flush as much as we can, possibly including an end-of-stream marker (BEV_FINISH).
    //   @ctx:   A user-supplied pointer.
    //
    //  Returns
    //     BEV_OK if we wrote some data;
    //     BEV_NEED_MORE if we can't produce any more output until we get some arg;
    //     BEV_ERROR on an error.
    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // 实现 bufferevent 过滤器的回调函数。
    //
    // 参数
    //   @src:   从中提取数据的 evbuffer。
    //   @dst:   将数据添加到的 evbuffer。
    //   @limit: 向 dst 写入的字节数建议上限。过滤器可以忽略该值，但忽略可能导致 dst 的高水位标记溢出。-1 表示“无限制”。
    //   @mode:  指定写入数据的模式，可以是以下之一：BEV_NORMAL（按方便方式写入）、BEV_FLUSH（尽可能多地刷新数据）、BEV_FINISH（尽可能多地刷新数据，可能包括流结束标记）。
    //   @ctx:   用户提供的指针。
    //
    // 返回值
    //     BEV_OK         表示写入了数据；
    //     BEV_NEED_MORE  表示在获取更多输入之前无法生成更多输出；
    //     BEV_ERROR      表示发生错误。
    using bufferevnt_filter = bufferevent_filter_result (*)(evbuffer* src, evbuffer* dst,
                                                            ev_ssize_t dst_limit,
                                                            bufferevent_flush_mode mode, void* ctx);

    using connect_filter       = bool (*)(event_context* ctx);

    using read_filter          = bufferevnt_filter;

    using write_filter         = bufferevnt_filter;

    // process_filter:
    //     process_filter 在最后一个 read_filter 返回 BEV_OK 后开始调用, 它将(默认)在 Handler 线程执行, 不再对 socket 进行读写操作
    //
    //     传递给 process_filter 的函数签名遵循 unique_ptr<R> (*)(event_context*, unique_ptr<Arg>) 格式, 如:
    //     unique_ptr<json_object> parse_json(event_context* ctx, unique_ptr<string> json_str) {
    //         ...
    //     }
    //     其中第二个参数是上一个 process_filter 的结果, 返回值则是传递给下一个 process_filter 的参数
    //
    //     首个被调用的 process_filter 的第二个参数是 unique_ptr<raw_buffer>
    // -----------------------------------------------------------------------------------------------------------------
    // process_filter_proxy:
    //     要在 process_filter 的调用链中适配它们的参数, 通过 tcp_kit::func_traits<R, Arg> 保存 process_filter 的参数和返回值
    //     类型, 将他们封装成一个可调用函数(见 tcp_kit::filter::make<R, Arg>());
    //
    //     又因为需要在各个 process_filter 中检查他们的类型 是否与下一个 process_filter 的类型匹配, 所以在 @param3 和 @param4 中
    //     将 process_filter 的返回值与参数类型的类型名(std::type_info::name())进行交换.
    //
    //     参数:
    //         @ctx   : 事件上下文
    //         @arg : 传入的数据
    //         @in_t  : 传入数据的类型 std::type_info
    //         @out_t : 输出类型的类型 std::type_info
    using raw_ptr_deleter = void(*)(void*);
    using process_filter_proxy = unique_ptr<void, raw_ptr_deleter>(*)(const event_context* ctx,
                                                                      unique_ptr<void, raw_ptr_deleter> arg,
                                                                      const type_info* in_t,
                                                                      const type_info*& out_t);

    class filter {

    public:
        connect_filter       connect;
        read_filter          read;
        write_filter         write;
        process_filter_proxy process;

        template <typename R, typename Arg>
        static filter make(connect_filter connect_, read_filter read_, write_filter write_,
                           unique_ptr<R> (*process_)(event_context*, unique_ptr<Arg>));

        static filter make(connect_filter connect_, read_filter read_, write_filter write_);

        bool operator==(const filter&) const;

    private:
        filter() = default;
        filter(connect_filter connect_, read_filter read_, write_filter write_);

    };

    template <typename R, typename Arg>
    filter filter::make(connect_filter connect_, read_filter read_, write_filter write_,
                        unique_ptr<R> (*process_)(event_context *, unique_ptr<Arg>)) {
        filter f(connect_, read_, write_);
        using result_t = typename func_traits<decltype(process_)>::result_type;
        using arg_t    = typename func_traits<decltype(process_)>::args_type;
        f.process = [process_](event_context* ctx, unique_ptr<void> in, const type_info* in_t, const type_info*& out_t) {
            if(in_t->operator==(typeid(arg_t))) {
                unique_ptr<arg_t> arg(static_cast<arg_t*>(in.release()));
                unique_ptr<result_t> res = process_(ctx, arg); // TODO: 错误处理
                out_t = &typeid(result_t);
                auto deleter = [](void* ptr) { delete static_cast<result_t*>(ptr); };
                return unique_ptr<void, decltype(deleter)>(res.release(), deleter);
            } else {
                log_warn("Processor parameter type mismatch: Expected: [%s] Actual: [%s]", typeid(arg_t).name(), in_t->name());
                throw invalid_argument("Processor parameter type mismatch");
            }
        };

    }

}

#endif