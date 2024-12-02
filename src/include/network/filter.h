#ifndef TCP_KIT_FILTER_H
#define TCP_KIT_FILTER_H

#include <vector>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <network/event_context.h>
#include <util/func_traits.h>
#include <logger/logger.h>
#include <error/errors.h>

namespace tcp_kit {

    using namespace std;

    // SFINAE

    class raw_buffer {

    public:
        evbuffer* buffer;
        raw_buffer(evbuffer* buffer_): buffer(buffer_) { };

    };

    // 在对 process_filter 中参数与返回值进行数据转换时需要擦除指针类型, 删除器确保 unique_ptr 在类型被擦出之后仍可以正确释放内存
    using raw_ptr_deleter = void(*)(void*);

    // 同 libevent 中 bufferevent 声明的回调函数 bufferevent_filter_cb:
    // 需要注意的一点是:
    //   最后一个参数 ctx 在 bufferevent_filter_cb 中定义为 void*, 而此处为 event_context* 这是因为在 catchable_bufferevent_filter 中进行了类型转换
    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // A callback function to implement a builtin for a bufferevent.
    //
    // Parameters
    //   @src:   An evbuffer to drain data from.
    //   @dst:   An evbuffer to add data to.
    //   @limit: A suggested upper bound of bytes to write to dst. The builtin may ignore this template_params_valid, but doing so means that it will overflow the high-water mark associated with dst. -1 means "no limit".
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
                                                            bufferevent_flush_mode mode, event_context* ctx);

    // 连接正式建立前回调
    // 参数
    //   @ctx: 事件上下文
    using connect_filter       = void (*)(event_context* ctx);

    // bufferevent 的读过滤器
    using read_filter          = bufferevnt_filter;

    // bufferevent 的写过滤器
    using write_filter         = bufferevnt_filter;

    // process_filter 在最后一个 read_filter 返回 BEV_OK 后开始调用, 它将(默认)在 Handler 线程执行, 不再对 socket 进行读写操作
    //
    // 传递给 process_filter 的函数签名遵循 unique_ptr<R> (*)(event_context*, unique_ptr<Arg>) 格式, 如:
    // unique_ptr<json_object> parse_json(event_context* ctx, unique_ptr<string> json_str) {
    //     ...
    // }
    // 其中第二个参数是上一个 process_filter 的结果, 返回值则是传递给下一个 process_filter 的参数
    //
    // 首个被调用的 process_filter 的第二个参数与最后一个被调用的 process_filter 的返回值必须是 unique_ptr<raw_buffer>
    template<typename R, typename Arg, typename RD = default_delete<R>, typename ArgD = default_delete<Arg>>
    using process_filter = std::unique_ptr<R, RD> (*)(event_context*, std::unique_ptr<Arg, ArgD>);

    // 要在 process_filter 的调用链中适配它们的参数, 通过 tcp_kit::func_traits<R, Arg> 保存 process_filter 的参数和返回值
    // 类型, 将他们封装成一个可调用函数(见 tcp_kit::filter::make<R, Arg>());
    //
    // 又因为需要在各个 process_filter 中检查他们的类型 是否与下一个 process_filter 的类型匹配, 所以在 @param3 和 @param4 中
    // 将 process_filter 的返回值与参数类型的类型名(std::type_info::name())进行交换.
    //
    //  参数:
    //    @ctx   : 事件上下文
    //    @arg   : 传入的数据
    //    @in_t  : 传入数据的类型 std::type_info
    //    @out_t : 输出类型的类型 std::type_info
    using process_filter_proxy = function<unique_ptr<void, raw_ptr_deleter>(const event_context* ctx,
                                                                      unique_ptr<void, raw_ptr_deleter> arg,
                                                                      const type_info* in_t,
                                                                      const type_info*& out_t)>;

    // 所有 filter 中的 connect 事件由该函数代理, 它对过滤器进行错误处理
    template<connect_filter F>
    void catchable_connect_filter(event_context* ctx) {
        try {
            F(ctx);
        } catch (...) {
            // TODO
            throw;
        }
    }

    // 所有 filter 中的 read/write 事件由该函数代理, 它对过滤器进行错误处理, 并且将 void* 转换为 event_context* 类型
    // TODO: 此回调函数具有 event_context 的所有权, 错误发生时释放 event_context
    template<bufferevnt_filter F>
    bufferevent_filter_result catchable_bufferevent_filter(evbuffer* src, evbuffer* dst,
                                                           ev_ssize_t dst_limit,
                                                           bufferevent_flush_mode mode, void* ctx) {
        try {
            return F(src, dst, dst_limit, mode, static_cast<event_context*>(ctx));
        } catch (...) {
            throw;
        }
    }

    template<typename R, typename... Args>
    unique_ptr<void, raw_ptr_deleter> catchable_process_filter(const event_context* ctx,
                                                               unique_ptr<void, raw_ptr_deleter> in,
                                                               const type_info* in_t,
                                                               const type_info*& out_t) {
    }

    // tcp_kit::filter 类可以介入 TCP 连接的整个周期
    class filter {

    public:
        connect_filter         connect;
        bufferevent_filter_cb  read;
        bufferevent_filter_cb  write;
        process_filter_proxy   process;

        // 构造一个 filter, 并将各回调函数传递给 catchable_ 开头的系列函数处理
        // 参数
        //   @template<CONN_F>:  TCP 建立连接前的回调函数, 见 connect_filter
        //   @template<READ_F>:  bufferevent 读回调函数, 见 read_filter
        //   @template<WRITE_F>: bufferevent 写回调函数, 见 write_filter
        // 返回值
        //   构造完成的 filter
         template<connect_filter CONN_F = nullptr, bufferevent_filter_cb READ_F = nullptr, bufferevent_filter_cb WRITE_F = nullptr>
         static filter make();

        // 构造一个 filter, 并将各回调函数传递给 catchable_ 开头的系列函数处理. 其中 process_ 参数与其他回调函数不同, 作为形参传入, 这是因为
        // process 回调需要对参数与返回值进行转换, 普通的函数指针做不到类型的动态转换
        //
        // 为什么不统一所有回调函数的构造方式?
        //   std::function 调用是存在额外开销的, 除非迫不得已, 才采用 std::function 的方式, 这就造成了构造 filter 时存在的两种不同范式
        //   一种是将回调函数的函数名作为模版特化: tcp_kit:filter::make<any_cb>(); 另一种将函数指针作为实参: tcp_kit:filter::make(any_cb);
        //   所以以下的构造方式都是合法的:
        //     1. tcp_kit::filter f = tcp_kit::filter::make<connect_cb>();
        //     2. tcp_kit::filter f = tcp_kit::filter::make<nullptr, read_cb, write_cb>();
        //     3. tcp_kit::filter f = tcp_kit::filter::make(process_cb);
        //     4. tcp_kit::filter f = tcp_kit::filter::make<connect_cb>(process_cb);
        //     5. tcp_kit::filter f = tcp_kit::filter::make<connect_cb, read_cb, write_cb>(process_cb);
        //
        // 参数
        //   @template<CONNECT_>: TCP 建立连接前的回调函数
        //   @template<READ_>:    bufferevent 读回调函数
        //   @template<WRITE_>:   bufferevent 写回调函数
        //   @process_:           一次完整报文读取完成时的回调
        // 返回值
        //   构造完成的 filter
        template <connect_filter CONNECT_ = nullptr,
                  read_filter READ_ = nullptr,
                  write_filter WRITE_ = nullptr,
                  typename ProcessFilter>
        static filter make(ProcessFilter p);

        bool operator==(const filter&) const;

    private:
        filter() = default;
        filter(connect_filter connect_, bufferevent_filter_cb read_,
               bufferevent_filter_cb write_): connect(connect_),
                                              read(read_),
                                              write(write_) { };

    };

    template<connect_filter CONN_F, bufferevent_filter_cb READ_F, bufferevent_filter_cb WRITE_F>
    filter filter::make() {
        return filter(CONN_F, READ_F, WRITE_F);
    }

    template<connect_filter CONNECT_,
            read_filter READ_,
            write_filter WRITE_,
            typename ProcessFilter>
    filter filter::make(ProcessFilter p) {
        filter f = filter::make<CONNECT_ ? catchable_connect_filter<CONNECT_> : nullptr,
                                READ_ ? catchable_bufferevent_filter<READ_> : nullptr,
                                WRITE_ ? catchable_bufferevent_filter<WRITE_> : nullptr>();
        f.process = [p](const event_context* ctx, unique_ptr<void, raw_ptr_deleter> in,
                        const type_info* in_t, const type_info*& out_t) {
            using result_t = typename func_traits<ProcessFilter>::result_type;
            using args_t = typename func_traits<ProcessFilter>::args_type;
            using arg_inner_t = typename tuple_element_t<1, args_t>::element_type;
            using arg_inner_d_t = typename tuple_element_t<1, args_t>::deleter_type;
            using result_inner_t = typename result_t::element_type;
            using result_inner_d_t = typename result_t::deleter_type;
            if(in_t->operator==(typeid(arg_inner_t))) {
                unique_ptr<arg_inner_t, arg_inner_d_t> arg(static_cast<arg_inner_t*>(in.release())); // TODO: 删除器
                unique_ptr<result_inner_t> res = p(ctx, move(arg)); // TODO: 错误处理
                out_t = &typeid(result_inner_t);
                return unique_ptr<void, raw_ptr_deleter>(res.release(), [](void* ptr) { delete static_cast<result_inner_t*>(ptr); }); // TODO: 如果指针本身带有删除器, 保留原有的
            } else {
                throw generic_error<PRCS_ARG_MISMATCHED>("Processor parameter type mismatch: Expected: [%s] Actual: [%s]", typeid(result_inner_t).name(), in_t->name());
            }
        };
        return f;
    }


}

#endif