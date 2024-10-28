#ifndef TCP_KIT_FILTER_H
#define TCP_KIT_FILTER_H

#include <vector>
#include <event2/bufferevent.h>
#include "event_context.h"

namespace tcp_kit {

    using namespace std;

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
    //     BEV_NEED_MORE if we can't produce any more output until we get some input;
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
    //     BEV_OK 表示写入了数据；
    //     BEV_NEED_MORE 表示在获取更多输入之前无法生成更多输出；
    //     BEV_ERROR 表示发生错误。
    using bufferevnt_filter = bufferevent_filter_result (*)(evbuffer* src, evbuffer* dst,
                                                            ev_ssize_t dst_limit,
                                                            bufferevent_flush_mode mode, void* ctx);

    using connect_filter = bool (*)(event_context& ctx);
    using read_filter    = bufferevnt_filter;
    using write_filter   = bufferevnt_filter;


    struct filter {
        connect_filter connect;
        read_filter    read;
        write_filter   write;

        bool operator==(const filter&) const;
    };

}

#endif