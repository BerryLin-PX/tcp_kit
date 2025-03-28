#pragma once
#include <event2/event.h>
#include <event2/buffer.h>

namespace tcp_kit {

    // handler 线程不允许直接对 bufferevent 访问, 这将引发线程安全问题, 将输入输出的缓冲数据作为线程独享, 并通过事件回调
    // 通知 event handler 线程可以避免处理线程安全问题(要求事件本身设置为线程安全的)
    struct msg_context {
        uint32_t    conn_id;        // tcp 连接唯一id
        char       *in;             // 输入缓冲区, 一般是一个完整的消息
        size_t      in_len;
        char       *out;            // 输出缓冲区, 缓存回写给客户端的数据
        size_t      out_len;
        bool        event_fired;    // 标志回调是否已触发
        event      *done_ev;        // 处理结束回调
        event      *error_ev;       // 处理结束回调
        bool        error_flag;     // 错误标志

        // -------------以下事件只能有一个被触发--------------------
        void done();
        void error();
    };

}