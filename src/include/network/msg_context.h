#pragma once
#include <event2/event.h>
#include <event2/buffer.h>

namespace tcp_kit {

    // handler 线程不允许直接对 bufferevent 访问, 这将引发线程安全问题, 将输入输出的缓冲数据作为线程独享, 并通过事件回调
    // 通知 event handler 线程可以避免处理线程安全问题(要求事件本身设置为线程安全的)
    struct msg_context {
        evbuffer*   in;         // 输入缓冲区, 一般是一个完整的消息
        evbuffer*   out;        // 输出缓冲区, 缓存回写给客户端的数据
        event*      done_ev;    // 处理结束回调
        bool        error;      // 错误标志
        const char* error_msg;  // 错误信息

        void done();

    };

}