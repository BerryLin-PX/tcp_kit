#ifndef TCP_KIT_SERVER_H
#define TCP_KIT_SERVER_H

#include <stdint.h>
#include <thread/thread_pool.h>
#include <util/tcp_util.h>
#include <concurrent/blocking_queue.hpp>
#include <event2/event.h>
#include <event2/listener.h>

#define N_ACCEPTOR 1

#define EV_HANDLER_CAPACITY      0x3fff
#define HANDLER_CAPACITY         0x3fff

#define EV_HANDLER_EXCEPT_SCALE  0.3f

#define EV_HANDLER_OFFSET        14
#define STATE_OFFSET             28

namespace tcp_kit {

    void acceptor_callback(struct evconnlistener* listener, socket_t fd, struct sockaddr* address, int socklen, void* arg);

    // n_acceptor   -> 接收连接线程数(始终为 1, 多个线程不会提高效率且产生竞争与增加线程切换)
    // n_ev_handler -> 处理连接事件线程数(一般是读、写)
    // n_handler    -> 处理消息线程数
    //
    // 默认情况下(CPU 核心数 > 1):
    //      acceptor   线程数量为 1
    //      ev_handler 线程数量为目标值 * 2, 目标是占 CPU 核心数的 30% (四舍五入) // TODO 权衡最优占比
    //      handler    线程数量为 (CPU核心数 - 1 - ev_handler目标值), 且最少为 1
    //
    // 假设 CPU 核心数是 4, acceptor 线程数为 1, ev_handler 线程数为 1 * 2 = 2, handler 线程数为 4 - 1 - 1 = 2
    //
    // ** 当 CPU 核心数为 1 时, acceptor 线程数量为 1, ev_handler 与 handler 共用一个线程 **
    // 默认的线程配置综合运行效率与各类角色的业务逻辑 (ev_handler 属于 IO 密集型, handler 属于 CPU 密集型) 考虑, 尽量避免线程上下文切换

    class server {

    public:
        const uint16_t port;

        explicit server(uint16_t port_ = 3000,
                        uint16_t n_ev_handler = 0,
                        uint16_t n_handler = 0);
        void start();

        server(const server&) = delete;
        server(server&&) = delete;
        server& operator=(const server&) = delete;

    private:
        static const uint32_t NEW        = 0;
        static const uint32_t READY      = 1 << STATE_OFFSET;
        static const uint32_t RUNNING    = 2 << STATE_OFFSET;
        static const uint32_t STOPPING   = 3 << STATE_OFFSET;
        static const uint32_t SHUTDOWN   = 4 << STATE_OFFSET;
        static const uint32_t TERMINATED = 5 << STATE_OFFSET;

        atomic<uint32_t>         _ctl;     // [4][14][14]: run_state | n_of_acceptor | n_of_ev_handler | n_of_handler
        unique_ptr<thread_pool>  _threads;

        void acceptor();
        uint16_t count_of_ev_handler();
        uint16_t count_of_handler();
    };

}

#endif
