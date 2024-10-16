#ifndef TCP_KIT_SERVER_H
#define TCP_KIT_SERVER_H

#include <cstdint>
#include <thread/thread_pool.h>
#include <util/tcp_util.h>
#include <concurrent/blocking_fifo.hpp>
#include <concurrent/lock_free_fifo.hpp>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>

//#define N_ACCEPTOR 1

#define EV_HANDLER_CAPACITY      0x3fff
#define HANDLER_CAPACITY         0x3fff

#define EV_HANDLER_EXCEPT_SCALE  0.28f

#define EV_HANDLER_OFFSET        14
#define STATE_OFFSET             28

#ifndef N_FIFO_SIZE_OF_TASK
#define N_FIFO_SIZE_OF_TASK      3
#endif

namespace tcp_kit {

    void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
    void read_callback(bufferevent *bev, void *arg);
    void write_callback(bufferevent *bev, void *arg);
    void event_callback(bufferevent *bev, short what, void *arg);

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

    // NEW        | server 的初始状态
    // READY      | 所有工作线程准备就绪, 随时进入 RUNNING 状态
    // RUNNING    | server 接收/处理连接
    // STOPPING   | server 因运行时异常或中断或手动关闭, 此时不再接收连接请求, 所有连接关闭后进入 SHUTDOWN 状态
    // SHUTDOWN   | 所有连接已断开, 待线程池终结、释放事件循环等资源、日志记录等工作完成后进入 TERMINATED 状态
    // TERMINATED | server 生命周期结束

    // 生命周期函数:
    //   when_ready(): 进入 READY 状态后回调, 随后进入 RUNNING 状态

    class server {

    public:
        const uint16_t port;

        explicit server(uint16_t port_ = 3000,
                        uint16_t n_ev_handler = 0,
                        uint16_t n_handler = 0);
        void start();

        virtual void when_ready();

        friend void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
        friend void read_callback(bufferevent *bev, void *arg);
        friend void write_callback(bufferevent *bev, void *arg);
        friend void event_callback(bufferevent *bev, short what, void *arg);

        server(const server&) = delete;
        server(server&&) = delete;
        server& operator=(const server&) = delete;

    private:
        class handler;

        class ev_handler {
        public:
            vector<handler*> handlers;

            friend void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
            friend void read_callback(bufferevent *bev, void *arg);
            friend void write_callback(bufferevent *bev, void *arg);
            friend void event_callback(bufferevent *bev, short what, void *arg);

            ev_handler();
            ~ev_handler();
            void bind_and_run(server* server_ptr);

        private:
            server*         _server;
            event_base*     _ev_base;
            mutex           _mutex;
            evconnlistener* _evc;

        };

        class handler {
        public:
            using msg = int;
            using b_fifo  = blocking_fifo<msg>;
            using lf_fifo = lock_free_fifo<msg>;

            handler() = default;
            ~handler();
            void bind_and_run(server* server_ptr);

            bool compete;
            atomic<void*> fifo;

        private:
            server* _server;

        };

//        static thread_local void* _this_thread;

        static constexpr uint32_t NEW        = 0;
        static constexpr uint32_t READY      = 1 << STATE_OFFSET;
        static constexpr uint32_t RUNNING    = 2 << STATE_OFFSET;
        static constexpr uint32_t STOPPING   = 3 << STATE_OFFSET;
        static constexpr uint32_t SHUTDOWN   = 4 << STATE_OFFSET;
        static constexpr uint32_t TERMINATED = 5 << STATE_OFFSET;

        // [4][14][14]: run_state | n_of_acceptor | n_of_ev_handler | n_of_handler
        atomic<uint32_t>          _ctl;
        mutex                     _mutex;
        condition_variable_any    _state;
        atomic<uint32_t>          _ready_threads;
        unique_ptr<thread_pool>   _threads;
        event_base*               _acceptor_ev_base;
        vector<ev_handler>        _ev_handlers;
        vector<handler>           _handlers;

        void try_ready();
        void trans_to(uint32_t rs);
        void wait_at_least(uint32_t rs);
        uint32_t handlers_map();
        uint32_t ctl_of(uint32_t rs, uint32_t hp);
        bool run_state_at_least(uint32_t rs);
        uint16_t n_of_ev_handler();
        uint16_t n_of_handler();
    };

}

#endif
