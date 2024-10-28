#ifndef TCP_KIT_SERVER_H
#define TCP_KIT_SERVER_H

#include <cstdint>
#include <logger/logger.h>
#include <thread/thread_pool.h>
#include <util/tcp_util.h>
#include <util/system_util.h>
#include <concurrent/blocking_fifo.hpp>
#include <concurrent/lock_free_fifo.hpp>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include "filter.h"

#define EV_HANDLER_CAPACITY      0x3fff
#define HANDLER_CAPACITY         0x3fff

#define EV_HANDLER_EXCEPT_SCALE  0.28f

#define EV_HANDLER_OFFSET        14
#define STATE_OFFSET             28

#ifndef TASK_FIFO_SIZE
#define TASK_FIFO_SIZE      3
#endif

namespace tcp_kit {

    class ev_handler_base;
    class handler_base;

    // 该类制定了 server 的状态控制行为
    //
    // NEW        | server 的初始状态
    // READY      | 所有工作线程准备就绪, 随时进入 RUNNING 状态
    // RUNNING    | server 接收/处理连接
    // STOPPING   | server 因运行时异常或中断或手动关闭, 此时不再接收连接请求, 所有连接关闭后进入 SHUTDOWN 状态
    // SHUTDOWN   | 所有连接已断开, 待线程池终结、释放事件循环等资源、日志记录等工作完成后进入 TERMINATED 状态
    // TERMINATED | server 生命周期结束

    class server_base {

    public:
        server_base();
        virtual ~server_base() = default;

    protected:
        friend ev_handler_base;
        friend handler_base;

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

        virtual void try_ready() = 0;
        void trans_to(uint32_t rs);
        void wait_at_least(uint32_t rs);
        uint32_t handlers_map();
        uint32_t ctl_of(uint32_t rs, uint32_t hp);
        bool run_state_at_least(uint32_t rs);

    };

    // ev_handler_base 与 handler_base 作为 Event Handler 与 Handler 的基本实现
    // 由于它们始终被 server 调度, 所以它们在 bind_and_run() 中与 server 进行状态同步
    // !!! 请确保在派生类中的构造函数是无参的 !!!
    // ----------------------------------------------------------------------
    // init() 在线程创建之初被调度, 此时 server 状态为 NEW, 派生类应该在此处进行初始化
    // run()  在 server 进入 RUNNING 状态后被调度, 派生类应该在此处做任务处理

    class ev_handler_base {

    public:
        vector<handler_base*> handlers;

        void bind_and_run(server_base* server_ptr);
        virtual void init(server_base* server_ptr) = 0;
        virtual void run() = 0;

        // 实现 builtin 可在 socket 的各个生命周期介入
        // 当任何事件发生, 过滤器依次进入
        void append_filter(const filter& filter_);
        void add_filter_before(const filter& what, const filter& filter_);
        void add_filter_after(const filter& what, const filter& filter_);
        void check_filter_duplicate(const filter& filter_);
        void replace_filters(vector<filter>&& filters);

        virtual ~ev_handler_base() = default;

    protected:
        server_base*    _server_base;
        vector<filter>  _filters;

        bool invoke_conn_filters(event_context& ctx);
        void register_read_write_filters(event_context& ctx);

    };

    class handler_base {

    public:
        void bind_and_run(server_base* server_ptr);
        virtual void init(server_base* server_ptr) = 0;
        virtual void run() = 0;
        virtual ~handler_base();

        bool compete;

    protected:
        using msg = int;
        using b_fifo  = blocking_fifo<msg>;
        using lf_fifo = lock_free_fifo<msg>;

        server_base* _server;
        atomic<void*> fifo;

    };

    // 模版参数 P 代表协议(Protocol), 因此可以将 server 指定为任意协议实现, 如: server<generic> svr;
    // P 必须满足以下条件:
    //   1: P 拥有子类 ev_handler 且派生于 ev_handler_base 类
    //   2: P 拥有子类 handler    且派生于 handler 类
    //
    // n_ev_handler -> 处理连接事件线程数(一般是读、写)
    // n_handler    -> 处理消息线程数
    //
    // 默认情况下(CPU 核心数 > 1):
    //      ev_handler_base 线程数量为目标值 * 2, 目标是占 CPU 核心数的 30% (四舍五入)
    //      handler_base    线程数量为 (CPU核心数 - 1 - ev_handler目标值), 且最少为 1
    //
    // 假设 CPU 核心数是 4, ev_handler_base 线程数为 1 * 2 = 2, handler_base 线程数为 4 - 1 = 3
    //
    // ** 当 CPU 核心数为 1 时, ev_handler_base 与 handler_base 共用一个线程 **
    // 默认的线程配置综合运行效率与各类角色的业务逻辑 (ev_handler_base 属于 IO 密集型, handler_base 属于 CPU 密集型) 考虑, 尽量避免线程上下文切换

    // 生命周期函数:
    //   when_ready(): 进入 READY 状态后回调, 随后进入 RUNNING 状态

    template <typename P>
    class server: public server_base {

    using ev_handler_t = typename P::ev_handler;
    using handler_t    = typename P::handler;

    static_assert(std::is_base_of<ev_handler_base, ev_handler_t>::value , "P::ev_handler must be derived from ev_handler_base.");
    static_assert(std::is_base_of<handler_base, handler_t>::value , "P::handler must be derived from handler_base.");

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
        friend ev_handler_t;
        friend handler_t;

        atomic<uint32_t>          _ready_threads;
        unique_ptr<thread_pool>   _threads;
        vector<ev_handler_t>      _ev_handlers;
        vector<handler_t>         _handlers;

        void try_ready() override;
        virtual void when_ready();

        uint16_t n_of_ev_handler();
        uint16_t n_of_handler();

    };

    template <typename P>
    server<P>::server(uint16_t port_, uint16_t n_ev_handler, uint16_t n_handler): port(port_) {
        if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
            throw invalid_argument("Illegal Parameter.");
        _ctl |= NEW;
        uint16_t n_of_processor = (uint16_t) numb_of_processor();
        if(n_of_processor != 1) {
            uint16_t expect = uint16_t((n_of_processor * EV_HANDLER_EXCEPT_SCALE) + 0.5);
            if(!n_ev_handler) n_ev_handler = expect << 1;
            if(!n_handler) n_handler = n_of_processor - 1 - expect;
        } else if(n_ev_handler || n_handler) {
            if(!n_ev_handler) n_ev_handler = 1;
            if(!n_handler) n_handler = 1;
        }
        _ctl |= (n_ev_handler << EV_HANDLER_OFFSET);
        _ctl |= n_handler;
    }

    // 4 - 2 ev_h: [1][2][1,2][1,2]
    // 3 - 5 ev_h: [1,4][2,4][3,5]
    // 30-37
    template <typename P>
    void server<P>::start() {
        uint16_t n_ev_handler = n_of_ev_handler();
        uint16_t n_handler = n_of_handler();
        uint32_t n_thread = n_ev_handler + n_handler;
        //log_debug("The server will start %d event handler_base thread(s) and %d handler_base thread(s)", n_ev_handler, n_handler);
        _threads = make_unique<thread_pool>(n_thread,n_thread,0l, make_unique<blocking_fifo<runnable>>(n_thread));
        if(n_ev_handler) {
            _ev_handlers = vector<ev_handler_t>(n_ev_handler);
            _handlers = vector<handler_t>(n_handler);
            for(uint16_t i = 0; i < max(n_ev_handler, n_handler); ++i) {
                if(i < min(n_ev_handler, n_handler)) {
                    _ev_handlers[i].handlers.push_back(&_handlers[i]);
                    _handlers[i].compete = n_ev_handler > n_handler;
                    _threads->execute(&ev_handler_t::bind_and_run, &_ev_handlers[i], this);
                    _threads->execute(&handler_t::bind_and_run, &_handlers[i], this);
                    //log_debug("n(th) of handler_base: %d | fifo: %s", i + 1, n_ev_handler > n_handler ? "blocking" : "lock free");
                } else if (i < n_ev_handler) { // ev_handler_base 线程多于 handler_base 时
                    for(uint16_t j = 0; j < n_ev_handler; ++j) {
                        _ev_handlers[i].handlers.push_back(&_handlers[j]);
                    }
                    _threads->execute(&ev_handler_t::bind_and_run, &_ev_handlers[i], this);
                } else { // ev_handler_base 线程少于 handler_base 时
                    uint16_t n_share = uint16_t(float(n_ev_handler) / float(n_handler - n_ev_handler) + 0.5);
                    uint16_t start = n_share * (i - n_ev_handler);
                    uint16_t end = (i + 1 == n_handler) ? n_ev_handler : start + n_share;
                    for(uint16_t j = start; j < end; j++) {
                        _ev_handlers[j].handlers.push_back(&_handlers[i]);
                    }
                    _handlers[i].compete = end - start > 1;
                    _threads->execute(&handler_t::bind_and_run, &_handlers[i], this);
                    //log_debug("N(th) of handler_base: %d | fifo: %s", i + 1, end - start > 1 ? "blocking" : "lock free");
                }
            }
            wait_at_least(READY);
            when_ready();
            trans_to(RUNNING);
            log_info("The server is started on port: %d", port);
            wait_at_least(SHUTDOWN);
        }
    }

    template <typename P>
    void server<P>::try_ready() {
        if(++_ready_threads == n_of_ev_handler() + n_of_handler()) {
            trans_to(READY);
        }
    }

    template <typename P>
    void server<P>::when_ready() { }

    template <typename P>
    inline uint16_t server<P>::n_of_ev_handler() {
        return (_ctl.load(memory_order_relaxed) >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
    }

    template <typename P>
    inline uint16_t server<P>::n_of_handler() {
        return _ctl.load(memory_order_relaxed) & HANDLER_CAPACITY;
    }

}

#endif
