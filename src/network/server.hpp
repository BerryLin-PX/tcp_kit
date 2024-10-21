#ifndef TCP_KIT_SERVER_HPP
#define TCP_KIT_SERVER_HPP

#include <cstdint>
#include <thread/thread_pool.h>
#include <util/tcp_util.h>
#include <concurrent/blocking_fifo.hpp>
#include <concurrent/lock_free_fifo.hpp>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>

#define EV_HANDLER_CAPACITY      0x3fff
#define HANDLER_CAPACITY         0x3fff

#define EV_HANDLER_EXCEPT_SCALE  0.28f

#define EV_HANDLER_OFFSET        14
#define STATE_OFFSET             28

#ifndef TASK_FIFO_SIZE
#define TASK_FIFO_SIZE      3
#endif

namespace tcp_kit {

    template<typename P>
    class server;

    template<typename P>
    class handler_base;

    template<typename P>
    class ev_handler_base {
    public:
        vector<handler_base<P>*> handlers;
        void bind_and_run(server<P>* server_ptr);
        virtual void bind() = 0;
        virtual void run() = 0;

    protected:
        server<P>*      _server;
        event_base*     _ev_base;
        mutex           _mutex;
        evconnlistener* _evc;

    };

    template<typename P>
    class handler_base {
    public:
        using msg = int;
        using b_fifo  = blocking_fifo<msg>;
        using lf_fifo = lock_free_fifo<msg>;

        void bind_and_run(server<P>* server_ptr);
        virtual void init() = 0;
        virtual void run() = 0;

        bool compete;
        atomic<void*> fifo;

    protected:
        server<P>* _server;

    };

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

    // NEW        | server 的初始状态
    // READY      | 所有工作线程准备就绪, 随时进入 RUNNING 状态
    // RUNNING    | server 接收/处理连接
    // STOPPING   | server 因运行时异常或中断或手动关闭, 此时不再接收连接请求, 所有连接关闭后进入 SHUTDOWN 状态
    // SHUTDOWN   | 所有连接已断开, 待线程池终结、释放事件循环等资源、日志记录等工作完成后进入 TERMINATED 状态
    // TERMINATED | server 生命周期结束

    // 生命周期函数:
    //   when_ready(): 进入 READY 状态后回调, 随后进入 RUNNING 状态

    template<typename P>
    class server {

//    static_assert(std::is_base_of<ev_handler_base<P>, typename P::ev_handler>::value_type , "");

    public:
        const uint16_t port;

        explicit server(uint16_t port_ = 3000,
                        uint16_t n_ev_handler = 0,
                        uint16_t n_handler = 0);
        void start();

        virtual void when_ready();

        friend class P::ev_handler;
        friend class P::handler;

        server(const server&) = delete;
        server(server&&) = delete;
        server& operator=(const server&) = delete;

    private:
        static constexpr uint32_t NEW        = 0;
        static constexpr uint32_t READY      = 1 << STATE_OFFSET;
        static constexpr uint32_t RUNNING    = 2 << STATE_OFFSET;
        static constexpr uint32_t STOPPING   = 3 << STATE_OFFSET;
        static constexpr uint32_t SHUTDOWN   = 4 << STATE_OFFSET;
        static constexpr uint32_t TERMINATED = 5 << STATE_OFFSET;

        // [4][14][14]: run_state | n_of_acceptor | n_of_ev_handler | n_of_handler
        atomic<uint32_t>           _ctl;
        mutex                      _mutex;
        condition_variable_any     _state;
        atomic<uint32_t>           _ready_threads;
        unique_ptr<thread_pool>    _threads;
        event_base*                _acceptor_ev_base;
        vector<ev_handler_base<P>> _ev_handlers;
        vector<handler_base<P>>    _handlers;

        void try_ready();
        void trans_to(uint32_t rs);
        void wait_at_least(uint32_t rs);
        uint32_t handlers_map();
        uint32_t ctl_of(uint32_t rs, uint32_t hp);
        bool run_state_at_least(uint32_t rs);
        uint16_t n_of_ev_handler();
        uint16_t n_of_handler();

    };

    template<typename P>
    server<P>::server(uint16_t port_, uint16_t n_ev_handler, uint16_t n_handler): port(port_) {
        if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
            throw invalid_argument("Illegal Parameter");
        _ctl |= NEW;
        uint16_t n_of_processor = (uint16_t) numb_of_processor();
        if(n_of_processor != 1) {
            uint16_t expect = (uint16_t) ((n_of_processor * EV_HANDLER_EXCEPT_SCALE) + 0.5);
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
    template<typename P>
    void server<P>::start() {
        uint16_t n_ev_handler = n_of_ev_handler();
        uint16_t n_handler = n_of_handler();
        uint32_t n_thread = n_ev_handler + n_handler;
        log_debug("The server will start %d event handler_base thread(s) and %d handler_base thread(s)", n_ev_handler, n_handler);
        _threads = make_unique<thread_pool>(n_thread,n_thread,0l,
                                            make_unique<blocking_fifo<runnable>>(n_thread));
        if(n_ev_handler) {
            _handlers = vector<handler_base<P>>(n_handler);
            _ev_handlers = vector<ev_handler_base<P>>(n_ev_handler);
            for(uint16_t i = 0; i < max(n_ev_handler, n_handler); ++i) {
                if(i < min(n_ev_handler, n_handler)) {
                    _ev_handlers[i].handlers.push_back(&_handlers[i]);
                    _handlers[i].compete = n_ev_handler > n_handler;
                    _threads->execute(&P::ev_handler::bind_and_run, &_ev_handlers[i], this);
                    _threads->execute(&P::handler::bind_and_run, &_handlers[i], this);
                    log_debug("n(th) of handler_base: %d | fifo: %s", i + 1, n_ev_handler > n_handler ? "blocking" : "lock free");
                } else if (i < n_ev_handler) { // ev_handler_base 线程多于 handler_base 时
                    for(uint16_t j = 0; j < n_ev_handler; ++j) {
                        _ev_handlers[i].handlers.push_back(&_handlers[j]);
                    }
                    _threads->execute(&P::ev_handler::bind_and_run, &_ev_handlers[i], this);
                } else { // ev_handler_base 线程少于 handler_base 时
                    uint16_t n_share = uint16_t(float(n_ev_handler) / float(n_handler - n_ev_handler) + 0.5);
                    uint16_t start = n_share * (i - n_ev_handler);
                    uint16_t end = (i + 1 == n_handler) ? n_ev_handler : start + n_share;
                    for(uint16_t j = start; j < end; j++) {
                        _ev_handlers[j].handlers.push_back(&_handlers[i]);
                    }
                    _handlers[i].compete = end - start > 1;
                    _threads->execute(&P::handler::bind_and_run, &_handlers[i], this);
                    log_debug("N(th) of handler_base: %d | fifo: %s", i + 1, end - start > 1 ? "blocking" : "lock free");
                }
            }
            wait_at_least(READY);
            when_ready();
            trans_to(RUNNING);
            log_info("The server is started on port: %d", port);
            wait_at_least(SHUTDOWN);
        }
    }

    template<typename P>
    void server<P>::when_ready() { }

    // Event Handler
    template<typename P>
    void ev_handler_base<P>::bind_and_run(server<P>* server_ptr) {
        assert(server_ptr);
        _server = server_ptr;
        bind(server_ptr);
        _server->try_ready();
        _server->wait_at_least(server<P>::RUNNING);
        log_debug("Event handler_base running...");
        run();
    }

    // Handler
    template<typename P>
    void handler_base<P>::bind_and_run(server<P>* server_ptr) {
        assert(server_ptr);
        _server = server_ptr;
        init(server_ptr);
        _server->try_ready();
        _server->wait_at_least(server<P>::RUNNING);
        log_debug("Handler running...");
        run();
    }

    template<typename P>
    void server<P>::try_ready() {
        if(++_ready_threads == n_of_ev_handler() + n_of_handler()) {
            unique_lock<mutex> lock(_mutex);
            _ctl.store(ctl_of(READY, handlers_map()), memory_order_release);
            _state.notify_all();
        }
    }

    template<typename P>
    void server<P>::trans_to(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        _ctl.store(ctl_of(rs, handlers_map()), memory_order_release);
        _state.notify_all();
    }

    template<typename P>
    void server<P>::wait_at_least(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        while(!run_state_at_least(rs)) {
            interruption_point();
            interruptible_wait(_state, lock);
            interruption_point();
        }
    }

    template<typename P>
    inline uint32_t server<P>::handlers_map() {
        return _ctl & ((1 << STATE_OFFSET) - 1);
    }

    template<typename P>
    inline uint32_t server<P>::ctl_of(uint32_t rs, uint32_t hp) {
        return rs | hp;
    }

    template<typename P>
    inline bool server<P>::run_state_at_least(uint32_t rs) {
        return _ctl.load(memory_order_acquire) >= rs;
    }

    template<typename P>
    inline uint16_t server<P>::n_of_ev_handler() {
        return (_ctl.load(memory_order_relaxed) >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
    }

    template<typename P>
    inline uint16_t server<P>::n_of_handler() {
        return _ctl.load(memory_order_relaxed) & HANDLER_CAPACITY;
    }

}

#endif
