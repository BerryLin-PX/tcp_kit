#pragma once

#include <cstdlib>
#include <network/filter_chain.h>
#include <network/ev_context.h>
#include <network/msg_context.h>
#include <util/int_types.h>
#include <logger/logger.h>
#include <thread/thread_pool.h>
#include <util/tcp_util.h>
#include <util/system_util.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>
#include <concurrent/lock_free_queue.h>
#include <concurrent/lock_free_spsc_queue.h>

#define EV_HANDLER_CAPACITY      0x3fff
#define HANDLER_CAPACITY         0x3fff
#define RUN_STATE_CAPACITY       0xf

#define EV_HANDLER_EXCEPT_SCALE  0.3f

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
        server_base(filter_chain filters_);

        bool is_running();

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

        // [4][14][14]: run_state | n_of_ev_handler | n_of_handler
        std::atomic<uint32_t>       _ctl;
        std::mutex                  _mutex;
        std::condition_variable_any _state;
        filter_chain                _filters;

        virtual void try_ready() = 0;
        void trans_to(uint32_t rs);
        void wait_at_least(uint32_t rs);
        uint32_t handlers_map();
        uint32_t ctl_of(uint32_t rs, uint32_t hp);
        uint32_t run_state_of(uint32_t rs);
        bool run_state_at_least(uint32_t rs);

    };

    // ev_handler_base 与 handler_base 作为 Event Handler 与 Handler 的基本实现
    // 由于它们始终被 server 调度, 所以它们在 bind_and_run() 中与 server 进行状态同步
    // 请确保在派生类中的构造函数是无参的
    class ev_handler_base {

    public:
        size_t n_handler;
        std::vector<handler_base*> handlers;

        ev_handler_base();

        void bind_and_run(server_base* server_ptr);
        // 在线程创建之初被调度, 此时 server 状态为 NEW, 派生类应在此处进行初始化
        virtual void init(server_base* server_ptr) = 0;
        // 在 server 进入 RUNNING 状态后被调度, 派生类应在此处做任务处理
        virtual void run() = 0;

        virtual ~ev_handler_base() = default;

    protected:
        server_base*    _server_base;
        filter_chain*   _filters;

        void call_conn_filters(ev_context* ctx);
        void register_read_write_filters(ev_context* ctx);
        // std::unique_ptr<evbuffer_holder> call_process_filters(ev_context* ctx);

    };

    class handler_base {

    public:
        void bind_and_run(server_base* server_ptr);
        virtual void init(server_base* server_ptr) = 0;
        virtual void run() = 0;
        virtual ~handler_base();

        using msg = msg_context*;
        bool race;
        std::unique_ptr<queue<msg>> msg_queue;

    protected:
        filter_chain* _filters;
        server_base*  _server_base;
        std::unique_ptr<evbuffer_holder> call_process_filters(ev_context* ctx);

    };

    struct api_dispatcher_p {};

    // 模版参数 Protocols:
    // Protocols 代表协议(Protocol), 将 server 指定为任意协议实现, 如: server<generic> svr; 或: server<http> svr;
    // Protocols 必须满足以下条件:
    //   1: Protocols 有子类 ev_handler 且派生于 ev_handler_base 类, 无参构造函数
    //   2: Protocols 有子类 handler 且派生于 handler 类, 无参构造函数
    //   3: Protocols 如若有默认的过滤器集, 声明 filters 类型, 如: using filters = type_list<filter1, filter2, filter3...>;
    //   4: Protocols 有子类 api_dispatcher<uint16_t PORT>, 无参构造函数, 且具备以下条件:
    //      ----------------------------------------------------------------------------------------------------------
    //      1. api 函数(static 修饰)
    //      该函数在 server 中注册一个 api, 以便对消息进行区分: svr.api("echo", [](string msg){ return msg; });
    //      参数
    //        @id:   消息处理器的唯一标识类型
    //        @prcs: 消息处理器, 可能是一个函数指针, 也可能是一个 lambda 表达式
    //      template<typename Identity, typename Processor>
    //      void api(Identity id, Processor prcs);
    //      ----------------------------------------------------------------------------------------------------------
    //      2. 过滤器
    //      如有需要注册过滤器, 在此类中实现 filter 的 process 函数
    //      使用 api_dispatcher_p 代替实际类型, 如: using filters = type_list<filter1, api_dispatcher_p>
    //
    // 线程的分配:
    //   ev_handler -> 处理连接事件线程(一般是读、写)
    //   handler    -> 处理消息线程
    //
    // 默认的分配规则:
    //   ev_handler 线程数: CPU 核心数 * 0.3
    //   handler    线程数: CPU 核心数 - ev_handler 线程数
    //   ev_handler 与 handler 线程数量必须满足倍数关系, 所以在计算完线程分配后将进行调整
    //
    // 默认的线程配置综合运行效率与各类角色的业务逻辑 (ev_handler_base 属于 IO 密集型, handler_base 属于 IO 密集型或 CPU 密集型)
    // 考虑, 尽量避免线程上下文切换
    //
    // 消息队列
    //   ev_handler 和 handler 使用队列传递消息. 根据它们线程数的不同, 自动选择使用是否支持在多线程同步的队列, 在分配线程时尽量不要使
    //   ev_handler 的线程数多于 handler 的线程数, 这将产生竞争

    // 生命周期函数:
    //   when_ready(): 进入 READY 状态后回调, 随后进入 RUNNING 状态

    template <typename Protocols, uint16_t PORT = 3000>
    class server: public server_base {

    protected:
        using ev_handler_t     = typename Protocols::template ev_handler<PORT>;
        using handler_t        = typename Protocols::handler;
        using api_dispatcher_t = typename Protocols::template api_dispatcher<PORT>;
        using filters          = typename replace_type<typename Protocols::filters, api_dispatcher_p, api_dispatcher_t>::type;

        static_assert(std::is_base_of<ev_handler_base, ev_handler_t>::value , "Protocols::ev_handler must be derived from ev_handler_base.");
        static_assert(std::is_base_of<handler_base, handler_t>::value , "Protocols::handler must be derived from handler_base.");

    public:
        explicit server(uint16_t n_ev_handler = 0,
                        uint16_t n_handler = 0);

        void start();

        template<typename Identity, typename Processor>
        void api(const Identity& id, Processor prcs);

        server(const server&) = delete;
        server(server&&) = delete;
        server& operator=(const server&) = delete;

    private:
        friend ev_handler_t;
        friend handler_t;

        std::atomic<uint32_t>          _ready_threads;
        std::unique_ptr<thread_pool>   _threads;
        std::vector<ev_handler_t>      _ev_handlers;
        std::vector<handler_t>         _handlers;

        void try_ready() override;
        virtual void when_ready();

        inline bool is_multiple(uint16_t a, uint16_t b);
        void adjust_to_multiple(uint16_t& a, uint16_t& b);
        uint16_t n_of_ev_handler();
        uint16_t n_of_handler();

    };

    template <typename Protocols, uint16_t PORT>
    server<Protocols,PORT>::server(uint16_t n_ev_handler, uint16_t n_handler) : server_base(filter_chain::make(filters{})) {
        if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
            throw std::invalid_argument("Illegal Parameter.");
        _ctl |= NEW;
        uint16_t n_of_processor = (uint16_t) numb_of_processor();
        if(n_of_processor != 1) {
            uint16_t expect = uint16_t((n_of_processor * EV_HANDLER_EXCEPT_SCALE) + 0.5);
            if(!n_ev_handler) n_ev_handler = expect << 1;
            if(!n_handler) n_handler = n_of_processor - expect;
        } else {
            if(!n_ev_handler) n_ev_handler = 1;
            if(!n_handler) n_handler = 1;
        }
        adjust_to_multiple(n_ev_handler, n_handler);
        _ctl |= (n_ev_handler << EV_HANDLER_OFFSET);
        _ctl |= n_handler;
    }

    // 4 - 2 ev_h: [1][2][1,2][1,2]
    // 3 - 5 ev_h: [1,4][2,4][3,5]
    // 30-37
    template <typename Protocols, uint16_t PORT>
    void server<Protocols, PORT>::start() {
        //evthread_use_pthreads();
        uint16_t n_ev_handler = n_of_ev_handler();
        uint16_t n_handler = n_of_handler();
        uint32_t n_thread = n_ev_handler + n_handler;
        //log_debug("The server will start %d event handler_base thread(s) and %d handler_base thread(s)", n_ev_handler, n_handler);
        _threads = std::make_unique<thread_pool>(n_thread, n_thread, 0l, std::make_unique<blocking_fifo<runnable>>(n_thread));
        _ev_handlers = std::vector<ev_handler_t>(n_ev_handler);
        _handlers = std::vector<handler_t>(n_handler);
        if(n_ev_handler > n_handler) {
            uint16_t n_share = n_ev_handler / n_handler;
            for(uint16_t handler_i = 0; handler_i < n_handler; ++handler_i) {
                for(uint16_t j = 0; j < n_share; ++j) {
                    uint16_t ev_handler_i = handler_i * n_share + j;
                    _ev_handlers[ev_handler_i].handlers.push_back(&_handlers[handler_i]);
                    _ev_handlers[ev_handler_i].n_handler = 1;
                    _threads->execute(&ev_handler_t::bind_and_run, &_ev_handlers[ev_handler_i], this);
                }
                _handlers[handler_i].race = true;
                _threads->execute(&handler_t::bind_and_run, &_handlers[handler_i], this);
            }
        } else {
            uint16_t n_own = n_handler / n_ev_handler;
            for(uint16_t ev_handler_i = 0; ev_handler_i < n_ev_handler; ++ev_handler_i) {
                for(uint16_t j = 0; j < n_own; ++j) {
                    uint16_t handler_i = ev_handler_i * n_own + j;
                    _handlers[handler_i].race = false;
                    _ev_handlers[ev_handler_i].handlers.push_back(&_handlers[handler_i]);
                    _threads->execute(&handler_t::bind_and_run, &_handlers[handler_i], this);
                }
                _ev_handlers[ev_handler_i].n_handler = n_own;
                _threads->execute(&ev_handler_t::bind_and_run, &_ev_handlers[ev_handler_i], this);
            }
        }
        wait_at_least(READY);
        when_ready();
        trans_to(RUNNING);
        log_info("The server is started on port: %d", PORT);
        wait_at_least(SHUTDOWN);
    }

    template<typename Protocols, uint16_t PORT>
    template<typename Identity, typename Processor>
    void server<Protocols, PORT>::api(const Identity& id, Processor prcs) {
        api_dispatcher_t::api(id, prcs);
    }

    template <typename Protocols, uint16_t PORT>
    void server<Protocols, PORT>::try_ready() {
        if(++_ready_threads == n_of_ev_handler() + n_of_handler()) {
            trans_to(READY);
        }
    }

    template <typename Protocols, uint16_t PORT>
    void server<Protocols, PORT>::when_ready() { }

    template <typename Protocols, uint16_t PORT>
    bool server<Protocols, PORT>::is_multiple(uint16_t a, uint16_t b) {
        return a % b == 0 || b % a == 0;
    }

    // 以最小的调整次数, 将两整数 ab 调整为倍数关系.
    // 在同等调整次数的情况下, 一增一减优于同增同减. 一增一减的情况下, a 减 b 增优于 a 增 b 减; 同增同减的情况下, 同增优于同减
    // 输入
    //   @a: 前者 u16 引用
    //   @b: 后者 u16 引用
    template <typename Protocols, uint16_t PORT>
    void server<Protocols, PORT>::adjust_to_multiple(uint16_t& a, uint16_t& b) {
        if(!is_multiple(a, b)) {
            uint16_t max = std::abs(a - b);
            for(uint16_t limit = 1; limit <= max; ++limit) {
                for(uint16_t step_a = 0; step_a <= limit; ++step_a) {
                    uint16_t step_b = limit - step_a;
                    if(is_multiple(a - step_a, b + step_b)) {
                        a -= step_a; b += step_b; return;
                    } else if(is_multiple(a + step_a, b - step_b)) {
                        a += step_a; b -= step_b; return;
                    } else if(is_multiple(a + step_a, b + step_b)) {
                        a += step_a; b += step_b; return;
                    } else if(is_multiple(a - step_a, b - step_b)) {
                        a -= step_a; b -= step_b; return;
                    }
                }
            }
        }
    }

    template <typename Protocols, uint16_t PORT>
    uint16_t server<Protocols, PORT>::n_of_ev_handler() {
        return (_ctl.load(std::memory_order_relaxed) >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
    }

    template <typename Protocols, uint16_t PORT>
    inline uint16_t server<Protocols, PORT>::n_of_handler() {
        return _ctl.load(std::memory_order_relaxed) & HANDLER_CAPACITY;
    }

}