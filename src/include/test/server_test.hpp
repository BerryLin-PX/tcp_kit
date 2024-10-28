#include <logger/logger.h>
#include <network/server.h>
#include <network/protocol/generic.h>

namespace tcp_kit {

    namespace server_test {

        uint32_t n_with_scale(uint16_t n_of_processor, float scale, uint16_t n_ev_handler = 0, uint16_t n_handler = 0) {
            uint32_t _ctl = 0;
            if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
                throw invalid_argument("Illegal Parameter");
            if(n_of_processor != 1) {
                uint16_t expect = (uint16_t) ((n_of_processor * scale) + 0.5);
                if(!n_ev_handler) n_ev_handler = expect << 1;
                if(!n_handler) n_handler = n_of_processor - expect;
//                if(!n_handler) n_handler = 1;
            } else if(n_ev_handler | n_handler) {
                if(!n_ev_handler) n_ev_handler = 1;
                if(!n_handler) n_handler = 1;
            }
            _ctl = _ctl | (n_ev_handler << EV_HANDLER_OFFSET);
            _ctl = _ctl | n_handler;
            return _ctl;
        }

        uint32_t n(uint16_t n_of_processor, uint16_t n_ev_handler = 0, uint16_t n_handler = 0) {
            return n_with_scale(n_of_processor, EV_HANDLER_EXCEPT_SCALE, n_ev_handler, n_handler);
        }

        void t1() {
//            server basic_svr = basic_server(3000);
//            basic_svr.init("sum", [](int a, int b){
//                return a + b;
//            });
//            basic_svr.start();
        }

        // 应输出 1-58-46
        void t2() {
            auto ctl = n(105);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        // 应输出 1-0-0
        void t3() {
            auto ctl = n(1, 0, 0);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        // 应输出 1-1-1
        void t4() {
            auto ctl = n(1, 1, 0);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        // 应输出 1-1-1
        void t5() {
            auto ctl = n(2, 1, 0);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        // 应输出 1-2-1
        void t6() {
            auto ctl = n(2, 0, 1);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        // 应输出 1-1-1
        void t7() {
            auto ctl = n(3, 1, 0);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     0,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

        void t8() {
            t2();
            t3();
            t4();
            t5();
            t6();
            t7();
        }

        // 当 ev_handler_base 线程数不等于 handler_base 线程数时, 就认为存在竞争, 所以应尽量让他们的数量相等
        // 1. ev_handler_base 线程数量多于 handler_base 数量时:
        //      假设 ev_handler_base: 4, handler_base:3 那么首先三个 ev_handler_base 线程各自被分配一个handler 线程,
        //      多出来的一个 ev_handler_base 线程将轮询将任务分配给这三个 handler_base 线程, 相当于在一个 handler_base 上
        //      的同一时刻可能存在两个线程竞争, 而另外两个线程不受到影响. 在 ev_handler_base 不过多大于 handler_base
        //      的前提下, 竞争是相当小的
        // 2. ev_handler_base 线程数量少于 handler_base 数量时:
        //      假设 ev_handler_base: 60, handler_base:69 同样首先为每个 ev_handler_base 线程各自被分配一个handler 线程,
        //      handler_base 比 ev_handler_base 多出来的 9 个线程将按照 60 / 9 ≈ 6.667 向上取整 => 7 个线程分配一个
        //      多出来的 handler_base 线程, 也就是 8 组 7 个 ev_handler_base 线程分配8个多出来的 handler_base 线程, 剩下
        //      4 个 ev_handler_base 线程分配第九个 handler_base 线程, 由于 ev_handler_base 的任务优先递交给自己独占的
        //      handler_base 线程, 当这个线程无法处理时, 才分配给那个额外的 handler_base 线程, 所以即便 7 个线程有可能
        //      会在这个额外的线程产生竞争, 达到最大竞争的可能性也是很小的
        // 在测试的 CPU 核心数 1-100 的数量里,
        void t9() {
            float total = 0;
            uint32_t times = 100;
            for(uint32_t i = 1; i <= times; ++i) {
                auto ctl = n_with_scale(i, 0.28f,0, 0);
                auto n_ev_handler = (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
                auto n_handler = ctl & HANDLER_CAPACITY;
                log_info("n_processor: %d | n_ev_handler: %d | n_handler: %d | gap: %d", i, n_ev_handler, n_handler, n_ev_handler - n_handler);
            }
        }

        template <typename Func, typename... Args>
        void run(Func&& a_func, Args&&... args) {
            auto runnable_func = std::bind(std::forward<Func>(a_func), std::forward<Args>(args)...);
            runnable_func();
        }

        void t10() {
            class a_class {
              void a_function() {
                  log_info("a_function");
              }
            public:
              void t3_inner() {
                  run(&a_class::a_function, this);
              }
            };
            a_class a;
            a.t3_inner();
        }

        void t11() {
            struct a_struct {
                int a = 1;
                int b = 2;
            };
            lock_free_fifo<a_struct> fifo(3);
            a_struct as;
            fifo.push_by_shallow_copy(&as);
            a_struct as_out;
            fifo.pop(&as_out);
            log_info("a: %d, b: %d", as_out.a, as_out.b);
        }

        void t12() {
            server<generic> svr;
            svr.start();
        }

    }

}