#include <logger/logger.h>
#include <network/server.h>

namespace tcp_kit {

    namespace server_test {

        uint32_t n(uint16_t n_of_processor, uint8_t n_acceptor, uint16_t n_ev_handler, uint16_t n_handler) {
            uint32_t _ctl = 0;
            if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
                throw invalid_argument("Illegal Parameter");
            if(!n_ev_handler && !n_handler && n_of_processor != 1) {
                uint16_t expect = (uint16_t) ((n_of_processor * EV_HANDLER_EXCEPT_SCALE) + 0.5);
                n_ev_handler = expect << 1;
                n_handler = n_of_processor - 1 - expect;
            } else if(n_ev_handler | n_handler) {
                if(!n_ev_handler) n_ev_handler = 1;
                if(!n_handler) n_handler = 1;
            }
            _ctl = _ctl | (n_ev_handler << EV_HANDLER_OFFSET);
            _ctl = _ctl | n_handler;
            return _ctl;
        }

        void t1() {
//            server basic_svr = basic_server(3000);
//            basic_svr.bind("sum", [](int a, int b){
//                return a + b;
//            });
//            basic_svr.start();
        }

        void t2() {
            auto ctl = n(105, 1, 0, 0);
            log_info("n_acceptor: %d | n_ev_handler: %d | n_handler: %d",
                     N_ACCEPTOR,
                     (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                     ctl & HANDLER_CAPACITY);
        }

    }

}