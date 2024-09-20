#include <stdint.h>
#include <network/server.h>
#include <util/system_util.h>

namespace tcp_kit {

    server::server(uint16_t port_, uint16_t n_ev_handler, uint16_t n_handler): port(port_) {
        if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
            throw invalid_argument("Illegal Parameter");
        uint16_t n_of_processor = (uint16_t) numb_of_processor();
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
    }

    void server::start() {
        uint16_t n_ev_handler = count_of_ev_handler();
        uint16_t n_handler = count_of_handler();
        uint32_t n_thread = N_ACCEPTOR + n_ev_handler + n_handler;
        _threads = make_unique<thread_pool>(n_thread,
                                            n_thread,
                                            0l,
                                            make_unique<blocking_queue<runnable>>(n_thread));
        _threads->execute(&server::acceptor, this);
    }

    void server::acceptor() {

    }

    inline uint16_t server::count_of_ev_handler() {
        return (_ctl.load() >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
    }

    inline uint16_t server::count_of_handler() {
        return _ctl.load() & HANDLER_CAPACITY;
    }

}


