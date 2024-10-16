#include <cstdint>
#include <network/server.h>
#include <util/system_util.h>
#include <logger/logger.h>

namespace tcp_kit {

    void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        log_debug("New connection");
        server::ev_handler *ev_handler = (server::ev_handler *)(arg);
        bufferevent *bev = bufferevent_socket_new(ev_handler->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        bufferevent_enable(bev, EV_READ | EV_WRITE);
        bufferevent_setcb(bev,
                          read_callback, write_callback, event_callback,
                          ev_handler);
    }

    void read_callback(bufferevent *bev, void *arg) {

    }

    void write_callback(bufferevent *bev, void *arg) {

    }

    void event_callback(bufferevent *bev, short what, void *arg) {

    }

//    thread_local void* server::_this_thread;

    server::server(uint16_t port_, uint16_t n_ev_handler, uint16_t n_handler): port(port_) {
        if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
            throw invalid_argument("Illegal Parameter");
        _ctl |= NEW;
        uint16_t n_of_processor = (uint16_t) numb_of_processor();
        if(n_of_processor != 1) {
            uint16_t expect = (uint16_t) ((n_of_processor * EV_HANDLER_EXCEPT_SCALE) + 0.5);
            if(!n_ev_handler) n_ev_handler = expect << 1;
            if(!n_handler) n_handler = n_of_processor - 1 - expect;
            if(!n_handler) n_handler = 1;
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
    void server::start() {
        uint16_t n_ev_handler = n_of_ev_handler();
        uint16_t n_handler = n_of_handler();
        uint32_t n_thread = n_ev_handler + n_handler;
        log_debug("The server will start %d event handler thread(s) and %d handler thread(s)", n_ev_handler, n_handler);
        _threads = make_unique<thread_pool>(n_thread,n_thread,0l,
                                            make_unique<blocking_fifo<runnable>>(n_thread));
        if(n_ev_handler) {
            _handlers = vector<handler>(n_handler);
            _ev_handlers = vector<ev_handler>(n_ev_handler);
            for(uint16_t i = 0; i < max(n_ev_handler, n_handler); ++i) {
                if(i < min(n_ev_handler, n_handler)) {
                    _ev_handlers[i].handlers.push_back(&_handlers[i]);
                    _handlers[i].compete = n_ev_handler > n_handler;
                    _threads->execute(&server::ev_handler::bind_and_run, &_ev_handlers[i], this);
                    _threads->execute(&server::handler::bind_and_run, &_handlers[i], this);
                    log_debug("n(th) of handler: %d | fifo: %s", i + 1, n_ev_handler > n_handler ? "blocking" : "lock free");
                } else if (i < n_ev_handler) { // ev_handler 线程多于 handler 时
                    for(uint16_t j = 0; j < n_ev_handler; ++j) {
                        _ev_handlers[i].handlers.push_back(&_handlers[j]);
                    }
                    _threads->execute(&server::ev_handler::bind_and_run, &_ev_handlers[i], this);
                } else { // ev_handler 线程少于 handler 时
                    uint16_t n_share = uint16_t(float(n_ev_handler) / float(n_handler - n_ev_handler) + 0.5);
                    uint16_t start = n_share * (i - n_ev_handler);
                    uint16_t end = (i + 1 == n_handler) ? n_ev_handler : start + n_share;
                    for(uint16_t j = start; j < end; j++) {
                        _ev_handlers[j].handlers.push_back(&_handlers[i]);
                    }
                    _handlers[i].compete = end - start > 1;
                    _threads->execute(&server::handler::bind_and_run, &_handlers[i], this);
                    log_debug("n(th) of handler: %d | fifo: %s", i + 1, end - start > 1 ? "blocking" : "lock free");
                }
            }
            wait_at_least(READY);
            when_ready();
            trans_to(RUNNING);
            log_info("The server is started on port: %d", port);
            wait_at_least(SHUTDOWN);
        }
    }

    void server::when_ready() { }

    // Event Handler

    server::ev_handler::ev_handler() {
        //server::_this_thread = this;
        _ev_base = event_base_new();
    }

    server::ev_handler::~ev_handler() {
        evconnlistener_free(_evc);
        event_base_free(_ev_base);
    }

    void server::ev_handler::bind_and_run(server* server_ptr) {
        assert(server_ptr);
        _server = server_ptr;
        sockaddr_in sin = socket_address(_server->port);
        _evc = evconnlistener_new_bind(
                                _ev_base, listener_callback, this,
                                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                                -1, (sockaddr*) &sin, sizeof(sin));
        _server->try_ready();
        _server->wait_at_least(RUNNING);
        log_debug("Event handler running...");
        event_base_dispatch(_ev_base);
    }

    // Handler

    server::handler::~handler() {
        if(compete) delete static_cast<b_fifo*>(fifo.load());
        else        delete static_cast<lf_fifo*>(fifo.load());
    }

    void server::handler::bind_and_run(server* server_ptr) {
        assert(server_ptr);
        _server = server_ptr;
        fifo.store(compete ? (void*) new b_fifo(N_FIFO_SIZE_OF_TASK)
                              : (void*) new lf_fifo(N_FIFO_SIZE_OF_TASK),
                  memory_order_relaxed);
        _server->try_ready();
        _server->wait_at_least(RUNNING);
        log_debug("Handler running...");
    }

    void server::try_ready() {
        if(++_ready_threads == n_of_ev_handler() + n_of_handler()) {
            unique_lock<mutex> lock(_mutex);
            _ctl.store(ctl_of(READY, handlers_map()), memory_order_release);
            _state.notify_all();
        }
    }

    void server::trans_to(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        _ctl.store(ctl_of(rs, handlers_map()), memory_order_release);
        _state.notify_all();
    }

    void server::wait_at_least(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        while(!run_state_at_least(rs)) {
            interruption_point();
            interruptible_wait(_state, lock);
            interruption_point();
        }
    }

    inline uint32_t server::handlers_map() {
        return _ctl & ((1 << STATE_OFFSET) - 1);
    }

    inline uint32_t server::ctl_of(uint32_t rs, uint32_t hp) {
        return rs | hp;
    }

    inline bool server::run_state_at_least(uint32_t rs) {
        return _ctl.load(memory_order_acquire) >= rs;
    }

    inline uint16_t server::n_of_ev_handler() {
        return (_ctl.load(memory_order_relaxed) >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
    }

    inline uint16_t server::n_of_handler() {
        return _ctl.load(memory_order_relaxed) & HANDLER_CAPACITY;
    }

}


