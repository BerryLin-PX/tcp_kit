#include <network/server.h>
#include <error/errors.h>

namespace tcp_kit {

    using namespace errors;

    server_base::server_base(): _ctl(0) { }

    void server_base::trans_to(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        _ctl.store(ctl_of(rs, handlers_map()), memory_order_release);
        _state.notify_all();
    }

    void server_base::wait_at_least(uint32_t rs) {
        unique_lock<mutex> lock(_mutex);
        while(!run_state_at_least(rs)) {
            interruption_point();
            interruptible_wait(_state, lock);
            interruption_point();
        }
    }

    inline uint32_t server_base::handlers_map() {
        return _ctl & ((1 << STATE_OFFSET) - 1);
    }

    inline uint32_t server_base::ctl_of(uint32_t rs, uint32_t hp) {
        return rs | hp;
    }

    inline bool server_base::run_state_at_least(uint32_t rs) {
        return _ctl.load(memory_order_acquire) >= rs;
    }

    void server_base::append_filter(const filter& filter_) {
        check_filter_duplicate(filter_);
        _filters.push_back(move(filter_));
    }

    void server_base::add_filter_before(const filter& what, const filter& filter_) {
        check_filter_duplicate(filter_);
        auto it = _filters.begin();
        for (; it != _filters.end(); ++it) {
            if (*it == what) {
                break;
            }
        }
        _filters.insert(it, filter_);
    }

    void server_base::add_filter_after(const filter& what, const filter& filter_) {
        check_filter_duplicate(filter_);
        auto it = _filters.begin();
        for (; it != _filters.end(); ++it) {
            if (*it == what) {
                ++it;
                break;
            }
        }
        _filters.insert(it, filter_);
    }

    void server_base::check_filter_duplicate(const filter& filter_) {
        for(const filter& f : _filters) {
            if(f == filter_) {
                throw invalid_argument("The builtin of this type is already present in the builtin chain.");
            }
        }
    }

    void server_base::replace_filters(vector<filter>&& filters) {
        _filters = vector<filter>();
        for(const filter& f : filters) {
            append_filter(f);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------

    void ev_handler_base::bind_and_run(server_base* server_ptr) {
        assert(server_ptr);
        _server_base = server_ptr;
        _filters = &_server_base->_filters;
        init(server_ptr);
        _server_base->try_ready();
        _server_base->wait_at_least(server_base::RUNNING);
        //log_debug("Event handler_base running...");
        run();
    }

    void ev_handler_base::call_conn_filters(event_context* ctx) {
        for(const filter& f : *_filters) {
            if(f.connect) {
                f.connect(ctx);
            }
        }
    }

    void ev_handler_base::register_read_write_filters(event_context* ctx) {
        for(auto it = _filters->begin(); it != _filters->end(); ++it) {
            if(it->read || it->write) {
                bufferevent* nested_bev = bufferevent_filter_new(ctx->bev, it->read,it->write,
                                                                 BEV_OPT_CLOSE_ON_FREE, nullptr, ctx);
                if(nested_bev) {
                    ctx->bev = nested_bev;
                } else {
                    throw generic_error<CONS_BEV_ERR>("Failed to register filter with index [%d]",distance(_filters->begin(), it));
                }
            }
        }
    }

    void ev_handler_base::call_process_filters(const event_context *ctx) {
        raw_ptr_deleter deleter = [](void* ptr) { delete static_cast<raw_buffer*>(ptr); };
        auto next_arg = unique_ptr<void, raw_ptr_deleter>(new raw_buffer(ctx->bev), deleter);
        for(auto it = _filters->begin(); it != _filters->end(); ++it) {
            const type_info* in_t = &typeid(raw_buffer);
            const type_info* out_t = nullptr;
            try {
                next_arg = it->process(ctx, move(next_arg), in_t, out_t);
                in_t = out_t;
            } catch (generic_error<PRCS_ARG_ERR>& err) {
                log_warn(err.what());
                continue;
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------

    void handler_base::bind_and_run(server_base* server_ptr) {
        assert(server_ptr);
        _server = server_ptr;
        _fifo.store(compete ? (void*) new b_fifo(TASK_FIFO_SIZE) : (void*) new lf_fifo(TASK_FIFO_SIZE),
                    memory_order_relaxed);
        init(server_ptr);
        _server->try_ready();
        _server->wait_at_least(server_base::RUNNING);
        //log_debug("Handler running...");
        run();
    }

    handler_base::~handler_base() {
        if(compete) delete static_cast<b_fifo*>(_fifo.load());
        else        delete static_cast<lf_fifo*>(_fifo.load());
    }

}