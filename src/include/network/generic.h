#pragma once

#include <network/server.h>
#include <include/logger/logger.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <include/error/errors.h>
#include <unordered_map>
#include <network/generic_msg.pb.h>
#include <network/generic_reply.pb.h>
#include <network/msg_context.h>
#include <stdlib.h>

#define SUCCESSFUL 0 // libevent API 表示成功的值

namespace tcp_kit {

    // 作为 server 的通用协议实现
    class generic {
    public:
        class protobuf_deserializer;
        class protobuf_serializer;

        using filters = type_list<protobuf_deserializer, api_dispatcher_p, protobuf_serializer>;

        template<uint16_t PORT>
        class ev_handler: public ev_handler_base {
        public:
            ev_handler();
            ~ev_handler();
#ifdef __APPLE__
            static void accept_callback0(evutil_socket_t, short, void *arg);
#endif
            static void accept_callback(evconnlistener *listener, socket_t fd, sockaddr *address, int socklen, void *arg);
            static void read_callback(bufferevent *bev, void *arg);
            static void write_callback(bufferevent *bev, void *arg);
            static void event_callback(bufferevent *bev, short what, void *arg);
            static void process_callback(evutil_socket_t, short, void *arg);
            static void process_error_callback(evutil_socket_t, short, void *arg);

            static msg_context* msg_context_new(ev_context *ctx, char *msg_line, const size_t in_len);

            static void when_error(ev_context *ctx);
            static bool try_close(ev_context *ctx);
            static void terminate(ev_context *ctx);
            static bool try_free_ctx(ev_context *ctx);
            static void msg_ctx_free(msg_context *ctx);

        protected:
            server<generic, PORT>        *_server;
            event_base                   *_ev_base;
            std::mutex                    _mutex;
            size_t                        _next;
            static std::atomic<uint32_t>  _id_alloc;
#ifdef __APPLE__
            event *_accept_ev;
            event *init(server_base *server_ptr) override;
#elif __linux__
            evconnlistener        *_evc;
            void init(server_base *server_ptr) override;
#endif
            void run() override;
            inline handler_base* next();
        };

        class handler: public handler_base {
        public:
            handler() = default;

        protected:
            void init(server_base *server_ptr) override;
            void run() override;
            inline msg_context* pop();

        };

        template<uint16_t PORT>
        class api_dispatcher {
        public:
            static std::unique_ptr<GenericReply> process(msg_context *ctx, std::unique_ptr<GenericMsg> msg);

            template<typename Processor>
            static void api(const std::string &id, Processor prcs);

            template<typename T>
            static std::unique_ptr<GenericReply> serialize(T &data);

            template<typename Tuple>
            static Tuple deserialize(msg_context *ctx, std::unique_ptr<GenericMsg> &);

        private:
            using map_t = std::unordered_map<std::string , std::function<std::unique_ptr<GenericReply>(msg_context *ctx, std::unique_ptr<GenericMsg>)>>;
            static map_t _api_map;

        };

        class protobuf_deserializer {
        public:
            static std::unique_ptr<GenericMsg> process(msg_context *ctx, std::unique_ptr<msg_buffer> input);
        };

        class protobuf_serializer {
        public:
            static std::unique_ptr<msg_buffer> process(msg_context *ctx, std::unique_ptr<GenericReply> input);
        };

    };

    template<uint16_t PORT>
    std::atomic<uint32_t> generic::ev_handler<PORT>::_id_alloc{0};

#ifdef __APPLE__
    template<uint16_t PORT>
    void generic::ev_handler<PORT>::accept_callback0(int, short, void *arg) {
        auto *ev_handler_ = static_cast<generic::ev_handler<PORT> *>(arg);
        conn_info *c_info_ = &ev_handler_->c_info;
        // TODO: 传 nullptr 可能有问题
        accept_callback(nullptr, c_info_->fd, c_info_->address, c_info_->socklen, arg);
    }
#endif

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::accept_callback(evconnlistener *listener, socket_t fd, sockaddr *address, int socklen, void *arg) {
        auto *ev_handler_ = static_cast<generic::ev_handler<PORT> *>(arg);
        bufferevent *bev = bufferevent_socket_new(ev_handler_->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        if(!bev) {
            log_error("Failed to allocate the bufferevent");
            bufferevent_free(bev);
            return;
        }
        ev_context *ctx = new ev_context{{0, ev_context::CONNECTED, 0}, fd, address, socklen,
                                          ev_handler_->_id_alloc++, ev_handler_, ev_handler_->next(), bev};
        try {
            ev_handler_->call_conn_filters(ctx);
            ev_handler_->register_read_write_filters(ctx);
            ctx->ctl.state = ev_context::READY;
            if(bufferevent_enable(ctx->bev, EV_READ | EV_WRITE) == SUCCESSFUL) {
                bufferevent_setcb(ctx->bev, read_callback, write_callback, event_callback, ctx);
                ctx->ctl.state = ev_context::ACTIVE;
            } else {
                throw generic_error<CONS_BEV_FAILED>("Failed to enable the read/write events of bufferevent");
            }
        } catch (const std::exception &err) {
            log_error(err.what());
            when_error(ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::read_callback(bufferevent *bev, void *arg) {
        auto *ctx = static_cast<ev_context *>(arg);
        msg_context *msg_ctx;
        try {
            if(ctx->ctl.state == ev_context::ACTIVE) {
                evbuffer *input = bufferevent_get_input(ctx->bev);
                size_t len = 0;
                char *msg_line = evbuffer_readln(input, &len, EVBUFFER_EOL_CRLF);
                if(msg_line) {
                    msg_ctx = msg_context_new(ctx, msg_line, len);
                    ctx->handler->msg_queue->push(msg_ctx);
                    ++ctx->ctl.n_async;
                }
            } else {
                try_free_ctx(ctx);
            }
        } catch (const std::exception &err) {
            log_error(err.what());
            if(msg_ctx)
                msg_ctx_free(msg_ctx);
            when_error(ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::write_callback(bufferevent *bev, void *arg) {

    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::event_callback(bufferevent *bev, short what, void *arg) {
        auto *ctx = static_cast<ev_context *>(arg);
        if(what & BEV_EVENT_EOF) {
            log_debug("CONNECTION WILL CLOSE");
            try_free_ctx(ctx);
        } else if(what & BEV_EVENT_ERROR) {
            when_error(ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::process_callback(int, short, void *arg) {
        auto pair = static_cast<std::pair<ev_context*, msg_context*>*>(arg);
        ev_context *ctx = pair->first;
        msg_context *msg_ctx = pair->second;
        delete pair;
        --ctx->ctl.n_async;
        if(ctx->ctl.state == ev_context::ACTIVE) {
            evbuffer_add_reference(bufferevent_get_output(ctx->bev), msg_ctx->out, msg_ctx->out_len,
                                   [](const void *data, size_t len, void *arg) { free(static_cast<char *>(arg)); },
                                   msg_ctx->out);
            msg_ctx->out = nullptr;
            msg_ctx_free(msg_ctx);
        } else {
            msg_ctx_free(msg_ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::process_error_callback(int, short, void *arg) {
        auto pair = static_cast<std::pair<ev_context *, msg_context *> *>(arg);
        ev_context *ctx = pair->first;
        msg_context *msg_ctx = pair->second;
        delete pair;
        --ctx->ctl.n_async;
        msg_ctx_free(msg_ctx);
        when_error(ctx);
        try_free_ctx(ctx);
    }

    template <uint16_t PORT>
    msg_context* generic::ev_handler<PORT>::msg_context_new(ev_context *ctx, char *msg_line, const size_t in_len) {
        auto *base = static_cast<ev_handler<PORT> *>(ctx->ev_handler)->_ev_base;
        msg_context *msg_ctx = new msg_context{ctx->conn_id, msg_line, in_len, nullptr, 0, false, nullptr, nullptr, false};
        auto ctx_pair = new std::pair<ev_context *, msg_context *>(ctx, msg_ctx);
        msg_ctx->done_ev = event_new(base, -1, 0, process_callback, ctx_pair);
        msg_ctx->error_ev = event_new(base, -1, 0, process_error_callback, ctx_pair);
        if(msg_ctx->done_ev && msg_ctx->error_ev) {
            return msg_ctx;
        } else {
            delete ctx_pair;
            msg_ctx_free(msg_ctx);
            throw generic_error<CONS_EVENT_FAILED>("Failed to construct the done event");
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::when_error(ev_context *ctx) {
        ctx->ctl.error = true;
        const char *err_msg = "An error_flag occurred, the connection will be closed shortly.\n";
        bufferevent_write(ctx->bev, err_msg, strlen(err_msg));
    }

    template<uint16_t PORT>
    bool generic::ev_handler<PORT>::try_close(ev_context *ctx) {
        if(ctx->ctl.state >= ev_context::CLOSED)
            return true;
        ctx->ctl.state = ev_context::CLOSING;
        if(ctx->ctl.n_async == 0) {
            auto *ev_handler_ = static_cast<generic::ev_handler<PORT>*>(ctx->ev_handler);
            ev_handler_->call_close_filters(ctx);
            bufferevent_free(ctx->bev);
            ctx->bev = nullptr;
            ctx->ctl.state = ev_context::CLOSED;
            return true;
        }
        return false;
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::terminate(ev_context* ctx) {
        ctx->ctl.state = ev_context::TERMINATED;
    }

    template<uint16_t PORT>
    bool generic::ev_handler<PORT>::try_free_ctx(ev_context* ctx) {
        if(try_close(ctx)) {
            terminate(ctx);
            delete ctx;
            return true;
        }
        return false;
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::msg_ctx_free(msg_context* ctx) {
        if(ctx->in) free(ctx->in);
        if(ctx->out) free(ctx->out);
        if(ctx->done_ev) event_free(ctx->done_ev);
        if(ctx->error_ev) event_free(ctx->error_ev);
        delete ctx;
    }

    template<uint16_t PORT>
    generic::ev_handler<PORT>::ev_handler(): _ev_base(event_base_new()), _next(0) { }

#ifdef __APPLE__
    template<uint16_t PORT>
    event *generic::ev_handler<PORT>::init(server_base *server_ptr) {
        _server = static_cast<server<generic, PORT>*>(server_ptr);
        _accept_ev =  event_new(_ev_base, -1, EV_PERSIST, accept_callback0, this);
        return _accept_ev;
    }
#elif __linux__
    template<uint16_t PORT>
    void generic::ev_handler<PORT>::init(server_base* server_ptr) {
        _server = static_cast<server<generic, PORT>*>(server_ptr);
        sockaddr_in sin = socket_address(PORT);
        _evc = evconnlistener_new_bind(
                _ev_base, &generic::ev_handler<PORT>::accept_callback, this,
                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                -1, (sockaddr*) &sin, sizeof(sin));
    }
#endif


    template<uint16_t PORT>
    void generic::ev_handler<PORT>::run() {
#ifdef __APPLE__
        event_base_loop(_ev_base, EVLOOP_NO_EXIT_ON_EMPTY);
#elif __linux
        event_base_dispatch(_ev_base);
#endif
    }

    template<uint16_t PORT>
    handler_base* generic::ev_handler<PORT>::next() {
        handler_base* handler_ = handlers[_next];
        if(++_next == n_handler)
            _next = 0;
        return handler_;
    }

    template<uint16_t PORT>
    generic::ev_handler<PORT>::~ev_handler() {
#ifdef __APPLE__
        if(_accept_ev)
            event_free(_accept_ev);
#elif __linux
        if(_evc)
            evconnlistener_free(_evc);
#endif
        if(_ev_base)
            event_base_free(_ev_base);
    }

    // -----------------------------------------------------------------------------------------------------------------

    template<uint16_t PORT>
    typename generic::api_dispatcher<PORT>::map_t generic::api_dispatcher<PORT>::_api_map;

    template<uint16_t PORT>
    std::unique_ptr<GenericReply> generic::api_dispatcher<PORT>::process(msg_context* ctx, std::unique_ptr<GenericMsg> msg) {
        auto it = api_dispatcher<PORT>::_api_map.find(msg->api());
        if(it != api_dispatcher<PORT>::_api_map.end()) {
            return it->second(ctx, std::move(msg));
        } else {
            std::unique_ptr<GenericReply> reply = std::make_unique<GenericReply>();
            reply->set_code(GenericReply::RES_NOT_FOUND);
            return reply;
        }
    }

    template<uint16_t PORT>
    template<typename Processor>
    void generic::api_dispatcher<PORT>::api(const std::string& id, Processor prcs) {
        using result_t = typename func_traits<Processor>::result_type;
        using args_t = typename func_traits<Processor>::args_type;
        api_dispatcher<PORT>::_api_map[id] = [prcs](msg_context *ctx, std::unique_ptr<GenericMsg> msg) -> std::unique_ptr<GenericReply> {
            try {
                args_t args = deserialize<args_t>(ctx, msg);
                result_t res = call(prcs, move(args));
                return serialize(res);
            } catch (const std::exception &err) {
                log_error(err.what());
                std::unique_ptr<GenericReply> reply = std::make_unique<GenericReply>();
                reply->set_code(GenericReply::ERROR);
                reply->set_msg(err.what());
                return reply;
            }
        };
    }

    // -----------------------------------------------------------------------------------------------------------------

    // 推断类型 T 是否与 Any... 中任意类型匹配
    // match_any<bool, int, bool, double, float>::value -> true
    // match_any<long, int, bool, double, float>::value -> false
    template<typename T, typename... Any>
    struct match_any {

        using tp = std::tuple<Any...>;

        template<size_t N>
        static constexpr bool or_conditions() {
            return std::is_same<T, typename std::tuple_element<N, tp>::type>::value || or_conditions<N - 1>();
        }

        template<>
        static constexpr bool or_conditions<0>() {
            return std::is_same<T, typename std::tuple_element<0, tp>::type>::value;
        }

        static constexpr bool value = or_conditions<std::tuple_size<tp>::value - 1>();

    };

    template<typename T>
    struct match_basic_type : match_any<T, uint32_t, int32_t, uint64_t, int64_t, float, double, bool, std::string> {};


    template<typename Tuple, size_t N>
    struct infer_offset {
        static constexpr int32_t value = (match_basic_type<typename std::tuple_element<N, Tuple>::type>::value ? 1 : 0) +
                                         infer_offset<Tuple, N - 1>::value;
    };

    template<typename Tuple>
    struct infer_offset<Tuple, 0> {
        static constexpr int32_t value = match_basic_type<typename std::tuple_element<0, Tuple>::type>::value ? 0 : -1;
    };

    template<typename T, uint32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<!std::is_base_of<google::protobuf::Message, T>::value && !match_basic_type<T>::value && !std::is_same<T, msg_context *>::value>::type* = nullptr) {
        throw generic_error<UNSUPPORTED_TYPE>("Only the following types are supported as parameters for the API handler: [unsigned int 32, signed int 32, unsigned int 64, signed int 64, float, double, boolean, string, msg_context *, any type that conforms to the Protobuf 3 specification].", typeid(T).name());
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_base_of<google::protobuf::Message, T>::value>::type* = nullptr) {
        T t;
        if(msg->body().UnpackTo(&t)) {
            return move(t);
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message body into a [%s] type failed.", typeid(T).name());
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *ctx, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, msg_context *>::value>::type* = nullptr) {
        return ctx;
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, uint32_t>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_u32()) {
            return p.u32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, int32_t>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_s32()) {
            return p.s32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, uint64_t>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_u64()) {
            return p.u64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, int64_t>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_s64()) {
            return p.s64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_f()) {
            return p.f();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [float] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_d()) {
            return p.d();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [double] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_b()) {
            return p.b();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [boolean] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(msg_context *, std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, std::string>::value>::type* = nullptr) {
        BasicType p = msg->params(Offset);
        if(p.has_str()) {
            return move(p.str());
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [std::string] type failed.");
        }
    }

    template<typename Tuple, size_t... I>
    Tuple unpack(index_seq<I...>, msg_context *ctx, std::unique_ptr<GenericMsg>& msg) {
        Tuple res = {unpack_to<typename std::tuple_element<I, Tuple>::type, infer_offset<Tuple, I>::value>(ctx, msg)...};
        return res;
    }

    template<uint16_t PORT>
    template<typename Tuple>
    Tuple generic::api_dispatcher<PORT>::deserialize(msg_context *ctx, std::unique_ptr<GenericMsg>& msg) {
        if(std::tuple_size<Tuple>::value && (msg->params_size() || msg->has_body())) {
            return unpack<Tuple>(make_index_seq<Tuple>(), ctx, msg);
        }
        return Tuple();
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, google::protobuf::Message>::value>::type* = nullptr) {
        reply->body().PackFrom(data);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, uint32_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_u32(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, int32_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_s32(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, uint64_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_u64(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, int64_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_s64(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, float_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_f(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, double_t>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_d(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_b(data);
        reply->set_allocated_result(result);
    }

    template<typename T>
    void pack_from(std::unique_ptr<GenericReply>& reply, T& data, typename std::enable_if<std::is_same<T, std::string>::value>::type* = nullptr) {
        GenericReply::BasicType* result = new GenericReply::BasicType;
        result->set_str(data);
        reply->set_allocated_result(result);
    }

    template<uint16_t PORT>
    template<typename T>
    std::unique_ptr<GenericReply> generic::api_dispatcher<PORT>::serialize(T& data) {
        std::unique_ptr<GenericReply> reply = std::make_unique<GenericReply>();
        reply->set_code(GenericReply::SUCCESS);
        pack_from(reply, data);
        return reply;
    }

}