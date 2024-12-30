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

#define SUCCESSFUL 0 // libevent API 表示成功的值

namespace tcp_kit {

    // 作为 server 的通用协议实现
    class generic {

    public:

        using filters = type_list<api_dispatcher_p>;

        template<uint16_t PORT>
        class ev_handler: public ev_handler_base {
        public:
            ev_handler();
            ~ev_handler();

            static void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
            static void read_callback(bufferevent *bev, void *arg);
            static void write_callback(bufferevent *bev, void *arg);
            static void event_callback(bufferevent *bev, short what, void *arg);
            static void process_callback(evutil_socket_t, short, void * arg);

            static msg_context* msg_context_new(event_base* base, ev_context* ctx);

            static void when_error(ev_context* ctx);
            static bool try_close(ev_context* ctx);
            static void terminate(ev_context* ctx);
            static bool try_free_ctx(ev_context* ctx);
            static void free_msg_ctx(msg_context* ctx);

        protected:
            server<generic, PORT>* _server;
            event_base*            _ev_base;
            std::mutex             _mutex;
            evconnlistener*        _evc;
            size_t                 _next;

            void init(server_base* server_ptr) override;
            void run() override;
            inline handler_base* next();
        };

        class handler: public handler_base {
        public:
            handler() = default;

        protected:
            void init(server_base* server_ptr) override;
            void run() override;
            inline msg_context* pop();

        };

        template<uint16_t PORT>
        class api_dispatcher {
        public:
            static std::unique_ptr<GenericReply> process(const ev_context* ctx, std::unique_ptr<GenericMsg> msg);

            template<typename Processor>
            static void api(const std::string& id, Processor prcs);

            template<typename T>
            static std::unique_ptr<GenericReply> serialize(T& data);

            template<typename Tuple>
            static Tuple deserialize(std::unique_ptr<GenericMsg>&);

        private:
            using map_t = std::unordered_map<std::string , std::function<std::unique_ptr<GenericReply>(std::unique_ptr<GenericMsg>)>>;
            static map_t _api_map;

        };
    };

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        generic::ev_handler<PORT>* ev_handler_ = static_cast<generic::ev_handler<PORT>*>(arg);
        bufferevent* bev = bufferevent_socket_new(ev_handler_->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        if(!bev) {
            log_error("Failed to allocate the bufferevent");
            bufferevent_free(bev);
            return;
        }
        ev_context* ctx = new ev_context({0, ev_context::CONNECTED, 0}, fd, address, socklen,
                                         ev_handler_, ev_handler_->next(), bev);
        try {
            ev_handler_->call_conn_filters(ctx);
            ev_handler_->register_read_write_filters(ctx);
            ctx->ctl.state = ev_context::READY;
            if(bufferevent_enable(ctx->bev, EV_READ | EV_WRITE) == SUCCESSFUL) {
                bufferevent_setcb(ctx->bev, read_callback, write_callback, event_callback, ctx);
                ctx->ctl.state = ev_context::WORKING;
            } else {
                throw generic_error<CONS_BEV_FAILED>("Failed to enable the events of bufferevent");
            }
        } catch (const std::exception& err) {
            log_error(err.what());
            when_error(ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::read_callback(bufferevent* bev, void* arg) {
        ev_context* ctx = static_cast<ev_context*>(arg);
        msg_context* msg_ctx;
        try {
            if(ctx->ctl.state == ev_context::WORKING) {
                generic::ev_handler<PORT>* ev_handler_ = static_cast<generic::ev_handler<PORT>*>(ctx->ev_handler);
                msg_ctx = msg_context_new(ev_handler_->_ev_base, ctx);
                ctx->handler->msg_queue->push(msg_ctx);
            } else {
                try_free_ctx(ctx);
            }
        } catch (const std::exception& err) {
            log_error(err.what());
            if(msg_ctx)
                free_msg_ctx(msg_ctx);
            when_error(ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::write_callback(bufferevent* bev, void* arg) {

    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::event_callback(bufferevent* bev, short what, void* arg) {

    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::process_callback(int, short, void* arg) {
        auto pair = static_cast<std::pair<ev_context*, msg_context*>*>(arg);
        ev_context* ctx = pair->first;
        msg_context* msg_ctx = pair->second;
        delete pair;
        --ctx->ctl.n_async;
        if(ctx->ctl.state == ev_context::WORKING) {
            evbuffer_add_buffer(bufferevent_get_output(ctx->bev), msg_ctx->out);
        } else {
            free_msg_ctx(msg_ctx);
            try_free_ctx(ctx);
        }
    }

    template<uint16_t PORT>
    msg_context* generic::ev_handler<PORT>::msg_context_new(event_base* base, ev_context* ctx) {
        msg_context* msg_ctx = new msg_context{evbuffer_new(), evbuffer_new(), nullptr};
        evbuffer_add_buffer(msg_ctx->in, bufferevent_get_input(ctx->bev));
        event* done_ev = event_new(base, -1, 0, process_callback, new std::pair<ev_context*, msg_context*>(ctx, msg_ctx));
        msg_ctx->done_ev = done_ev;
        ++ctx->ctl.n_async;
        return msg_ctx;
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::when_error(ev_context* ctx) {
        ctx->ctl.error = true;
        // TODO
    }

    template<uint16_t PORT>
    bool generic::ev_handler<PORT>::try_close(ev_context* ctx) {
        if(ctx->ctl.state >= ev_context::CLOSED)
            return true;
        ctx->ctl.state = ev_context::CLOSING;
        if(ctx->ctl.n_async == 0) {
            // TODO: 关闭连接
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
    void generic::ev_handler<PORT>::free_msg_ctx(msg_context* ctx) {
        evbuffer_free(ctx->in);
        evbuffer_free(ctx->out);
        event_free(ctx->done_ev);
        delete ctx;
    }

    template<uint16_t PORT>
    generic::ev_handler<PORT>::ev_handler() {
        _ev_base = event_base_new();
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::init(server_base* server_ptr) {
        _server = static_cast<server<generic, PORT>*>(server_ptr);
        sockaddr_in sin = socket_address(PORT);
        _evc = evconnlistener_new_bind(
                _ev_base, &generic::ev_handler<PORT>::listener_callback, this,
                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                -1, (sockaddr*) &sin, sizeof(sin));
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::run() {
        event_base_dispatch(_ev_base);
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
        evconnlistener_free(_evc);
        event_base_free(_ev_base);
    }

    // -----------------------------------------------------------------------------------------------------------------

    template<uint16_t PORT>
    typename generic::api_dispatcher<PORT>::map_t generic::api_dispatcher<PORT>::_api_map;

    template<uint16_t PORT>
    std::unique_ptr<GenericReply> generic::api_dispatcher<PORT>::process(const ev_context* ctx, std::unique_ptr<GenericMsg> msg) {
        api_dispatcher<PORT>::_api_map.find(msg->api())->second(move(msg));
    }

    template<uint16_t PORT>
    template<typename Processor>
    void generic::api_dispatcher<PORT>::api(const std::string& id, Processor prcs) {
        using result_t = typename func_traits<Processor>::result_type;
        using args_t = typename func_traits<Processor>::args_type;
        api_dispatcher<PORT>::_api_map[id] = [prcs](std::unique_ptr<GenericMsg> msg) -> std::unique_ptr<GenericReply> {
            args_t args = deserialize<args_t>(msg);
            result_t res = call(prcs, move(args));
            return serialize(res);
        };
    }

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
        static constexpr int32_t value = (match_basic_type<typename std::tuple_element<N, Tuple>>::type ? 1 : 0) +
                                         infer_offset<Tuple, N - 1>::value;
    };

    template<typename Tuple>
    struct infer_offset<Tuple, 0> {
        static constexpr int32_t value = match_basic_type<typename std::tuple_element<0, Tuple>::type>::value ? 0 : -1;
    };

    template<typename T, uint32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<!std::is_base_of<google::protobuf::Message, T>::value && !match_basic_type<T>::value>::type* = nullptr) {
        throw generic_error<UNSUPPORTED_TYPE>("Only the following types are supported as parameters for the API handler: [unsigned int 32, signed int 32, unsigned int 64, signed int 64, float, double, boolean, string, any type that conforms to the Protobuf 3 specification].", typeid(T).name());
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_base_of<google::protobuf::Message, T>::value>::type* = nullptr) {
        T t;
        if(msg->body().UnpackTo(&t)) {
            return move(t);
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message body into a [%s] type failed.", typeid(T).name());
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, uint32_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_u32()) {
            return p.u32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, int32_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_s32()) {
            return p.s32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, uint64_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_u64()) {
            return p.u64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, int64_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_s64()) {
            return p.s64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_f()) {
            return p.f();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [float] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_d()) {
            return p.d();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [double] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_b()) {
            return p.b();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [boolean] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(std::unique_ptr<GenericMsg>& msg, typename std::enable_if<std::is_same<T, std::string>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_str()) {
            return p.str();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [std::string] type failed.");
        }
    }

    template<typename Tuple, size_t... I>
    Tuple unpack(index_seq<I...>, std::unique_ptr<GenericMsg>& msg) {
        return {unpack_to<typename std::tuple_element<I, Tuple>::type, infer_offset<Tuple, I>::value>(msg)...};
    }

    template<uint16_t PORT>
    template<typename Tuple>
    Tuple generic::api_dispatcher<PORT>::deserialize(std::unique_ptr<GenericMsg>& msg) {
        if(std::tuple_size<Tuple>::value && (msg->params_size() || msg->has_body())) {
            return unpack<Tuple>(make_index_seq<Tuple>(), msg);
        }
        return Tuple();
    }

    template<uint16_t PORT>
    template<typename T>
    std::unique_ptr<GenericReply> generic::api_dispatcher<PORT>::serialize(T& data) {
        std::unique_ptr<GenericReply> reply = std::make_unique<GenericReply>();
        reply->set_code(GenericReply::SUCCESS);
        // reply->body().PackFrom(data); // TODO
        return move(reply);
    }

}