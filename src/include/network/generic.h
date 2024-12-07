#ifndef TCP_KIT_GENERIC_H
#define TCP_KIT_GENERIC_H

#include <network/server.h>
#include <include/logger/logger.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include <include/error/errors.h>
#include <unordered_map>
#include <network/generic_msg.pb.h>
#include <network/generic_reply.pb.h>

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

        protected:
            server<generic, PORT>* _server;
            event_base*            _ev_base;
            mutex                  _mutex;
            evconnlistener*        _evc;

            void init(server_base* server_ptr) override;
            void run() override;

        };

        class handler: public handler_base {
        public:
            handler() = default;

        protected:
            void init(server_base* server_ptr) override;
            void run() override;

        };

        template<uint16_t PORT>
        class api_dispatcher {

        public:

            static unique_ptr<GenericReply> process(const event_context* ctx, unique_ptr<GenericMsg> msg);

            template<typename Processor>
            static void api(const string& id, Processor prcs);

            template<typename T>
            static unique_ptr<GenericReply> serialize(T& data);

            template<typename Tuple>
            static Tuple deserialize(unique_ptr<GenericMsg>&);

        private:
            using map_t = unordered_map<string , function<unique_ptr<GenericReply>(unique_ptr<GenericMsg>)>>;
            static map_t _api_map;

        };

    };

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        generic::ev_handler<PORT>* ev_handler = static_cast<generic::ev_handler<PORT> *>(arg);
        bufferevent* bev = bufferevent_socket_new(ev_handler->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        if(!bev) {
            log_error("Failed to allocate the bufferevent");
            close_socket(fd);
            return;
        }
        event_context* ctx = new event_context{fd, address, socklen, bev};
        try {
             ev_handler->call_conn_filters(ctx);
             ev_handler->register_read_write_filters(ctx);
            if(bufferevent_enable(ctx->bev, EV_READ | EV_WRITE) == SUCCESSFUL) {
                bufferevent_setcb(ctx->bev,read_callback, write_callback, event_callback, ctx);
            } else {
                throw generic_error<CONS_BEV_FAILED>("Failed to enable the events of bufferevent");
            }
        } catch (const exception& err) {
            log_error(err.what());
            bufferevent_free(ctx->bev);
            delete ctx;
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::read_callback(bufferevent *bev, void *arg) {
        try {
            generic::ev_handler<PORT>* ev_handler = static_cast<generic::ev_handler<PORT>*>(arg);
        } catch (const exception& err) {
            log_error(err.what());
        }
    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::write_callback(bufferevent *bev, void *arg) {

    }

    template<uint16_t PORT>
    void generic::ev_handler<PORT>::event_callback(bufferevent *bev, short what, void *arg) {

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
    generic::ev_handler<PORT>::~ev_handler() {
        evconnlistener_free(_evc);
        event_base_free(_ev_base);
    }

    // -----------------------------------------------------------------------------------------------------------------

    template<uint16_t PORT>
    typename generic::api_dispatcher<PORT>::map_t generic::api_dispatcher<PORT>::_api_map;

    template<uint16_t PORT>
    unique_ptr<GenericReply> generic::api_dispatcher<PORT>::process(const event_context* ctx, unique_ptr<GenericMsg> msg) {
        api_dispatcher<PORT>::_api_map.find(msg->api())->second(move(msg));
    }

    template<uint16_t PORT>
    template<typename Processor>
    void generic::api_dispatcher<PORT>::api(const string& id, Processor prcs) {
        using result_t = typename func_traits<Processor>::result_type;
        using args_t = typename func_traits<Processor>::args_type;
        api_dispatcher<PORT>::_api_map[id] = [prcs](unique_ptr<GenericMsg> msg) -> unique_ptr<GenericReply> {
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

        using tp = tuple<Any...>;

        template<size_t N>
        static constexpr bool or_conditions() {
            return is_same<T, tuple_element_t<N, tp>>::value || or_conditions<N - 1>();
        }

        template<>
        static constexpr bool or_conditions<0>() {
            return is_same<T, tuple_element_t<0, tp>>::value;
        }

        static constexpr bool value = or_conditions<tuple_size<tp>::value - 1>();

    };

    template<typename T>
    struct match_basic_type : match_any<T, uint32_t, int32_t, uint64_t, int64_t, float, double, bool, string> {};


    template<typename Tuple, size_t N>
    struct infer_offset {
        static constexpr int32_t value = (match_basic_type<tuple_element_t<N, Tuple>>::value ? 1 : 0) +
                                        infer_offset<Tuple, N - 1>::value;
    };

    template<typename Tuple>
    struct infer_offset<Tuple, 0> {
        static constexpr int32_t value = match_basic_type<tuple_element_t<0, Tuple>>::value ? 0 : -1;
    };

    template<typename T, uint32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<!is_base_of<google::protobuf::Message, T>::value && !match_basic_type<T>::value>::type* = nullptr) {
        throw generic_error<UNSUPPORTED_TYPE>("Only the following types are supported as parameters for the API handler: [unsigned int 32, signed int 32, unsigned int 64, signed int 64, float, double, boolean, string, any type that conforms to the Protobuf 3 specification].", typeid(T).name());
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_base_of<google::protobuf::Message, T>::value>::type* = nullptr) {
        T t;
        if(msg->body().UnpackTo(&t)) {
            return move(t);
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message body into a [%s] type failed.", typeid(T).name());
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, uint32_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_u32()) {
            return p.u32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, int32_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_s32()) {
            return p.s32();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 32] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, uint64_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_u64()) {
            return p.u64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [unsigned int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, int64_t>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_s64()) {
            return p.s64();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [int 64] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, float>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_f()) {
            return p.f();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [float] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, double>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_d()) {
            return p.d();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [double] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, bool>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_b()) {
            return p.b();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [boolean] type failed.");
        }
    }

    template<typename T, int32_t Offset>
    T unpack_to(unique_ptr<GenericMsg>& msg, typename enable_if<is_same<T, std::string>::value>::type* = nullptr) {
        Param p = msg->params(Offset);
        if(p.has_str()) {
            return p.str();
        } else {
            throw generic_error<API_ARGS_MISMATCHED>("The attempt to unpack the message parameter into a [std::string] type failed.");
        }
    }

    template<typename Tuple, size_t... I>
    Tuple unpack(index_seq<I...>, unique_ptr<GenericMsg>& msg) {
        return {unpack_to<tuple_element_t<I, Tuple>, infer_offset<Tuple, I>::value>(msg)...};
    }

    template<uint16_t PORT>
    template<typename Tuple>
    Tuple generic::api_dispatcher<PORT>::deserialize(unique_ptr<GenericMsg>& msg) {
        if(tuple_size<Tuple>::value && (msg->params_size() || msg->has_body())) {
            return unpack<Tuple>(make_index_seq<Tuple>(), msg);
        }
        return Tuple();
    }

    template<uint16_t PORT>
    template<typename T>
    unique_ptr<GenericReply> generic::api_dispatcher<PORT>::serialize(T& data) {
        unique_ptr<GenericReply> reply = make_unique<GenericReply>();
        reply->set_code(GenericReply::SUCCESS);
//        reply->body().PackFrom(data); // TODO
        return move(reply);
    }

}

#endif