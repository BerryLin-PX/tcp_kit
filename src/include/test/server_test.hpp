#include <logger/logger.h>
#include <network/server.h>
#include <network/generic.h>
#include <tuple>
#include <type_traits>
#include <util/func_traits.h>
#include <network/json.h>
#include <google/protobuf/util/json_util.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <thread>
#include <random>

namespace tcp_kit {

    namespace server_test {

        using namespace std;

        bool is_multiple(uint16_t a, uint16_t b) {
            return a % b == 0 || b % a == 0;
        }

        void adjust_to_multiple(uint16_t& a, uint16_t& b) {
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

        uint32_t n_with_scale(uint16_t n_of_processor, float scale, uint16_t n_ev_handler = 0, uint16_t n_handler = 0) {
            uint32_t _ctl = 0;
            if(n_ev_handler > EV_HANDLER_CAPACITY || n_handler > HANDLER_CAPACITY)
                throw std::invalid_argument("Illegal Parameter.");
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

        // 当 ev_handler 线程数不等于 handler 线程数时, 就认为存在竞争, 所以应尽量让他们的数量相等
        // 1. ev_handler 线程数量多于 handler 数量时:
        //      假设 ev_handler: 4, handler:3 那么首先三个 ev_handler 线程各自被分配一个handler 线程,
        //      多出来的一个 ev_handler 线程将轮询将任务分配给这三个 handler 线程, 相当于在一个 handler 上
        //      的同一时刻可能存在两个线程竞争, 而另外两个线程不受到影响. 在 ev_handler 不过多大于 handler
        //      的前提下, 竞争是相当小的
        // 2. ev_handler 线程数量少于 handler 数量时:
        //      假设 ev_handler: 60, handler: 69 同样首先为每个 ev_handler 线程各自被分配一个 handler 线程,
        //      handler 比 ev_handler 多出来的 9 个线程将按照 60 / 9 ≈ 6.667 向上取整 => 7 个线程分配一个
        //      多出来的 handler 线程, 也就是 8 组 7 个 ev_handler 线程分配8个多出来的 handler 线程, 剩下
        //      4 个 ev_handler 线程分配第九个 handler 线程, 由于 ev_handler 的任务优先递交给自己独占的
        //      handler 线程, 当这个线程无法处理时, 才分配给那个额外的 handler 线程, 所以即便 7 个线程有可能
        //      会在这个额外的线程产生竞争, 达到最大竞争的可能性也是很小的
        void t9() {
            float total = 0;
            uint32_t times = 100;
            for(uint32_t i = 1; i <= times; ++i) {
                auto ctl = n_with_scale(i, 0.3f , 0, 0);
                auto n_ev_handler = (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY;
                auto n_handler = ctl & HANDLER_CAPACITY;
                // log_info("n_processor: %d | n_ev_handler: %d | n_handler: %d | gap: %d", i, n_ev_handler, n_handler, n_ev_handler - n_handler);
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

        class a_class {};

        void t11() {
//            using types = replace_type<type_list<api_dispatcher_p>, api_dispatcher_p, a_class>::type;
//            types{};
        }

        void t12() {
//            server<generic> svr;
//            svr.api("echo", [](string msg){ return msg; });
//            svr.start();
        }

        void t13() {
//            server<generic, 3000> svr;
//            svr.start();
        }

        void t14() {
            server<json> svr;
            svr.api("plus", [](int32_t a, int32_t b) {
                return a + b;
            });
            svr.api("echo", [](std::string msg) {
                return msg;
            });
            svr.start();
        }

        void t15() {
            server<generic> svr;
            svr.api("echo", [](std::string msg) {
                return msg;
            });
            svr.start();
        }

        void adjust_func_test() {
            std::vector<std::pair<uint16_t, uint16_t>> pairs({{6, 4}, {7, 5}, {8, 5}, {12, 5}, {6, 6}});
            for(auto &pair : pairs) {
                auto a = pair.first; auto b = pair.second;
                adjust_to_multiple(a, b);
                log_info("%d:%d -> %d:%d", pair.first, pair.second, a, b);
            }
        }

        void thread_allocation_strategy_test() {
            for(uint16_t i = 1; i <= 10; ++i) {
                auto ctl = n_with_scale(i, 0.3f);
                log_info("core: %d | n of ev_handler: %d | n of handler: %d",
                         i,
                         (ctl >> EV_HANDLER_OFFSET) & EV_HANDLER_CAPACITY,
                         ctl & HANDLER_CAPACITY);
            }
        }

        void test_api() {
            server<json> svr;
            svr.api("plus", [](int32_t a, int32_t b) {
                return a + b;
            });
            svr.api("echo", [](std::string msg) {
                return msg;
            });
            svr.start();
        }

        void random_sleep() {
            static thread_local std::mt19937_64 rng(std::random_device{}());  // 线程局部随机数生成器
            std::uniform_int_distribution<int> dist(1, 10);
            std::this_thread::sleep_for(std::chrono::milliseconds (dist(rng))); // 随机休眠
        }

        void log_server() {
            server<json> svr;
            constexpr const char *log_directory = "/Users/linruixin/tcp_kit/log/";
            constexpr const char *level_file[] = {"debug", "info", "warning", "error"};
            std::mutex mutexes[4];
            // 0-debug 1-info 2-warning 3-error
            svr.api("log", [&](uint32_t level, std::string log) {
                random_sleep();
                std::unique_lock<std::mutex> lock(mutexes[level]);
                std::ostringstream filepath;
                filepath << log_directory << level_file[level] << ".log";
                std::ofstream file(filepath.str(), std::ios::app);
                if (file) {
                    file << log << std::endl;
                    return true;
                }
                return false;
            });
            svr.start();
        }

        constexpr const char *LOG_DIRECTORY = "/Users/linruixin/tcp_kit/log/";
        constexpr const char *LEVEL_FILE[] = {"debug", "info", "warning", "error"};
        std::mutex mutexes[4];

        bool write_log(uint32_t level, const std::string &log) {
            random_sleep();
            std::unique_lock<std::mutex> lock(mutexes[level]);
            std::ostringstream filepath;
            filepath << LOG_DIRECTORY << LEVEL_FILE[level] << ".log";
            std::ofstream file(filepath.str(), std::ios::app);
            if (file) {
                file << log << std::endl;
                return true;
            }
            return false;
        }

        struct worker_context {
            struct event_base *base;
            struct event *notify_event;
            evutil_socket_t fd;
        };

        std::vector<worker_context> workers;
        std::atomic<int> next_worker_index{0};

        void read_cb(struct bufferevent *bev, void *ctx) {
            evbuffer *input = bufferevent_get_input(bev);
            size_t len = 0;
            char *msg_line = evbuffer_readln(input, &len, EVBUFFER_EOL_CRLF);
            if (msg_line) {
                std::string json_str(msg_line);
                GenericMsg msg;
                if (google::protobuf::util::JsonStringToMessage(json_str, &msg).ok()) {
                    GenericReply reply;
                    reply.set_code(GenericReply::SUCCESS);
                    std::string json_reply;
                    auto *res = new GenericReply_BasicType;
                    if (msg.api() == "log") {
                        res->set_b(write_log(msg.params(0).u32(), msg.params(1).str()));
                        reply.set_allocated_result(res);
                    } else {
                        reply.set_code(GenericReply::RES_NOT_FOUND);
                    }
                    google::protobuf::util::MessageToJsonString(reply, &json_reply);
                    json_reply += "\r\n";
                    bufferevent_write(bev, json_reply.c_str(), json_reply.size());
                }
            }
        }

        void event_cb(struct bufferevent *bev, short events, void *ctx) {
            if (events & (BEV_EVENT_EOF | BEV_EVENT_ERROR)) {
                bufferevent_free(bev);
            }
        }

        void worker_cb(evutil_socket_t fd, short, void *ctx) {
            worker_context *worker = static_cast<worker_context *>(ctx);
            struct bufferevent *bev = bufferevent_socket_new(worker->base, worker->fd, BEV_OPT_CLOSE_ON_FREE);
            bufferevent_setcb(bev, read_cb, nullptr, event_cb, nullptr);
            bufferevent_enable(bev, EV_READ | EV_WRITE);
        }

        void accept_cb(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *, int, void *ctx) {
            int worker_id = next_worker_index.fetch_add(1) % workers.size();
            workers[worker_id].fd = fd;
            event_active(workers[worker_id].notify_event, 0, 0);
        }

        void run_worker(int index) {
            struct event_base *base = event_base_new();
            struct event *notify_event = event_new(base, -1, EV_PERSIST, worker_cb, &workers[index]);
            workers[index] = {base, notify_event};
            event_add(notify_event, nullptr);
            event_base_loop(base, EVLOOP_NO_EXIT_ON_EMPTY);
            event_free(notify_event);
            event_base_free(base);
        }

        void run_acceptor() {
            struct event_base *base = event_base_new();
            struct sockaddr_in sin = {};
            bzero(&sin, sizeof(sockaddr_in));
            sin.sin_family = AF_INET;
            sin.sin_addr.s_addr = htonl(INADDR_ANY);
            sin.sin_port = htons(3000);
            evconnlistener *listener = evconnlistener_new_bind(
                    base, accept_cb, nullptr,
                    LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                    -1, (sockaddr *) &sin, sizeof(sin));
            event_base_dispatch(base);
            evconnlistener_free(listener);
            event_base_free(base);
        }

        void log_server_by_libevent() {
            evthread_use_pthreads();
//            int thread_count = tcp_kit::numb_of_processor() * 2;
            int thread_count = 6;
            workers.resize(thread_count - 1);
            std::vector<std::thread> threads;
            threads.emplace_back(run_acceptor);
            for (int i = 0; i < thread_count - 1; ++i) {
                threads.emplace_back(run_worker, i);
            }
            for (auto &t : threads) {
                t.join();
            }
        }

        class conn_manager {
        public:
            struct conn_context {
                ev_context *ev_ctx;
                event      *msg_event;
                evbuffer   *msg_buffer;
            };
            static std::unordered_map<uint32_t, conn_context> cli_map;
            static void msg_callback(evutil_socket_t, short, void *arg) {
                auto *ev_ctx = static_cast<ev_context *>(arg);
                bufferevent_write_buffer(ev_ctx->bev, cli_map[ev_ctx->conn_id].msg_buffer);
            }
            static void connect(ev_context *ctx) {
                cli_map[ctx->conn_id] = {ctx, nullptr, evbuffer_new()};
                event *ev = event_new(bufferevent_get_base(ctx->bev), -1, EV_PERSIST, msg_callback, ctx);
                cli_map[ctx->conn_id].msg_event = ev;
            }
            static void send_global_msg(std::string &msg) {
                for(auto &cli_pair : cli_map) {
                    std::string json = "{\"msg\":\"" + msg + "\"}\r\n";
                    evbuffer_add(cli_pair.second.msg_buffer, json.c_str(), json.length());
                    event_active(cli_pair.second.msg_event, 0, 0);
                }
            }
            static void close(ev_context *ctx) {
                log_info("CLOSE");
                auto it = cli_map.find(ctx->conn_id);
                if (it != cli_map.end()) {
                    evbuffer_free(it->second.msg_buffer);
                    event_free(it->second.msg_event);
                    cli_map.erase(it);
                }
            }
        };

        std::unordered_map<uint32_t, conn_manager::conn_context> conn_manager::cli_map;

        class chat: public tcp_kit::json {
        public:
            using filters = type_list<conn_manager, json_deserializer, api_dispatcher_p, json_serializer>;
        };

        void chat_room() {
            server<chat> chat_server;
            std::unordered_map<uint32_t, std::string> user_map;
            chat_server.api("join", [&](msg_context *ctx, std::string name) {
                user_map[ctx->conn_id] = name;
                return ctx->conn_id;
            });
            chat_server.api("send", [&](uint32_t id, std::string msg) {
                auto it = user_map.find(id);
                if(it != user_map.end()) {
                    auto name_with_msg = it->second + ":" + msg;
                    conn_manager::send_global_msg(name_with_msg);
                    return true;
                }
                return false;
            });
            chat_server.start();
        }

        static const std::string base64_chars =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/";


        static inline bool is_base64(unsigned char c) {
            return (isalnum(c) || (c == '+') || (c == '/'));
        }

        std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
            std::string ret;
            int i = 0;
            int j = 0;
            unsigned char char_array_3[3];
            unsigned char char_array_4[4];

            while (in_len--) {
                char_array_3[i++] = *(bytes_to_encode++);
                if (i == 3) {
                    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                    char_array_4[3] = char_array_3[2] & 0x3f;

                    for(i = 0; (i <4) ; i++)
                        ret += base64_chars[char_array_4[i]];
                    i = 0;
                }
            }

            if (i)
            {
                for(j = i; j < 3; j++)
                    char_array_3[j] = '\0';

                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (j = 0; (j < i + 1); j++)
                    ret += base64_chars[char_array_4[j]];

                while((i++ < 3))
                    ret += '=';

            }

            return ret;

        }

        std::string base64_decode(std::string const& encoded_string) {
            int in_len = encoded_string.size();
            int i = 0;
            int j = 0;
            int in_ = 0;
            unsigned char char_array_4[4], char_array_3[3];
            std::string ret;

            while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
                char_array_4[i++] = encoded_string[in_]; in_++;
                if (i ==4) {
                    for (i = 0; i <4; i++)
                        char_array_4[i] = base64_chars.find(char_array_4[i]);

                    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                    for (i = 0; (i < 3); i++)
                        ret += char_array_3[i];
                    i = 0;
                }
            }

            if (i) {
                for (j = i; j <4; j++)
                    char_array_4[j] = 0;

                for (j = 0; j <4; j++)
                    char_array_4[j] = base64_chars.find(char_array_4[j]);

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
            }

            return ret;
        }

        void file_system() {
            server<json> file_svr;
            const std::string directory_path_of_file = "/Users/linruixin/tcp_kit/upload/";
            std::atomic<uint32_t> uid{0};
            std::unordered_map<uint32_t, std::string> file_map;
            file_svr.api("upload", [&](std::string filename) {
                auto id = uid++;
                file_map[id] = filename;
                return id;
            });
            file_svr.api("uploading", [&](uint32_t uid, bool err, bool end, int32_t offset, std::string base64) {
                if (file_map.find(uid) == file_map.end())
                    return false;
                std::string filename = directory_path_of_file + file_map[uid];
                if (err) {
                    std::remove(filename.c_str());
                    file_map.erase(uid);
                    return false;
                }
                std::string decoded_data = base64_decode(base64);
                if(offset == 0) {
                    std::ofstream file(filename, std::ios::binary | std::ios::app);
                    file.close();
                }
                std::ofstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
                if (!file.is_open()) {
                    return false;
                }
                file.seekp(offset, std::ios::beg);
                file.write(decoded_data.data(), decoded_data.size());
                file.close();
                if (end)
                    file_map.erase(uid);
                return true;
            });
            file_svr.start();
        }

    };

}