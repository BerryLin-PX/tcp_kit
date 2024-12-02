#ifndef TCP_KIT_ERRORS_H
#define TCP_KIT_ERRORS_H

#include <stdexcept>

namespace tcp_kit {

        std::string format_msg(const char* fmt, ...);

        enum error_flags {
            CONS_BEV_FAILED,     // 构造 bufferevent 时出错
            PRCS_ARG_MISMATCHED, // 匹配 process 过滤器参数时出错
            API_ARGS_MISMATCHED, // 匹配 api 参数时出错
            UNSUPPORTED_TYPE,    // 不支持的类型
        };

        template<error_flags F>
        class generic_error: public std::runtime_error {

        public:

            template<typename... Args>
            generic_error(const char* fmt, Args... args): std::runtime_error(format_msg(fmt, args...)) { }

        };


}

#endif