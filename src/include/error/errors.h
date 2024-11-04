#ifndef TCP_KIT_ERRORS_H
#define TCP_KIT_ERRORS_H

#include <stdexcept>

namespace tcp_kit {

    namespace errors {

        std::string format_msg(const char* fmt, ...);

        enum flags {
            CONS_BEV_ERR = 0, // 构造 bufferevent 时出错
            PRCS_ARG_ERR = 1  // 匹配 process 过滤器参数时出错
        };

        template<flags F>
        class generic_error: public std::runtime_error {

        public:

            template<typename... Args>
            generic_error(const char* fmt, Args... args): std::runtime_error(format_msg(fmt, args...)) { }

        };

    }

}

#endif