#pragma once

#include <stdexcept>

namespace tcp_kit {

        std::string format_msg(const char* fmt, ...);

        enum error_flags {
            CONS_BEV_FAILED,     // 构造 bufferevent 时出错
            CONS_EVENT_FAILED,   // 构造 event 时出错
            PRCS_ARG_MISMATCHED, // 匹配 process 过滤器参数时出错
            API_ARGS_MISMATCHED, // 匹配 api 参数时出错
            UNSUPPORTED_TYPE,    // 不支持的类型
            RES_NOT_FOUND,       // 资源不存在
            ILLEGALITY_ARGS,     // 非法参数
            SERIALIZE_MSG_ERROR  // 序列化消息失败
        };

        template<error_flags F>
        class generic_error: public std::runtime_error {

        public:

            template<typename... Args>
            generic_error(const char* fmt, Args... args): std::runtime_error(format_msg(fmt, args...)) { }

        };


}
