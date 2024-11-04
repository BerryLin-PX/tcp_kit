#include <error/errors.h>
#include <string>

namespace tcp_kit {

    namespace errors {

        std::string format_msg(const char* fmt, ...) {
            va_list args;
            va_start(args, fmt);
            int msg_len = snprintf(nullptr, 0, fmt, args);
            auto msg = std::make_unique<char[]>(msg_len + 1);
            vsprintf(msg.get(), fmt, args);
            va_end(args);
            msg[msg_len] = '\0';
            return std::move(std::string(msg.get()));
        }

    }

}

