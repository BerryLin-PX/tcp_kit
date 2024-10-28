#include <logger/logger.h>
#include <unistd.h>
#include <time.h>
#include <utility>

static const char* LEVEL_NAMES[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
static const char* LEVEL_COLORS[] = {"\x1b[94m", "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[35m"};
static const char* COLOR_RESET = "\x1b[0m";


inline int snprintf_base_msg(char* ptr, uint32_t size, uint8_t level, const char* file, uint32_t line, const tm* local_time);

// time pid [level] file line log
void log(uint8_t level, const char* file, uint32_t line, const char* fmt, ...) {
    if(level >= LOG_LEVEL) {
        va_list args;
        va_start(args, fmt);
        auto current_time = time(nullptr);
        auto* local_time = localtime(&current_time);
        auto base_msg_length = snprintf_base_msg(nullptr, 0, level, file, line, local_time);
        auto log_length = base_msg_length + vsnprintf(nullptr, 0, fmt, args);
        va_start(args, fmt);
        auto log_str = std::unique_ptr<char[]>(new char[log_length + 2]);
        snprintf_base_msg(log_str.get(), base_msg_length + 1, level, file, line, local_time);
        vsprintf(log_str.get() + base_msg_length, fmt, args);
        log_str[log_length] = '\n';
        log_str[log_length + 1] = '\0';
        if(level != LOG_ERROR) {
            printf("%s", log_str.get());
        } else {
            fprintf(stderr, "%s", log_str.get());
        }
    }
}


int snprintf_base_msg(char* ptr, uint32_t size, uint8_t level, const char* file, uint32_t line, const tm* time) {
    return snprintf(ptr, size, "%02d-%02d %02d:%02d:%02d %d %s[%s]%s %s %d: ",
                    time->tm_mon + 1,
                    time->tm_mday,
                    time->tm_hour,
                    time->tm_min,
                    time->tm_sec,
                    getpid(),
                    LEVEL_COLORS[level],
                    LEVEL_NAMES[level],
                    COLOR_RESET,
                    file,
                    line);
}