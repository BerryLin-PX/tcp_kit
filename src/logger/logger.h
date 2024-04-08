#ifndef _TCP_SERVER_LOGGER_
#define _TCP_SERVER_LOGGER_

#include <stdarg.h>
#include <string>

enum levels { LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL };

#define LOG_LEVEL LOG_DEBUG
#define LOG_QUEUE_SIZE 128

#define log_debug(...) log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define log_info(...)  log(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define log_warn(...)  log(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define log_error(...) log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define log_fatal(...) log(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

void log(uint8_t level, const char *file, uint32_t line, const char *fmt, ...);

#endif
