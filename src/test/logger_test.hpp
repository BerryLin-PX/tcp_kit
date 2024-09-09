#ifndef TCP_KIT_LOGGER_TEST_H
#define TCP_KIT_LOGGER_TEST_H

#include <gtest/gtest.h>
#include <logger/logger.h>

TEST(logger_tests, level_print) {
    log_debug("Hello %s", "world");
    log_info("Hello %s", "world");
    log_warn("Hello %s", "world");
    log_error("Hello %s", "world");
    log_fatal("Hello %s", "world");
}

#endif