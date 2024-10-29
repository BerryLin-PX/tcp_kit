#ifndef TCP_KIT_SYSTEM_UTIL_TEST_H
#define TCP_KIT_SYSTEM_UTIL_TEST_H

#include <gtest/gtest.h>
#include <util/system_util.h>
#include <logger/logger.h>

using namespace tcp_kit;

namespace tcp_kit {

    namespace system_util_test {

        void t1() {
            log_info("%d", numb_of_processor());
        }

    }

}

#endif
