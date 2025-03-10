#pragma once

#include <util/func_traits.h>
#include <logger/logger.h>

namespace tcp_kit {

    namespace func_traits_test {

        void index_seq_test() {
            auto seq = make_index_seq<std::tuple<int, double, bool, std::string>>();
            using t = decltype(seq);
            static_assert(std::is_same<index_seq<0, 1, 2, 3>, t>::value);
            log_info(typeid(t).name());
        }

    }

}