#pragma once

#include <network/generic.h>

namespace tcp_kit {

    class json: public generic {
    public:
        class json_deserializer;
        class json_serializer;

        using filters = type_list<json_deserializer, api_dispatcher_p, json_serializer>;

        class json_deserializer {
        public:
            static std::unique_ptr<GenericMsg> process(msg_context* ctx, std::unique_ptr<msg_buffer> input);

        };

        class json_serializer {
        public:
            static std::unique_ptr<msg_buffer> process(msg_context* ctx, std::unique_ptr<GenericReply> input);

        };

    };



}