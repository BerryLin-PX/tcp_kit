#include <network/json.h>
#include <google/protobuf/util/json_util.h>

namespace tcp_kit {

    std::unique_ptr<GenericMsg> json::json_deserializer::process(msg_context *ctx, std::unique_ptr<msg_buffer> input) {
        std::unique_ptr<GenericMsg> generic_msg(new GenericMsg);
        std::string json_str(input->ptr);
        google::protobuf::util::Status status = google::protobuf::util::JsonStringToMessage(json_str, generic_msg.get());
        return generic_msg;
    }

    std::unique_ptr<msg_buffer> json::json_serializer::process(msg_context *ctx, std::unique_ptr<GenericReply> reply) {
        log_debug("Output processor");
        std::string json_string;
        GenericReply reply_c(*reply);
        google::protobuf::util::MessageToJsonString(reply_c, &json_string);
        std::unique_ptr<msg_buffer> output(new msg_buffer(json_string.size() + 2));
        memcpy(output->ptr, json_string.c_str(), json_string.size());
        output->ptr[json_string.size()] = '\r';
        output->ptr[json_string.size() + 1] = '\n';
        return output;
    }

}
