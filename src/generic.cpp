#include <network/generic.h>
#include <error/errors.h>
#include <network/server.h>


namespace tcp_kit {

    void generic::handler::init(server_base* server_ptr) {

    }

    void generic::handler::run() {
         while(_server_base->is_running()) {
             msg_context* ctx = pop();
             try {
                 auto res = _filters->process(ctx, make_msg_buffer(ctx->in, ctx->in_len));
                 std::swap(ctx->out, res->ptr);
                 std::swap(ctx->out_len, res->size);
                 ctx->done();
             } catch (const std::exception& err) {
                 log_error(err.what());
                 ctx->error();
             }
         }
    }

    msg_context* generic::handler::pop() {
        std::unique_ptr<msg_context*> ptr_ptr = msg_queue->pop();
        return *(ptr_ptr.get());
    }

    std::unique_ptr<GenericMsg> generic::protobuf_deserializer::process(msg_context *ctx,
                                                                        std::unique_ptr<msg_buffer> input) {
        std::unique_ptr<GenericMsg> msg(new GenericMsg);
        if(msg->ParseFromArray(input->ptr, input->size)) {
            return msg;
        } else {
            throw generic_error<ILLEGALITY_ARGS>("Unable to parse the message to GenericMsg");
        }
    }

    std::unique_ptr<msg_buffer> generic::protobuf_serializer::process(msg_context *ctx,
                                                                      std::unique_ptr<GenericReply> reply) {
        size_t reply_size = reply->ByteSizeLong();
        std::unique_ptr<msg_buffer> buffer(new msg_buffer(reply_size + 2));
        if(reply->SerializeToArray(buffer->ptr, reply_size)) {
            buffer->ptr[reply_size] = '\r';
            buffer->ptr[reply_size + 1] = '\n';
            return buffer;
        } else {
            throw generic_error<SERIALIZE_MSG_ERROR>("Failed to serialize GenericMsg to array");
        }
    }

}