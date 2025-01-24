#include <network/generic.h>
#include <error/errors.h>
#include <network/server.h>


namespace tcp_kit {

    void generic::handler::init(server_base* server_ptr) {

    }

    void generic::handler::run() {
         while(_server_base->is_running()) {
             msg_context* ctx = pop();
             log_debug("Start to process the message");
             auto res = _filters->process(ctx, make_msg_buffer(ctx->in, ctx->in_len));
             ctx->out = res->ptr;
             ctx->out_len = res->size;
             ctx->done();
         }
    }

    msg_context* generic::handler::pop() {
        std::unique_ptr<msg_context*> ptr_ptr = msg_queue->pop();
        return *(ptr_ptr.get());
    }


}