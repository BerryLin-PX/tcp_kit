#include <network/generic.h>
#include <error/errors.h>
#include <network/server.h>

#define SUCCESSFUL 0 // libevent API 表示成功的值

namespace tcp_kit {

    void generic::handler::init(server_base* server_ptr) {

    }

    void generic::handler::run() {
        while(_server_base->is_running()) {
            msg_context* ctx = pop();
        }
    }

    msg_context* generic::handler::pop() {
        std::unique_ptr<msg_context*> ptr_ptr = msg_queue->pop();
        return *(ptr_ptr.get());
    }


}