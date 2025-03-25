# tcp_kit
tcp_kit 是一款轻量级 C++ 网络协议开发套件，利用它为你构建高效的网络服务。 其内部使用 libevent 驱动网络 IO 事件，Google Protocol Buffers 作为序列化工具。

## 环境
- tcp_kit 目前在 macOS 及 Linux 中支持。
- 编译器：GCC 4.8.5+
- 第三方库: libevent 2.1+, Protobuf 3.0+

## 示例
使用 tcp_kit 构建一个简单的 echo 服务
```cpp
#include <string>
#include <tcp_kit/server.h>
#include <tcp_kit/generic.h>

int main() {
    using namespace tcp_kit;
    server<generic> svr;
    svr.api("echo", [](std::string msg) {
        return msg;
    });
    svr.start();
}
```

## 如何开始?
1. 下载并安装 tcp_kit
2. 阅读概述

