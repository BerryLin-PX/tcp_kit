# 安装
确保你将 tcp_kit 运行在 macOS 或 Linux 上
## 克隆仓库
```shell
git clone https://github.com/BerryLin-PX/tcp_kit.git
cd tcp_kit
```
## 安装依赖库
- CentOS
   ```shell
   sudo yum install -y libevent-devel protobuf-devel
   ```
- macOS
   ```shell
   brew install libevent protobuf
   ```
#### 构建 tcp_kit
```shell
mkdir build && cd build
cmake ..
sudo make install
```