cmake_minimum_required(VERSION 2.8)
project(tcp_kit)

set(CMAKE_CXX_STANDARD 11)

set(LIBEVENT_LIB /usr/local/Cellar/libevent/2.1.12_1)
find_library(LIBEVENT_LIB /usr/local/Cellar/libevent/2.1.12_1)

set(GTEST_LIB /usr/local/opt/googletest)
find_package(GTest REQUIRED)
#set(GTEST_LIBRARIES gtest pthread)

file(GLOB_RECURSE ALL_FILES
        "${CMAKE_SOURCE_DIR}/src/*.h"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.hpp")

add_executable(tcp_kit ${ALL_FILES})

include_directories(src)
include_directories(/usr/local/Cellar/libevent/2.1.12_1/include)
include_directories(${GTEST_INCLUDE_DIRS})

target_link_libraries(tcp_kit ${LIBEVENT_LIB})
target_link_libraries(tcp_kit ${GTEST_LIBRARIES})
