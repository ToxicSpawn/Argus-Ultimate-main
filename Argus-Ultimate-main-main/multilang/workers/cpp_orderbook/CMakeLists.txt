cmake_minimum_required(VERSION 3.14)
project(argus_cpp_orderbook LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(cpp_orderbook orderbook.cpp)

# Install to same directory as source for bridge.py to find
install(TARGETS cpp_orderbook DESTINATION ${CMAKE_SOURCE_DIR})
