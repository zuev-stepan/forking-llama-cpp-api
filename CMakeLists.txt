cmake_minimum_required(VERSION 3.17.5)

# Project
project(llama_cpp_api)

add_executable(${PROJECT_NAME}
        src/model/llama.cpp
        src/model/message_sender.cpp
        src/model/model.cpp
        src/model/printer.cpp
        src/process/model_runner.cpp
        src/main.cpp
)

include_directories(${PROJECT_NAME} PRIVATE src)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)


# libs
include_directories(${PROJECT_NAME} PRIVATE lib)


# cpp-ipc
add_subdirectory(lib/cpp-ipc)
include_directories(${PROJECT_NAME} PRIVATE lib/cpp-ipc/include)
target_link_libraries(${PROJECT_NAME} ipc)


# PolyM
add_subdirectory(lib/PolyM)
include_directories(${PROJECT_NAME} PRIVATE lib/PolyM/include)
target_link_libraries(${PROJECT_NAME} polym)


# llama.cpp
add_subdirectory(lib/llama.cpp)
add_subdirectory(lib/llama.cpp/examples)

target_link_libraries(${PROJECT_NAME} llama common)