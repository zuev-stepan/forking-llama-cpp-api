#pragma once

#ifndef LLAMA_CPP_API_MESSAGES_BUFFER_H
#define LLAMA_CPP_API_MESSAGES_BUFFER_H

#include "libipc/ipc.h"

namespace llama_cpp_api
{

class MessageBuffer
{
public:
    MessageBuffer()
        : m_buffer(256, 0)
    { }

    void reserve(size_t size)
    {
        if (m_buffer.size() < size)
        {
            m_buffer.resize(size);
        }
    }

    char* get()
    {
        return m_buffer.data();
    }

private:
    std::vector<char> m_buffer;
};

inline std::string get_channel_name(int pid)
{
    return "ipc" + std::to_string(pid);
}

}

#endif // LLAMA_CPP_API_MESSAGES_BUFFER_H
