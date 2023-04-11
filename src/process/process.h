#pragma once

#ifndef LLAMA_CPP_API_PROCESS_PROCESS_H
#define LLAMA_CPP_API_PROCESS_PROCESS_H

#include <cassert>
#include <string>
#include <thread>
#include <memory>

#include "libipc/ipc.h"

#include "messages/buffer.h"

namespace llama_cpp_api
{

class Process
{
public:
    Process(int processId, uint64_t timeoutMs)
        : m_processId(processId), m_channel(get_channel_name(processId).c_str(), ipc::receiver), m_timeoutMs(timeoutMs)
    { }

    virtual ~Process() = default;

    std::unique_ptr<Process> loop()
    {
        while (true)
        {
            auto buffer = m_channel.recv(m_timeoutMs);
            auto newProcess = handleMessage(buffer.data(), buffer.size());
            if (newProcess)
            {
                return newProcess;
            }
        }
    }

protected:
    virtual std::unique_ptr<Process> handleMessage(const void* data, size_t size) = 0;

    int getProcessId() const
    {
        return m_processId;
    }

    MessageBuffer& getBuffer()
    {
        return m_buffer;
    }

    uint64_t getTimeout() const
    {
        return m_timeoutMs;
    }

private:
    int m_processId;
    ipc::channel m_channel;
    MessageBuffer m_buffer;

    uint64_t m_timeoutMs;
};

}

#endif // LLAMA_CPP_API_PROCESS_PROCESS_H
