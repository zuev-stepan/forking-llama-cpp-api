#pragma once

#ifndef LLAMA_CPP_API_MESSAGES_COMMON_H
#define LLAMA_CPP_API_MESSAGES_COMMON_H

#include <cstring>

#include "libipc/ipc.h"

#include "messages/buffer.h"

namespace llama_cpp_api
{

inline int& sender_id_in_buffer(void* buffer)
{
    return reinterpret_cast<int*>(buffer)[0];
}

inline int sender_id_in_buffer(const void* buffer)
{
    return reinterpret_cast<const int*>(buffer)[0];
}

inline int& message_id_in_buffer(void* buffer)
{
    return reinterpret_cast<int*>(buffer)[1];
}

inline int message_id_in_buffer(const void* buffer)
{
    return reinterpret_cast<const int*>(buffer)[1];
}

template <typename T = char>
inline T* message_data_in_buffer(void* buffer)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + sizeof(int) * 2);
}

template <typename T = char>
inline const T* message_data_in_buffer(const void* buffer)
{
    return reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + sizeof(int) * 2);
}

inline size_t calc_data_size_from_message_size(size_t size)
{
    return size - sizeof(int) * 2;
}

inline size_t calc_message_size_from_data_size(size_t size)
{
    return size + sizeof(int) * 2;
}

inline void fill_message_header(int processId, int messageId, MessageBuffer& rBuffer)
{
    sender_id_in_buffer(rBuffer.get()) = processId;
    message_id_in_buffer(rBuffer.get()) = messageId;
}

template <uint16_t kMessageId>
struct DataBufferMessage
{
    int senderId;
    const char* data;
    size_t size;

    static DataBufferMessage receive(const void* data, size_t size)
    {
        DataBufferMessage res;

        res.senderId = sender_id_in_buffer(data);
        res.data = message_data_in_buffer(data);
        res.size = calc_data_size_from_message_size(size);

        return res;
    }

    void send(ipc::channel& rChannel, MessageBuffer& rBuffer)
    {
        auto messageSize = calc_message_size_from_data_size(size);
        rBuffer.reserve(messageSize);

        fill_message_header(senderId, kMessageId, rBuffer);
        std::memcpy(message_data_in_buffer(rBuffer.get()), data, size);

        rChannel.wait_for_recv(1);
        rChannel.send(rBuffer.get(), messageSize);
    }

    void send(const std::string& channelName, MessageBuffer& rBuffer)
    {
        ipc::channel channel(channelName.c_str(), ipc::sender);
        send(channel, rBuffer);
    }
};

template <uint16_t kMessageId, typename T>
struct ValueMessage
{
    int senderId;
    const T* pValue;

    static ValueMessage receive(const void* data, size_t size)
    {
        ValueMessage res;

        res.senderId = sender_id_in_buffer(data);
        res.pValue = reinterpret_cast<const T*>(message_data_in_buffer(data));

        return res;
    }

    void send(ipc::channel& rChannel, MessageBuffer& rBuffer)
    {
        auto messageSize = calc_message_size_from_data_size(sizeof(T));
        rBuffer.reserve(messageSize);

        fill_message_header(senderId, kMessageId, rBuffer);
        std::memcpy(message_data_in_buffer(rBuffer.get()), pValue, sizeof(T));

        rChannel.wait_for_recv(1);
        rChannel.send(rBuffer.get(), messageSize);
    }

    void send(const std::string& channelName, MessageBuffer& rBuffer)
    {
        ipc::channel channel(channelName.c_str(), ipc::sender);
        send(channel, rBuffer);
    }
};

template <uint16_t kMessageId>
struct EmptyMessage
{
    int senderId;

    static EmptyMessage receive(const void* data, size_t size)
    {
        EmptyMessage res;

        res.senderId = sender_id_in_buffer(data);

        return res;
    }

    void send(ipc::channel& rChannel, MessageBuffer& rBuffer)
    {
        auto messageSize = calc_message_size_from_data_size(0);
        rBuffer.reserve(messageSize);

        fill_message_header(senderId, kMessageId, rBuffer);

        rChannel.wait_for_recv(1);
        rChannel.send(rBuffer.get(), messageSize);
    }

    void send(const std::string& channelName, MessageBuffer& rBuffer)
    {
        ipc::channel channel(channelName.c_str(), ipc::sender);
        send(channel, rBuffer);
    }
};

}

#endif // LLAMA_CPP_API_MESSAGES_COMMON_H
