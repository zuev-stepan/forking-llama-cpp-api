#pragma once

#ifndef LLAMA_CPP_API_PROCESS_MODEL_RUNNER_H
#define LLAMA_CPP_API_PROCESS_MODEL_RUNNER_H

#include "messages/common.h"
#include "process/process.h"
#include "model/model.h"

using namespace std::chrono_literals;

namespace llama_cpp_api
{

enum ModelRunnerMessageId : int
{
    eForkRequest,
    eForkResponse,

    eKillRequest,
    eKillResponse,

    eInitRequest,
    eInitResponse,

    eReceiveInputRequest,
    eReceiveInputResponse,

    eStopModelRequest,
    eStopModelResponse,

    eReleaseOutputRequest,
    eReleaseOutputResponse,

    eNotifyWhenReadyRequest,
    eReady,
};

using ModelRunnerForkRequest = EmptyMessage<ModelRunnerMessageId::eForkRequest>;
using ModelRunnerForkResponse = ValueMessage<ModelRunnerMessageId::eForkResponse, int>;

using ModelRunnerKillRequest = EmptyMessage<ModelRunnerMessageId::eKillRequest>;
using ModelRunnerKillResponse = EmptyMessage<ModelRunnerMessageId::eKillResponse>;

using ModelRunnerInitRequest = DataBufferMessage<ModelRunnerMessageId::eInitRequest>;
using ModelRunnerInitResponse = DataBufferMessage<ModelRunnerMessageId::eInitResponse>;

using ModelRunnerReceiveInputRequest = DataBufferMessage<ModelRunnerMessageId::eReceiveInputRequest>;
using ModelRunnerReceiveInputResponse = DataBufferMessage<ModelRunnerMessageId::eReceiveInputResponse>;

using ModelRunnerStopModelRequest = EmptyMessage<ModelRunnerMessageId::eStopModelRequest>;
using ModelRunnerStopModelResponse = EmptyMessage<ModelRunnerMessageId::eStopModelResponse>;

using ModelRunnerReleaseOutputRequest = EmptyMessage<ModelRunnerMessageId::eReleaseOutputRequest>;
struct ModelRunnerReleaseOutputResponse : public DataBufferMessage<ModelRunnerMessageId::eReleaseOutputResponse>
{
    bool hasMore;

    static ModelRunnerReleaseOutputResponse receive(const void* data, size_t size)
    {
        ModelRunnerReleaseOutputResponse res;
        auto dataBufferMessage = DataBufferMessage::receive(data, size);

        res.senderId = dataBufferMessage.senderId;
        res.data = dataBufferMessage.data + 1;
        res.size = dataBufferMessage.size - 1;
        res.hasMore = dataBufferMessage.data[0];

        return res;
    }

    void send(ipc::channel& rChannel, MessageBuffer& rBuffer)
    {
        auto messageSize = calc_message_size_from_data_size(size + 1);
        rBuffer.reserve(messageSize);

        fill_message_header(senderId, ModelRunnerMessageId::eReleaseOutputResponse, rBuffer);
        message_data_in_buffer(rBuffer.get())[0] = hasMore;
        std::memcpy(message_data_in_buffer(rBuffer.get()) + 1, data, size);

        rChannel.wait_for_recv(1);
        rChannel.send(rBuffer.get(), messageSize);
    }

    void send(const std::string& channelName, MessageBuffer& rBuffer)
    {
        ipc::channel channel(channelName.c_str(), ipc::sender);
        send(channel, rBuffer);
    }
};

using ModelRunnerNotifyWhenReadyRequest = EmptyMessage<ModelRunnerMessageId::eNotifyWhenReadyRequest>;
using ModelRunnerDone = EmptyMessage<ModelRunnerMessageId::eReady>;

std::unique_ptr<Process> make_model_runner(int processId, uint64_t timeoutMs, std::unique_ptr<Model> pModel);

}

#endif // LLAMA_CPP_API_PROCESS_MODEL_RUNNER_H
