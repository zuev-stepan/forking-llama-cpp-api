#include "process/model_runner.h"

#include <cassert>
#include <string>
#include <thread>
#include <memory>
#include <iostream>
#include <unistd.h>

#include "polym/Queue.hpp"

#include "model/model.h"
#include "model/message_sender.h"

using namespace std::chrono_literals;

namespace llama_cpp_api
{

class ModelRunner final : public Process
{
public:
    ModelRunner(int processId, uint64_t timeoutMs, std::unique_ptr<Model> pModel)
        : Process(processId, timeoutMs), m_pModel(std::move(pModel)),
        m_pMessageSender(create_model_message_sender(&m_queue))
    {
        m_pModel->subscribe(m_pMessageSender.get());
    }

private:
    std::unique_ptr<Process> handleMessage(const void* data, size_t size) override
    {
        if (size < calc_message_size_from_data_size(0))
        {
            handleMessageFromModel(m_queue.get(getTimeout()));
            return nullptr;
        }

        auto senderId = sender_id_in_buffer(data);
        auto messageId = message_id_in_buffer(data);

        switch (messageId)
        {
        case ModelRunnerMessageId::eForkRequest:
        {
            int pid = -1;
            if (!isBusy())
            {
                pid = fork();
                if (pid == 0)
                {
                    return make_model_runner(getpid(), getTimeout(), std::move(m_pModel));
                }
            }

            ModelRunnerForkResponse response{getProcessId(), &pid};
            response.send(get_channel_name(senderId), getBuffer());
            break;
        }
        case ModelRunnerMessageId::eKillRequest:
        {
            stopModel();
            ModelRunnerKillResponse response{getProcessId()};
            response.send(get_channel_name(senderId), getBuffer());
            return nullptr;
        }
        case ModelRunnerMessageId::eInitRequest:
        {
            auto message = ModelRunnerInitRequest::receive(data, size);

            auto result = init(message.data);
            ModelRunnerInitResponse response{getProcessId(), result.data(), result.size()};
            response.send(get_channel_name(senderId), getBuffer());
            break;
        }
        case ModelRunnerMessageId::eReceiveInputRequest:
        {
            auto message = ModelRunnerReceiveInputRequest::receive(data, size);

            auto result = receiveInput(message.data);
            ModelRunnerReceiveInputResponse response{getProcessId(), result.data(), result.size()};
            response.send(get_channel_name(senderId), getBuffer());
            break;
        }
        case ModelRunnerMessageId::eStopModelRequest:
        {
            stopModel();
            ModelRunnerStopModelResponse response{getProcessId()};
            response.send(get_channel_name(senderId), getBuffer());
            break;
        }
        case ModelRunnerMessageId::eReleaseOutputRequest:
        {
            auto result = releaseModelOutput();
            ModelRunnerReleaseOutputResponse response{getProcessId(), result.c_str(), result.size(), isBusy()};
            response.send(get_channel_name(senderId), getBuffer());
            break;
        }
        case ModelRunnerMessageId::eNotifyWhenReadyRequest:
        {
            if (!m_pModel->isBusy())
            {
                ModelRunnerDone response{getProcessId()};
                response.send(get_channel_name(senderId).c_str(), getBuffer());
            }
            else
            {
                m_notify.emplace_back(senderId);
            }
            break;
        }
        }

        handleMessageFromModel(m_queue.get(getTimeout()));
        return nullptr;
    }

    void handleMessageFromModel(std::unique_ptr<PolyM::Msg> msg)
    {
        if (!msg)
        {
            return;
        }

        switch (msg->getMsgId())
        {
        case ModelMessageId::eUpdate:
        {
            receiveModelOutput(static_cast<PolyM::DataMsg<std::string>*>(msg.get())->getPayload());
            break;
        }
        case ModelMessageId::eDone:
        {
            assert(!m_pModel->isBusy());

            for (auto senderId : m_notify)
            {
                ModelRunnerDone message{getProcessId()};
                message.send(get_channel_name(senderId).c_str(), getBuffer());
            }
            m_notify.clear();
            break;
        }
        }
    }

    std::string init(const char* prompt)
    {
        if (m_pModel->isBusy())
        {
            return "Error: Model is busy";
        }
        if (m_pModel->isInitialized())
        {
            return "Error: Already initialized";
        }
        if (!m_pModel->init(prompt))
        {
            return "Error: Unknown error";
        }

        return "Success";
    }

    std::string receiveInput(const char* input)
    {
        if (!m_modelOutput.empty())
        {
            return "Error: Read pending output first";
        }
        if (m_pModel->isBusy())
        {
            return "Error: Model is busy";
        }
        if (!m_pModel->processUserInput(input))
        {
            return "Error: Unknown error";
        }

        return "Success";
    }

    void receiveModelOutput(const std::string& output)
    {
        m_modelOutput += output;
    }

    std::string releaseModelOutput()
    {
        return std::move(m_modelOutput);
    }

    bool isBusy()
    {
        return m_pModel->isBusy();
    }

    void stopModel()
    {
        while (m_pModel->isBusy())
        {
            m_pModel->stop();
            std::this_thread::sleep_for(100ms);
        }
    }

private:
    std::unique_ptr<Model> m_pModel;
    PolyM::Queue m_queue;
    std::unique_ptr<ModelSubscriber> m_pMessageSender;
    std::string m_modelOutput;

    std::vector<int> m_notify;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Process> make_model_runner(int processId, uint64_t timeoutMs, std::unique_ptr<Model> pModel)
{
    return std::make_unique<ModelRunner>(processId, timeoutMs, std::move(pModel));
}

}
