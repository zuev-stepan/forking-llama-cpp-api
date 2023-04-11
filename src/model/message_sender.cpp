#include "model/message_sender.h"

#include <cassert>

#include "polym/Queue.hpp"

namespace llama_cpp_api
{

class ModelMessageSender : public ModelSubscriber
{
public:
    ModelMessageSender(PolyM::Queue* pQueue)
        : m_pQueue(pQueue)
    {
        assert(m_pQueue);
    }

    void update(const std::string& output) override
    {
        m_pQueue->put(PolyM::DataMsg<std::string>(ModelMessageId::eUpdate, output));
    }

    void done() override
    {
        m_pQueue->put(PolyM::Msg(ModelMessageId::eDone));
    }

private:
    PolyM::Queue* m_pQueue;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<ModelSubscriber> create_model_message_sender(PolyM::Queue* pQueue)
{
    return std::make_unique<ModelMessageSender>(pQueue);
}

}
