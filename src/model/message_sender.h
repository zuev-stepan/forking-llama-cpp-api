#pragma once

#ifndef LLAMA_CPP_API_MODEL_MESSAGE_SENDER_H
#define LLAMA_CPP_API_MODEL_MESSAGE_SENDER_H

#include <memory>

#include "model/subscriber.h"

namespace PolyM
{

class Queue;

}

namespace llama_cpp_api
{

enum ModelMessageId : char
{
    eUpdate = 1,
    eDone = 2,
};

std::unique_ptr<ModelSubscriber> create_model_message_sender(PolyM::Queue* pQueue);

}

#endif // LLAMA_CPP_API_MODEL_MESSAGE_SENDER_H
