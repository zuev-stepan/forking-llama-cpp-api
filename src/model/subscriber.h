#pragma once

#ifndef LLAMA_CPP_API_MODEL_SUBSCRIBER_H
#define LLAMA_CPP_API_MODEL_SUBSCRIBER_H

#include <string>

namespace llama_cpp_api
{

class ModelSubscriber
{
public:
    virtual ~ModelSubscriber() = default;

    virtual void update(const std::string& output) = 0;
    virtual void done() = 0;
};

}

#endif // LLAMA_CPP_API_MODEL_SUBSCRIBER_H
