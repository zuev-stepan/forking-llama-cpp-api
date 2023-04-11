#pragma once

#ifndef LLAMA_CPP_API_MODEL_MODEL_H
#define LLAMA_CPP_API_MODEL_MODEL_H

#include <string>
#include <memory>
#include <atomic>
#include <thread>

#include "model/subscriber.h"

namespace llama_cpp_api
{

class Model
{
public:
    virtual ~Model() = default;

    bool init(const std::string& prompt);
    bool processUserInput(const std::string& input);
    virtual void stop() = 0;

    void subscribe(ModelSubscriber* pSubscriber);
    bool isBusy();
    bool isInitialized();

protected:
    void update(const std::string& output);
    void done();

    virtual void initImpl(const std::string& input) = 0;
    virtual void processUserInputImpl(const std::string& input) = 0;

private:
    ModelSubscriber* m_pSubscriber;

    std::unique_ptr<std::thread> m_pThread;
    std::atomic<bool> m_isBusy = false, m_isInitialized = false;
};

}

#endif // LLAMA_CPP_API_MODEL_MODEL_H
