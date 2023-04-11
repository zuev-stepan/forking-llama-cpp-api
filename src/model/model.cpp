#include "model/model.h"

#include <cassert>
#include <iostream>

namespace llama_cpp_api
{

bool Model::init(const std::string& prompt)
{
    if (m_isBusy)
    {
        return false;
    }

    m_isBusy = true;
    m_isInitialized = true; // not actually, but will be busy until init is done

    assert(!m_pThread);
    m_pThread = std::make_unique<std::thread>([this, prompt]()
    {
        initImpl(prompt);
    });
    return true;
}

bool Model::processUserInput(const std::string& input)
{
    if (m_isBusy || !m_isInitialized)
    {
        return false;
    }

    m_isBusy = true;

    assert(m_pThread);
    m_pThread->join();
    m_pThread = std::make_unique<std::thread>([this, input]()
    {
        processUserInputImpl(input);
    });
    return true;
}

void Model::subscribe(ModelSubscriber* pSubscriber)
{
    assert(!m_isBusy && "This is not thread-safe");

    m_pSubscriber = pSubscriber;
}

bool Model::isBusy()
{
    return m_isBusy;
}

bool Model::isInitialized()
{
    return m_isInitialized;
}

void Model::update(const std::string& output)
{
    if (m_pSubscriber)
    {
        m_pSubscriber->update(output);
    }
}

void Model::done()
{
    m_isBusy = false;

    if (m_pSubscriber)
    {
        m_pSubscriber->done();
    }
}

}
