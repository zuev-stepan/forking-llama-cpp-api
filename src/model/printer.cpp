#include "model/printer.h"

#include <iostream>

namespace llama_cpp_api
{

class ModelPrinter : public ModelSubscriber
{
public:
    void update(const std::string& output) override
    {
        std::cout << output << std::flush;
    }

    void done() override
    {
        // nop
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<ModelSubscriber> create_model_printer()
{
    return std::make_unique<ModelPrinter>();
}

}
