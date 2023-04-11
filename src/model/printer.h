#pragma once

#ifndef LLAMA_CPP_API_MODEL_PRINTER_H
#define LLAMA_CPP_API_MODEL_PRINTER_H

#include <memory>

#include "model/subscriber.h"

namespace llama_cpp_api
{

std::unique_ptr<ModelSubscriber> create_model_printer();

}

#endif // LLAMA_CPP_API_MODEL_PRINTER_H
