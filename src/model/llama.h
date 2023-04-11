#pragma once

#ifndef LLAMA_CPP_API_MODEL_LLAMA_H
#define LLAMA_CPP_API_MODEL_LLAMA_H

#include <memory>

#include "llama.cpp/examples/common.h"

#include "model/model.h"

namespace llama_cpp_api
{

std::unique_ptr<Model> create_llama_model(const gpt_params& params, const char* inputPrefix, const char* outputPrefix);

}

#endif // LLAMA_CPP_API_MODEL_LLAMA_H
