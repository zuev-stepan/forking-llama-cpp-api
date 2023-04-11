#pragma once

#ifndef LLAMA_CPP_API_JSON_H
#define LLAMA_CPP_API_JSON_H

#include <sstream>

namespace llama_cpp_api
{

template <typename T>
void print_json_value(std::ostream& stream, const std::string& name, const T& value)
{
    stream << "  \"" << name << "\": ";
    if constexpr(std::is_same<typename std::decay<decltype(value)>::type, std::string>::value)
    {
        stream << "\"";
        for (auto c : value)
        {
            if (c == '\n')
            {
                stream << "\\n";
            }
            else if (c == '"')
            {
                stream << "\\\"";
            }
            else
            {
                stream << c;
            }
        }
        stream << "\"";
    }
    else
    {
        stream << value;
    }
}

template <typename T>
void print_json_values(std::ostream& stream, const std::string& name, const T& value)
{
    print_json_value(stream, name, value);
    stream << "\n";
}

template <typename T, typename... Args>
void print_json_values(std::ostream& stream, const std::string& name, const T& value, Args&&... args)
{
    print_json_value(stream, name, value);
    stream << ",\n";

    print_json_values(stream, std::forward<Args>(args)...);
}

template <typename... Args>
std::string get_json(Args&&... args)
{
    std::ostringstream stream;
    stream << "{\n";
    print_json_values(stream, std::forward<Args>(args)...);
    stream << "}\n";

    return stream.str();
}

}

#endif // LLAMA_CPP_API_JSON_H
