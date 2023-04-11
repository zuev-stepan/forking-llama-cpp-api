#include <iostream>
#include <set>
#include <sstream>
#include <mutex>

#include "libipc/ipc.h"
#include "cpp-httplib/httplib.h"

#include "json.h"
#include "model/llama.h"
#include "model/printer.h"
#include "process/model_runner.h"

using namespace llama_cpp_api;
using namespace std::chrono_literals;


int main(int argc, char** argv)
{
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params))
    {
        return 0;
    }

    auto pModel = create_llama_model(params, "\n\n### Instruction:\n\n", "\n\n### Response:\n\n");

    std::unordered_map<std::thread::id, int> serverThreadId;
    std::mutex serverThreadIdMutex;
    auto getServerThreadId = [&]()
    {
        auto threadId = std::this_thread::get_id();

        std::lock_guard<std::mutex> lock(serverThreadIdMutex);
        if (serverThreadId.find(threadId) == serverThreadId.end())
        {
            serverThreadId[threadId] = -int(serverThreadId.size()) - 1;
        }

        return serverThreadId.at(threadId);
    };

    std::set<int> chatIds;
    struct FindChatIdResult
    {
        int id;
        bool success;
        std::string message;
    };
    auto findChatId = [&](const std::string& str)
    {
        int id;
        try
        {
            id = std::stoi(str);
        }
        catch (const std::exception& e)
        {
            return FindChatIdResult{0, false, e.what()};
        }

        if (chatIds.find(id) == chatIds.end())
        {
            return FindChatIdResult{0, false, "Chat not found"};
        }

        return FindChatIdResult{id, true, ""};
    };

    MessageBuffer messageBuffer;

    httplib::Server server;

    /// Returns a list of current chat ids
    server.Get("/chats", [&](const httplib::Request& req, httplib::Response& res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        std::ostringstream str;
        str << "{\n";
        str << "  \"ids\": [\n";
        if (!chatIds.empty())
        {
            str << "    " << *chatIds.begin();
            for (auto it = ++chatIds.begin(); it != chatIds.end(); ++it)
            {
                str << ",\n";
                str << "    " << *it;
            }
        }
        str << "\n  ]\n";
        str << "}\n";

        res.set_content(str.str(), "application/json");
    });

    /// Fork existing chat and returns new chat id
    server.Post("/fork/([0-9]+)", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        auto request = ModelRunnerForkRequest{senderId};
        request.send(outputChannel, messageBuffer);
        auto buf = inputChannel.recv();
        auto response = ModelRunnerForkResponse::receive(buf.data(), buf.size());
        if (*response.pValue < 0)
        {
            res.set_content(get_json("error", std::string("Fork failed, model might be busy")), "application/json");
        }
        else
        {
            chatIds.emplace(*response.pValue);
            res.set_content(get_json("id", *response.pValue), "application/json");
        }
    });

    /// Delete chat
    server.Post("/delete/(\\d+)", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        auto request = ModelRunnerKillRequest{senderId};
        request.send(outputChannel, messageBuffer);
        auto buf = inputChannel.recv();

        chatIds.erase(chatId.id);
        res.set_content(get_json("deleted", chatId.id), "application/json");
    });

    /// Init chat with prompt
    server.Post("/init", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        // fork root
        int id = 0;
        {
            auto senderId = getServerThreadId();
            auto inputChannelName = get_channel_name(senderId);
            auto outputChannelName = get_channel_name(0);
            ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
            ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

            auto request = ModelRunnerForkRequest{senderId};
            request.send(outputChannel, messageBuffer);
            auto buf = inputChannel.recv();
            auto response = ModelRunnerForkResponse::receive(buf.data(), buf.size());

            id = *response.pValue;
            chatIds.emplace(id);
        }

        // init new chat
        {
            auto senderId = getServerThreadId();
            auto inputChannelName = get_channel_name(senderId);
            auto outputChannelName = get_channel_name(id);
            ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
            ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

            auto request = ModelRunnerInitRequest{senderId, req.body.data(), req.body.size()};
            request.send(outputChannel, messageBuffer);
            auto buf = inputChannel.recv();
            auto response = ModelRunnerInitResponse::receive(buf.data(), buf.size());
            if (response.data[0] != 'S') // Error
            {
                res.set_content(get_json("error", std::string(response.data, response.size)), "application/json");
            }
            else
            {
                res.set_content(get_json("id", id), "application/json");
            }
        }
    });

    /// Send message to chat
    server.Post("/send/(\\d+)", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        auto request = ModelRunnerReceiveInputRequest{senderId, req.body.data(), req.body.size()};
        request.send(outputChannel, messageBuffer);
        auto buf = inputChannel.recv();
        auto response = ModelRunnerReceiveInputResponse::receive(buf.data(), buf.size());

        if (response.data[0] != 'S') // Error
        {
            res.set_content(get_json("error", std::string(response.data, response.size)), "application/json");
        }
        else
        {
            res.set_content(get_json("sent", chatId.id), "application/json");
        }
    });

    /// Stop calculation in chat
    server.Post("/stop/(\\d+)", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        auto request = ModelRunnerStopModelRequest{senderId};
        request.send(outputChannel, messageBuffer);
        auto buf = inputChannel.recv();

        res.set_content(get_json("stopped", chatId.id), "application/json");
    });

    /// Get new text in chat
    server.Get("/update/(\\d+)", [&](const httplib::Request& req, httplib::Response &res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        auto request = ModelRunnerReleaseOutputRequest{senderId};
        request.send(outputChannel, messageBuffer);
        auto buf = inputChannel.recv();
        auto response = ModelRunnerReleaseOutputResponse::receive(buf.data(), buf.size());

        res.set_content(get_json("update", std::string(response.data, response.size), "finished", !response.hasMore),
                        "application/json");
    });

    /// Send message to chat, wait for response and return it
    server.Post("/interact/(\\d+)", [&](const httplib::Request& req, httplib::Response& res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto chatId = findChatId(req.matches[1]);
        if (!chatId.success)
        {
            res.set_content(get_json("error", chatId.message), "application/json");
            return;
        }

        auto senderId = getServerThreadId();
        auto inputChannelName = get_channel_name(senderId);
        auto outputChannelName = get_channel_name(chatId.id);
        ipc::channel inputChannel(inputChannelName.c_str(), ipc::receiver);
        ipc::channel outputChannel(outputChannelName.c_str(), ipc::sender);

        // send message
        {
            auto request = ModelRunnerReceiveInputRequest{senderId, req.body.data(), req.body.size()};
            request.send(outputChannel, messageBuffer);
            auto buf = inputChannel.recv();
            auto response = ModelRunnerReceiveInputResponse::receive(buf.data(), buf.size());
            if (response.data[0] != 'S') // Error
            {
                res.set_content(get_json("error", std::string(response.data, response.size)), "application/json");
                return;
            }
        }

        // wait until done
        {
            auto request = ModelRunnerNotifyWhenReadyRequest{senderId};
            request.send(outputChannel, messageBuffer);
            inputChannel.recv();
        }

        // get reply from model
        {
            auto request = ModelRunnerReleaseOutputRequest{senderId};
            request.send(outputChannel, messageBuffer);
            auto buf = inputChannel.recv();
            auto response = ModelRunnerReleaseOutputResponse::receive(buf.data(), buf.size());

            res.set_content(get_json("reply", std::string(response.data, response.size)), "application/json");
        }
    });

    server.set_exception_handler([](const auto& req, auto& res, std::exception_ptr ep)
    {
        res.set_header("Access-Control-Allow-Origin", "*");

        auto fmt = "<h1>Error 500</h1><p>%s</p>";
        char buf[BUFSIZ];
        try
        {
            std::rethrow_exception(ep);
        }
        catch (std::exception &e)
        {
            snprintf(buf, sizeof(buf), fmt, e.what());
        }
        catch (...)
        {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/html");
        res.status = 500;
    });

    auto pRunner = make_model_runner(0, 10, std::move(pModel));
    auto pid = fork();
    if (pid == 0)
    {
        while (pRunner)
        {
            pRunner = pRunner->loop();
        }

        return 0;
    }

    server.listen("0.0.0.0", 8880);

    return 0;
}