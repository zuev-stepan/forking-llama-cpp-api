#include "model/llama.h"

#include <random>
#include <stdexcept>
#include <iostream>

#include "llama.cpp/llama.h"

namespace llama_cpp_api
{

struct LlamaModelContext
{
    llama_context* ctx;

    std::vector<llama_token> embd_inp;

    std::vector<llama_token> inp_pfx, inp_sfx, llama_token_newline;
    std::vector<llama_token> last_n_tokens;

    std::vector<llama_token> embd;

    int n_ctx;
    int n_past;
    int n_remain;
    int n_consumed;

    bool input_noecho;
    bool is_antiprompt;
    bool waiting_input;

    std::atomic<bool> is_interacting;
};

static void load_llama_model(gpt_params& params, llama_context*& ctx)
{
    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        throw std::runtime_error("perplexity param not supported");
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        throw std::runtime_error("embedding param not supported");
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mlock  = params.use_mlock;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());

            throw std::runtime_error("failed to load model");
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        throw std::runtime_error("mem_test param not supported");
    }
}

static void init_llama_model(gpt_params& params, const char* input_prefix, const char* output_prefix,
                             llama_context*& ctx, std::vector<llama_token>& inp_pfx, std::vector<llama_token>& inp_sfx,
                             std::vector<llama_token>& embd_inp, std::vector<llama_token>& embd,
                             std::vector<llama_token>& last_n_tokens, std::vector<llama_token>& llama_token_newline,
                             int& n_remain, int& n_past, int& n_ctx, int& n_consumed, std::atomic<bool>& is_interacting,
                             bool& input_noecho, bool& is_antiprompt, bool& waiting_input)
{
    std::cout << params.prompt << std::endl;
    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        throw std::runtime_error("prompt is too long");
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    inp_pfx = ::llama_tokenize(ctx, input_prefix, true);
    inp_sfx = ::llama_tokenize(ctx, output_prefix, false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_start = true;
        params.antiprompt.push_back(input_prefix);
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.interactive_start) {
        params.interactive = true;
    }

    // determine newline token
    llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        if (params.n_keep > 0) {
        fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    if (params.interactive) {
        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (!params.input_prefix.empty()) {
            fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
        }
    }
    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    last_n_tokens.resize(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = params.interactive_start;
    }

    is_antiprompt = false;
    input_noecho  = false;

    n_past     = 0;
    n_remain   = params.n_predict;
    n_consumed = 0;

    embd.clear();
    waiting_input = false;
}

static void init_llama_model(gpt_params& params, const char* inputPrefix, const char* outputPrefix,
                             LlamaModelContext& context)
{
    init_llama_model(params, inputPrefix, outputPrefix, context.ctx, context.inp_pfx, context.inp_sfx, context.embd_inp,
                     context.embd, context.last_n_tokens, context.llama_token_newline, context.n_remain, context.n_past,
                     context.n_ctx, context.n_consumed, context.is_interacting, context.input_noecho,
                     context.is_antiprompt, context.waiting_input);
}

template <typename UpdateFunction>
void run_llama_model(const gpt_params& params, llama_context* ctx, std::vector<llama_token>& inp_pfx,
                     std::vector<llama_token>& inp_sfx, std::vector<llama_token>& embd_inp,
                     std::vector<llama_token>& embd, std::vector<llama_token>& last_n_tokens,
                     std::vector<llama_token>& llama_token_newline, int& n_remain, int& n_past, int& n_ctx,
                     int& n_consumed, std::atomic<bool>& is_interacting, bool& input_noecho, bool& is_antiprompt,
                     bool& waiting_input, const std::string& input, UpdateFunction update)
{
    while (waiting_input || n_remain != 0 || params.interactive) {
        if (!waiting_input) {
            // predict
            if (embd.size() > 0) {
                // infinite text generation via context swapping
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (n_past + (int) embd.size() > n_ctx) {
                    const int n_left = n_past - params.n_keep;

                    n_past = params.n_keep;

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                    //printf("\n---\n");
                    //printf("resetting: '");
                    //for (int i = 0; i < (int) embd.size(); i++) {
                    //    printf("%s", llama_token_to_str(ctx, embd[i]));
                    //}
                    //printf("'\n");
                    //printf("\n---\n");
                }

                if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    throw std::runtime_error("failed to eval");
                }
            }

            n_past += embd.size();
            embd.clear();

            if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
                // out of user input, sample next token
                const int32_t top_k          = params.top_k;
                const float   top_p          = params.top_p;
                const float   temp           = params.temp;
                const float   repeat_penalty = params.repeat_penalty;

                llama_token id = 0;

                {
                    auto logits = llama_get_logits(ctx);

                    if (params.ignore_eos) {
                        logits[llama_token_eos()] = 0;
                    }

                    id = llama_sample_top_p_top_k(ctx,
                            last_n_tokens.data() + n_ctx - params.repeat_last_n,
                            params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // replace end of text token with newline token when in interactive mode
                if (id == llama_token_eos() && params.interactive && !params.instruct) {
                    id = llama_token_newline.front();
                    if (params.antiprompt.size() != 0) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                    }
                }

                // add it to the context
                embd.push_back(id);

                // echo this to console
                input_noecho = false;

                // decrement remaining sampling budget
                --n_remain;
            } else {
                // some user input remains from prompt or interaction, forward it to processing
                while ((int) embd_inp.size() > n_consumed) {
                    embd.push_back(embd_inp[n_consumed]);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[n_consumed]);
                    ++n_consumed;
                    if ((int) embd.size() >= params.n_batch) {
                        break;
                    }
                }
            }

            // display text
            if (!input_noecho) {
                for (auto id : embd) {
                    update(llama_token_to_str(ctx, id));
                }
            }
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (waiting_input || (params.interactive && (int) embd_inp.size() <= n_consumed)) {
            if (!waiting_input) {
                // check for reverse prompt
                if (params.antiprompt.size()) {
                    std::string last_output;
                    for (auto id : last_n_tokens) {
                        last_output += llama_token_to_str(ctx, id);
                    }

                    is_antiprompt = false;
                    // Check if each of the reverse prompts appears at the end of the output.
                    for (const auto& antiprompt : params.antiprompt) {
                        if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                            is_interacting = true;
                            is_antiprompt = true;
                            break;
                        }
                    }
                }
            }

            if (waiting_input || (n_past > 0 && is_interacting)) {
                if (!waiting_input) {
                    waiting_input = true;
                    return;
                }

                waiting_input = false;

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                }

                buffer += input;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size();
                }

                input_noecho = true; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
}

template <typename UpdateFunction>
void run_llama_model(const gpt_params& params, LlamaModelContext& context, const std::string& input,
                     UpdateFunction update)
{
    run_llama_model(params, context.ctx, context.inp_pfx, context.inp_sfx, context.embd_inp, context.embd,
                    context.last_n_tokens, context.llama_token_newline, context.n_remain, context.n_past, context.n_ctx,
                    context.n_consumed, context.is_interacting, context.input_noecho, context.is_antiprompt,
                    context.waiting_input, input, update);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LlamaModel final : public Model
{
public:
    LlamaModel(const gpt_params& params, std::string inputPrefix, std::string outputPrefix)
        : m_params(params), m_inputPrefix(std::move(inputPrefix)), m_outputPrefix(std::move(outputPrefix))
    {
        load_llama_model(m_params, m_context.ctx);
    }

    ~LlamaModel()
    {
        llama_free(m_context.ctx);
    }

    void stop() override
    {
        m_context.is_interacting = true;
    }

protected:
    void initImpl(const std::string& prompt) override
    {
        m_params.prompt = prompt;
        init_llama_model(m_params, m_inputPrefix.c_str(), m_outputPrefix.c_str(), m_context);
        run_llama_model(m_params, m_context, "", [](auto){});
        done();
    }

    void processUserInputImpl(const std::string& input) override
    {
        run_llama_model(m_params, m_context, input, [&](const std::string& output)
        {
            update(output);
        });
        done();
    }

private:
    gpt_params m_params;
    std::string m_inputPrefix, m_outputPrefix;
    LlamaModelContext m_context;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Model> create_llama_model(const gpt_params& params, const char* inputPrefix, const char* outputPrefix)
{
    return std::make_unique<LlamaModel>(params, inputPrefix, outputPrefix);
}

}
