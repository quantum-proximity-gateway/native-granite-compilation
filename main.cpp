#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "llama.h"

// TODO:
// Fix GGML bindings

int main(int argc, char** argv) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model-path.gguf> \"<prompt text>\"\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    const std::string user     = argv[2];

    // 1. Set llama model and context parameters
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;          // Using 99 to utilize GPU acceleration, tested on Apple Silicon and
                                        // library correctly utilizes Metal API to take advantage of the chip.

    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.embeddings = false;
    ctx_params.n_ctx     = 2048;        // context size
    ctx_params.n_threads = 4;         // CPU threads for generation (adjust as desired)

    // 2. Load the model
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    // 3. Create a new context
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        fprintf(stderr, "Error: Failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }


    // Prepare prompt
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

        // add the user input to the message list and format it
        messages.push_back({"user", strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }

        // remove previous messages to obtain the prompt to generate the response
        std::string prompt(formatted.begin(), formatted.begin() + new_len);

    // 4. Tokenize prompt
    // Use the model’s vocab and call llama_tokenize
    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 2); // extra space for BOS, etc.
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        /* text_len       = */ (int32_t) prompt.size(),
        tokens.data(),
        /* n_tokens_max   = */ (int32_t) tokens.size(),
        /* add_special    = */ true,
        /* parse_special  = */ false
    );

    if (n_tokens < 0) {
        fprintf(stderr, "Error: Failed to tokenize prompt (returned %d)\n", n_tokens);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize(n_tokens);

    // 5. Generate tokens
   
    // initialize the sampler, sets temperature, distribution for model sampling (using values from llama.cpp repo)
    llama_sampler * smpl_chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Print the user prompt first
    printf("%s", prompt.c_str());

    int last_token_pos = n_tokens - 1; // Track the position of the last token
    
    std::string response;

    const bool is_first = llama_get_kv_cache_used_cells(ctx) == 0;

    // tokenize the prompt
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        //GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        if (llama_decode(ctx, batch)) {
            //GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(smpl_chain, ctx, -1);
        printf("%d", new_token_id);
        // check if token generated is end of generation token
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            //GGML_ABORT("failed to convert token to piece\n");
        }
        std::string piece(buf, n);
        printf("%s", piece.c_str());
        fflush(stdout);
        response += piece;

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    // 6. Clean up
    llama_sampler_free(smpl_chain); // also frees the individual samplers added to chain
    llama_free(ctx);
    llama_model_free(model);

    printf("\n");
    return 0;
}
