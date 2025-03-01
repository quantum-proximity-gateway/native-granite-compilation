# Granite Compiled Inference via llama.cpp
 - Marwan Yassini Chairi El Kamel
## Overview
This repository contains instructions and a simple script allowing for compiled inference of GGUF files utilising the API provided by llama.cpp

## Prerequisites
- CMake
- C++ Compiler (g++, clang)
- GGUF File for Granite (any other model works)
    https://huggingface.co/lmstudio-community/granite-3.2-8b-instruct-GGUF

## Installation
1. Clone this repository
   
2. Clone llama.cpp into the same folder
    ```bash
    git clone https://github.com/ggml-org/llama.cpp
    ```

3. Build the library
    ```bash
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release
    ```
4. Compile the program
    ```bash
    clang++ -std=c++11 -I./llama.cpp/include -I./llama.cpp/ggml/include main.cpp ./llama.cpp/build/bin/libllama.dylib -o gguf_infer -pthread -Wl,-rpath,./llama.cpp/build/bin;
    ```

5. Run the program with the following parameters
    ```bash
    ./gguf_infer <model-path.gguf> "<prompt text>"
    ```

