# 🌿 Daphne

## Non-AI Generated part

I started this repo to get a deeper understanding of what it takes to run inference for a given model, e.g. implement a model end-to-end.  For the longest time, I had a sense of what is needed, the mathematical operations involved, the runtime infrastructure needed, and challenges of dealing with weights, but found myeslf still asking questions that should have obvious answers to.

The intent is to update docs/ as I continue development, documenting my journey to stand-up an inference server, and hopefully leaving something for someone else to learn from; although, nowadays, agents can easily bang out an inference server in one shot and probably do a better job in teaching.

## AI Generated Overview 
> **"Just what the world needed: Yet Another Inference Server™ (YAIS)."**

Welcome to **Daphne**; a completely unhinged, built-for-fun learning exercise in implementing a high-performance LLM inference engine from scratch, starting with Google's **Gemma 4**.

---

## 🤔 Why "Daphne"?

A few working theories:
1. **Mythological escape:** In Greek myth, Daphne turned into a laurel tree just to get away from Apollo (the god of prophecies, light, and all-knowing knowledge). Much like her, this project runs in the opposite direction of 500,000-line monolithic inference frameworks just to figure out how transformers actually push weights around.
2. **Mystery-solving chic:** Daphne Blake from *Scooby-Doo* was always investigating why things were mysteriously haunted. Here, we investigate why our KV cache is haunted and eating all the VRAM.
3. **Honestly?** It just sounds a lot cooler than `cpp-gemma4-runner-final-v2-actually-final`.

---

## 🎯 What is this?

Let’s be real for a second: vLLM, TensorRT-LLM, SGLang, and llama.cpp already exist, work great, and are maintained by brilliant engineers who don't cry when writing custom CUDA kernels.

**Daphne is not trying to beat them.** 

Oh yeah, and we're using **C++** because we want to go *really fast* (and because I like C++).

Daphne is a hands-on sandbox and learning journey to build an LLM inference server from the ground up to deeply understand:
- 🧠 **Model Architecture:** Loading safetensors, parsing weights, and understanding every tensor operation inside **Gemma 4**.
- ⚡ **Token Generation & Sampling:** Greedy decoding, temperature, top-p, top-k, and penalty logic without magical black boxes.
- 📦 **KV Cache Management:** Paged attention, memory allocation, and not blowing up when context grows.
- 🌐 **Serving & API:** A clean, snappy HTTP/streaming endpoint that spits out tokens as fast as our code allows.

---

## 🚀 The Starter Target: Gemma 4

We are kicking things off with **Gemma 4** (`gemma-4-26B-A4B-it` and friends). 

Why Gemma 4? Because if you're going to learn how to juggle matrix multiplications, attention mechanisms, and rotary embeddings, you might as well do it with one of the slickest open architectures out there.

---

## 🛠️ The Roadmap (Or: "The Path of Most Resistance")

- [x] Download weights & stare at `.safetensors` files in awe
- [ ] Implement tokenizer & chat template parsing
- [ ] Build tensor tensor ops & model forward pass for Gemma 4
- [ ] Implement efficient KV caching
- [ ] Add streaming text generation
- [ ] Wrap in a tiny web server (OpenAI-compatible `/v1/chat/completions`)
- [ ] Benchmark, optimize, and profile until our fans sound like a jet engine

---

## 🏗️ Prerequisites & Building

Daphne uses **[Bazel](https://bazel.build/)** (with Bzlmod) and targets modern **C++23** / `/std:c++latest`.

### 1. Prerequisites

- **Build System:** [Bazelisk](https://github.com/bazelbuild/bazelisk) (recommended) or **Bazel 9.x+**
- **C++ Compiler:**
  - **Windows:** Visual Studio 2022 (MSVC) with the "Desktop development with C++" workload
  - **Linux:** GCC 13+ or Clang 16+ (supporting `-std=c++23`)
  - **macOS:** Apple Clang / Xcode (supporting `-std=c++23`)
- **Weights Download Tool (Optional):** [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf` or `huggingface-cli`) for fetching model weights into `models/weights/` (see [`models/README.md`](models/README.md))

External dependencies (such as [`nlohmann_json`](https://github.com/nlohmann/json) and [`mio`](https://github.com/vimpunk/mio)) are fetched automatically by Bazel via `MODULE.bazel`.

### 2. Building

Compile all targets across the workspace:

```bash
bazel build //...
```

### 3. Running Lessons & Tools

Run the Lesson 01 Safetensors inspection tool:

```bash
# Using root alias or full package target
bazel run //:safetensor_dmp -- <path/to/file.safetensors>
```

Example with downloaded Gemma 4 weights:

```bash
bazel run //:safetensor_dmp -- models/weights/gemma-4-26B-A4B-it/model-00001-of-00007.safetensors
```

### 4. IDE & LSP Setup (`compile_commands.json`)

To enable autocomplete, code navigation, and diagnostics in `clangd`, VS Code, Neovim, or CLion, generate a `compile_commands.json`:

```bash
bazel run //:refresh_compile_commands
```

---

## ⚠️ Disclaimer

This is a learning laboratory. If Daphne sets your GPU on fire, invents hallucinations never before witnessed by humanity, or produces 2 tokens per minute, remember: *it's built with love, curiosity, and pure determination.*

Pull requests, debugging tips, and emotional support are always welcome!
