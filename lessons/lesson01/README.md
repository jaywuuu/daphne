# Lesson 01 - Safetensors?  Just how safe?

One of the first questions I have is: given an open source model, where do I find the weights?  What about the source?  To implement a model by hand, I would need to know both: the ops involved, layers, computations, and the data.

For weights, it seems that huggingface is the most convenient way to obtain different kinds of weights for open source models.  Huggingface uses a binary format got safetensors, where the first 8 bytes stores a 64-bit integer that contains the length of the JSON header in bytes, followed by the raw binary weights.

So as a C struct, something like this:
```
struct header {
    uint64_t nbytes;
    char json_str[nbytes]; // obviously this doesn't compile
    char raw_bytes[..]
}
```

Easy enough to write a tool that memory maps the header.  Let's do that in [`safetensor_dmp.cpp`](safetensor_dmp.cpp).

### Running `safetensor_dmp`

Build and run `safetensor_dmp` using Bazel. You can pass the path to any `.safetensors` file as a positional argument or via `--file`:

```bash
# Pass as a positional argument:
bazel run //lessons/lesson01:safetensor_dmp -- <path/to/safetensors>

# Or pass via the --file flag (supports --indent, default: 2):
bazel run //lessons/lesson01:safetensor_dmp -- --file=<path/to/safetensors> --indent=2
```

> [!TIP]
> You can also use the root alias `//:safetensor_dmp` (e.g. `bazel run //:safetensor_dmp -- <path/to/safetensors>`).

For example, to dump the metadata header of Gemma 4 shard 1 (and optionally save to JSON):

```bash
bazel run //lessons/lesson01:safetensor_dmp -- models/weights/gemma-4-26B-A4B-it/model-00001-of-00007.safetensors > lessons/lesson01/gemma-4-26B-A4B-it_safetensors.json
```

After memory mapping and dumping, we see this:

```
"model.embed_vision.embedding_projection.weight": {
    "data_offsets": [
      0,
      6488064
    ],
    "dtype": "BF16",
    "shape": [
      2816,
      1152
    ]
  },
  "model.language_model.embed_tokens.weight": {
    "data_offsets": [
      6488064,
      1482883072
    ],
    "dtype": "BF16",
    "shape": [
      262144,
      2816
    ]
  },
  "model.language_model.layers.0.experts.down_proj": {
    "data_offsets": [
      1482883072,
      1990393856
    ],
    "dtype": "BF16",
    "shape": [
      128,
      2816,
      704
    ]
  },
```

Looks like a listing of weights, the data type, shape, and where to find them (offsets).  data_offsets gives the start and end (exclusive) byte bounds with start offset counting after the uint64_t and N bytes header.  Total size of the weight/tensor is then `end - start`.  Glancing that this already informs us on what our tensor class/struct likely needs to contain, i.e. shape, dtype, and where to find the data.  We also need some way to load the weights from file and keep track of where to find them, and also have some way to map the ops back to the constants.

Something like a key-value map or even an array of structs is probably good enough for now.  So to summarize, we need:
* A weights registry or something that parses the json headers of each weight file and records where to find them.
* Some kind of data structure that maps ops back to an entry in the registry (don't know what this is yet).
* A tensor class/struct for convenience and some factory functions.

Let's code this up in [`safetensor_loader.cpp`](safetensor_loader.cpp), [`types.h`](types.h), [`weights.h`](weights.h), and [`weights.cpp`](weights.cpp), and focus on the functionality and ignore all the usual best practices for expediency.

### Running `safetensor_loader`

Build and run `safetensor_loader` using Bazel. It accepts one or more Safetensor file paths passed as a comma-separated list to `--input_files`:

```bash
# Load a single safetensors file:
bazel run //lessons/lesson01:safetensor_loader -- --input_files=models/weights/gemma-4-26B-A4B-it/model-00001-of-00007.safetensors

# Load multiple safetensors shards:
bazel run //lessons/lesson01:safetensor_loader -- --input_files=models/weights/gemma-4-26B-A4B-it/model-00001-of-00007.safetensors,models/weights/gemma-4-26B-A4B-it/model-00002-of-00007.safetensors
```

> [!TIP]
> You can also use the root alias `//:safetensor_loader` (e.g. `bazel run //:safetensor_loader -- --input_files=...`).

When executed, it memory-maps each file, parses the tensor metadata into `weights_registry`, and prints the loaded count:

```text
Loaded 56 weights from 1 files
```

## The next question to answer: what are the ops?

Both the json header from the safetensors file and the `model.safetensors.index.json` file give an idea what sort of high level ops there are, e.g. projection, layer norm, self attention, some sliding window thing, etc., but the more direct way is to look at the reference implementations provided by the AI labs (or huggingface).  

For gemma4, we can look at:
* depemind's github page: https://github.com/google-deepmind/gemma/tree/main/gemma/gm/nn/gemma4.
* pytorch implementation: https://github.com/google/gemma_pytorch?utm_source=gemini.
* deepmind's C++ implementation: https://github.com/google/gemma.cpp?utm_source=gemini
* huggingface's transformer implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py


