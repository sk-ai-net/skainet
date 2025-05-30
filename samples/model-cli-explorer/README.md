# Model CLI Explorer

A command-line tool for exploring model files (GGUF, SafeTensors) and displaying their metadata, statistics, and tensor information.

## Features

- Automatic recognition of file format (GGUF, SafeTensors)
- Displays metadata from the model file
- Displays statistics (file size, number of tensors, memory consumption)
- Displays a list of tensors with their names, data types, and shapes

## Building

### JVM

```bash
./gradlew :samples:model-cli-explorer:build
```

### Native (Linux)

```bash
./gradlew :samples:model-cli-explorer:linkReleaseExecutableLinuxX64
```

### Native (macOS)

```bash
./gradlew :samples:model-cli-explorer:linkReleaseExecutableMacosX64
# or for Apple Silicon
./gradlew :samples:model-cli-explorer:linkReleaseExecutableMacosArm64
```

## Running

### JVM

```bash
java -jar samples/model-cli-explorer/build/libs/model-cli-explorer-jvm.jar -f /path/to/model/file
```

### Native (Linux)

```bash
samples/model-cli-explorer/build/bin/linuxX64/releaseExecutable/model-cli-explorer.kexe -f /path/to/model/file
```

### Native (macOS)

```bash
samples/model-cli-explorer/build/bin/macosX64/releaseExecutable/model-cli-explorer.kexe -f /path/to/model/file
# or for Apple Silicon
samples/model-cli-explorer/build/bin/macosArm64/releaseExecutable/model-cli-explorer.kexe -f /path/to/model/file
```

## Usage

```
Usage: model-explorer options_list
Options: 
    --file, -f -> Path to the model file (always required) { String }
    --help, -h -> Usage info 
```

## Example Output

```
Model Explorer
==============
File: /path/to/model/file

File Format: GGUF

Metadata:
  general.architecture: llama
  general.name: Llama 2 7B
  llama.context_length: 4096
  llama.embedding_length: 4096
  llama.block_count: 32
  llama.feed_forward_length: 11008
  llama.rope.dimension_count: 128
  llama.attention.head_count: 32
  llama.attention.head_count_kv: 32
  llama.attention.layer_norm_rms_epsilon: 0.000010
  tokenizer.ggml.model: llama
  tokenizer.ggml.tokens: 32000
  tokenizer.ggml.bos_token_id: 1
  tokenizer.ggml.eos_token_id: 2

Statistics:
  File Size: 13.24 GB
  Number of Tensors: 291
  Total Elements: 6,740,000,000
  Estimated Memory Consumption: 13.48 GB

Tensors:
  Name                                  | Data Type      | Shape
  ---------------------------------------------------------------------
  token_embd.weight                     | Float64        | Shape: Dimensions = [32000 x 4096], Size (Volume) = 131072000
  blk.0.attn_q.weight                   | Int8           | Shape: Dimensions = [4096 x 4096], Size (Volume) = 16777216
  blk.0.attn_k.weight                   | Int8           | Shape: Dimensions = [4096 x 4096], Size (Volume) = 16777216
  ...
```