# Text-to-SQL Model Deployment, Fine-tuning, and HTTP Inference Guide

This guide covers deployment, fine-tuning/training, and HTTP inference endpoint setup for Text-to-SQL models.

## Table of Contents

1. [Glossary - AI Terminology Explained](#glossary---ai-terminology-explained)
2. [Model Overview](#model-overview)
3. [Hardware Requirements](#hardware-requirements)
4. [Environment Setup](#environment-setup)
5. [Model Deployment](#model-deployment)
6. [Fine-tuning and Training](#fine-tuning-and-training)
7. [HTTP Inference API](#http-inference-api)
8. [Model Comparison](#model-comparison)

---

## Glossary - AI Terminology Explained

> This section explains all technical terms used in this guide. If you're new to AI/ML, please read this section first.

### Core Concepts

#### LLM (Large Language Model)

**What it is:** A type of AI model trained on massive amounts of text data that can understand and generate human language.

**Analogy:** Think of it as an extremely well-read assistant that has read billions of books, articles, and code, and can write text based on what it learned.

**In this guide:** All models mentioned (LLaMA, Qwen, SQLCoder) are LLMs specialized for SQL generation.

---

#### Inference

**What it is:** The process of using a trained model to make predictions or generate outputs.

**Analogy:** If training is like teaching a student, inference is like the student taking an exam - using what they learned to answer questions.

**In this guide:** When you send a question to the model and it returns SQL, that's inference.

---

#### Fine-tuning

**What it is:** Taking a pre-trained model and training it further on specific data to improve performance for a particular task.

**Analogy:** Like a general doctor specializing in cardiology - they already have medical knowledge, but they study more to become an expert in one area.

**In this guide:** Fine-tuning helps the model generate better SQL for your specific database schema and query patterns.

---

### Model Families

#### LLaMA (Large Language Model Meta AI)

**What it is:** A family of open-source LLMs created by Meta (Facebook). Known for good performance and being freely available.

| Version | Release | Notable Features |
|---------|---------|------------------|
| LLaMA 1 | 2023 Feb | First release, research only |
| LLaMA 2 | 2023 Jul | Commercial use allowed |
| LLaMA 3 | 2024 Apr | Best performance, used by SQLCoder |

**In this guide:** `llama-3-sqlcoder-8b` is based on LLaMA 3 with 8 billion parameters.

---

#### Qwen

**What it is:** A family of LLMs created by Alibaba. Strong performance in both English and Chinese, excellent for code generation.

**In this guide:** XiYanSQL-QwenCoder models are based on Qwen2, offering 3B to 32B parameter variants.

---

#### BERT

**What it is:** An older but still powerful model architecture created by Google, primarily used for understanding text (not generation).

**In this guide:** SPACE-T uses BERT architecture for Chinese table question-answering.

---

### Model Formats and Tools

#### GGUF (GPT-Generated Unified Format)

**What it is:** A file format for storing quantized (compressed) LLM models, designed to run efficiently on CPUs and consumer GPUs.

**Key benefits:**
- Runs on computers without powerful GPUs
- Smaller file sizes
- Faster loading times

**File naming convention:**
```
model_name.Q4_K_M.gguf
           │ │ │
           │ │ └── Variant (S=Small, M=Medium, L=Large)
           │ └──── Quantization method (K = K-quant)
           └────── Bits per weight (2/3/4/5/6/8)
```

**In this guide:** `sqlcoder-GGUF` allows running SQLCoder on machines without expensive GPUs.

---

#### Quantization Types Explained (Q2_K, Q4_K_M, etc.)

**What is Quantization?** 
Reducing the precision of model weights to make models smaller and faster, with some quality trade-off.

**Original model:** Uses 16 or 32 bits per weight (number) → Large file, high quality
**Quantized model:** Uses 2-8 bits per weight → Smaller file, slightly lower quality

**Quantization naming breakdown:**

| Component | Meaning | Example |
|-----------|---------|---------|
| **Q** | Quantization | All names start with Q |
| **Number (2-8)** | Bits per weight | Q4 = 4 bits, Q8 = 8 bits |
| **_K** | K-quant method | More advanced compression |
| **_S/_M/_L** | Size variant | S=Small, M=Medium (recommended), L=Large |
| **_0** | Basic quantization | Simpler method |

**Comparison table:**

| Type | Bits/Weight | Model Size | Quality | RAM Needed | Best For |
|------|-------------|------------|---------|------------|----------|
| Q2_K | ~2.5 | ~16% of original | Poor | ~6GB | Extreme memory limits |
| Q3_K_M | ~3.4 | ~22% of original | Fair | ~8GB | Low memory |
| **Q4_K_M** | ~4.5 | ~28% of original | **Good** | ~10GB | **Best balance (Recommended)** |
| Q5_K_M | ~5.5 | ~35% of original | Very Good | ~12GB | Quality priority |
| Q6_K | ~6.5 | ~40% of original | Excellent | ~15GB | Near-original quality |
| Q8_0 | 8 | ~50% of original | Best | ~18GB | Maximum quality |

**Visual guide:**
```
Quality:    Low ◄─────────────────────────────► High
            Q2_K   Q3_K_M   Q4_K_M   Q5_K_M   Q6_K   Q8_0
            
File Size:  Small ◄────────────────────────────► Large

Recommendation: Start with Q4_K_M for most use cases
```

---

#### llama.cpp

**What it is:** An open-source C/C++ project for running GGUF-format models on various hardware, especially CPUs.

> ⚠️ **Common Misconception:** Despite the name "llama.cpp", it's NOT limited to LLaMA models. It supports ANY model converted to GGUF format, including Qwen, Mistral, Phi, etc.

**Key features:**
- Pure C/C++ implementation (no Python required for inference)
- GGUF format support (the key requirement, not the model family)
- CPU and GPU acceleration (CUDA, Metal, OpenCL)
- Very memory efficient with quantization
- Runs on consumer hardware (laptops, Raspberry Pi, etc.)

**Supported model formats:**
| Format | Supported | Notes |
|--------|-----------|-------|
| GGUF | ✅ Yes | Primary format |
| SafeTensors | ❌ No | Must convert to GGUF first |
| PyTorch (.bin) | ❌ No | Must convert to GGUF first |

**In this guide:** Used to run GGUF-format SQLCoder models on CPU or low-VRAM GPUs.

---

#### vLLM

**What it is:** A high-performance inference engine for serving LLMs in production, supporting most popular model families.

> ⚠️ **Clarification:** vLLM supports LLaMA, Qwen, Mistral, and many other model families - it's not limited to any specific model type.

**Key features:**
- 10-24x faster than regular HuggingFace inference
- PagedAttention for efficient memory use
- OpenAI-compatible API (drop-in replacement)
- Continuous batching for high throughput
- Tensor parallelism for multi-GPU

**Supported model formats:**
| Format | Supported | Notes |
|--------|-----------|-------|
| SafeTensors | ✅ Yes | Preferred format |
| PyTorch (.bin) | ✅ Yes | Full support |
| GGUF | ⚠️ Limited | Experimental support |

**Supported model families:**
LLaMA, Qwen, Mistral, Phi, GPT-NeoX, Falcon, MPT, StarCoder, and 50+ more

**When to use:**
- Production deployments with high traffic
- When you have a GPU with sufficient VRAM
- When you need OpenAI-compatible API

**In this guide:** Recommended for deploying XiYanSQL and LLaMA-SQLCoder models in production.

---

#### HuggingFace Transformers

**What it is:** The most popular Python library for working with pre-trained models. Like an "App Store" for AI models.

**Key features:**
- Easy model loading with one line of code
- Supports thousands of models
- Works with PyTorch and TensorFlow
- Great for prototyping and development

**Limitations:**
- Slower inference than specialized engines
- No built-in batching or API server
- Higher memory usage

**In this guide:** Used for development, testing, and simple deployments.

---

### Inference Framework Comparison

Here's a comprehensive comparison of all popular LLM inference frameworks:

#### Quick Comparison Table

| Framework | Best For | Model Format | Hardware | Speed | Ease of Use |
|-----------|----------|--------------|----------|-------|-------------|
| **HuggingFace Transformers** | Development, Prototyping | SafeTensors, PyTorch | GPU | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **vLLM** | Production APIs | SafeTensors, PyTorch | GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **llama.cpp** | CPU/Edge, Low VRAM | GGUF | CPU/GPU | ⭐⭐⭐ | ⭐⭐⭐ |
| **TensorRT-LLM** | NVIDIA Production | TensorRT | NVIDIA GPU | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Ollama** | Local Desktop | GGUF (auto-download) | CPU/GPU | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Text Generation Inference (TGI)** | HuggingFace Cloud | SafeTensors | GPU | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SGLang** | Complex Prompts | SafeTensors | GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LMDeploy** | Chinese Models | SafeTensors, TurboMind | GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

#### Detailed Framework Descriptions

##### Ollama

**What it is:** A user-friendly tool for running LLMs locally with one-command setup.

**Key features:**
- One-command installation and model download
- Built-in model library (like Docker Hub for LLMs)
- REST API included
- Cross-platform (Windows, macOS, Linux)

**Best for:**
- Beginners wanting to try LLMs locally
- Desktop applications
- Quick prototyping

**Example usage:**
```bash
# Install and run a model in seconds
ollama run llama3
ollama run codellama
```

---

##### TensorRT-LLM

**What it is:** NVIDIA's optimized inference library for maximum performance on NVIDIA GPUs.

**Key features:**
- Highest possible performance on NVIDIA hardware
- FP8/INT8/INT4 quantization
- Multi-GPU and multi-node support
- Inflight batching

**Best for:**
- Production with NVIDIA GPUs
- Maximum throughput requirements
- Enterprise deployments

**Trade-offs:**
- Complex setup and compilation
- NVIDIA-only
- Requires model conversion

---

##### Text Generation Inference (TGI)

**What it is:** HuggingFace's production inference server, used by HuggingFace Inference API.

**Key features:**
- Production-ready out of the box
- Continuous batching
- Token streaming
- Prometheus metrics

**Best for:**
- HuggingFace ecosystem users
- Cloud deployments
- When you want HuggingFace support

---

##### SGLang

**What it is:** A fast inference engine optimized for complex prompt patterns and structured generation.

**Key features:**
- RadixAttention for prompt caching
- Excellent for multi-turn conversations
- Structured output (JSON mode) optimization
- Comparable speed to vLLM

**Best for:**
- Complex prompt workflows
- Agent applications
- Structured output generation

---

##### LMDeploy

**What it is:** An inference engine from Shanghai AI Lab, optimized for Chinese models and the Qwen family.

**Key features:**
- TurboMind high-performance engine
- Excellent Qwen/InternLM support
- KV cache quantization
- W4A16 quantization

**Best for:**
- Chinese language models
- Qwen family models
- When using InternLM models

---

#### Decision Flowchart

```
                    ┌─────────────────────────┐
                    │   Do you have a GPU?    │
                    └───────────┬─────────────┘
                           ╱           ╲
                         Yes            No
                         ╱               ╲
              ┌─────────▼──────┐    ┌────▼────────────┐
              │ Is it NVIDIA?  │    │ Use llama.cpp   │
              └───────┬────────┘    │ or Ollama       │
                 ╱         ╲        │ (GGUF format)   │
               Yes          No      └─────────────────┘
               ╱             ╲
    ┌─────────▼───────┐   ┌──▼──────────────┐
    │Production or Dev│   │ Use llama.cpp   │
    └───────┬─────────┘   │ (AMD/Intel GPU) │
       ╱         ╲        └─────────────────┘
  Production    Dev
     ╱            ╲
┌───▼────────┐  ┌──▼────────────────┐
│Need max    │  │ Use Transformers  │
│performance?│  │ (easy to use)     │
└──────┬─────┘  └───────────────────┘
   ╱       ╲
  Yes       No
  ╱          ╲
┌▼───────────┐ ┌▼───────────────────┐
│TensorRT-LLM│ │vLLM or TGI         │
│(complex    │ │(good balance)      │
│setup)      │ │                    │
└────────────┘ └────────────────────┘
```

---

#### Framework + Model Compatibility Matrix

| Model Family | Transformers | vLLM | llama.cpp | Ollama | TensorRT-LLM | TGI | LMDeploy |
|--------------|:------------:|:----:|:---------:|:------:|:------------:|:---:|:--------:|
| LLaMA 3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phi-3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| DeepSeek | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| StarCoder | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| BERT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

✅ = Full support | ⚠️ = Partial/Experimental | ❌ = Not supported

---

#### Recommendation Summary

| Your Situation | Recommended Framework |
|----------------|----------------------|
| Just learning, want easiest setup | **Ollama** |
| Python development, need flexibility | **HuggingFace Transformers** |
| Production API, good GPU | **vLLM** |
| Production API, max NVIDIA performance | **TensorRT-LLM** |
| No GPU / laptop / edge device | **llama.cpp** |
| Chinese models (Qwen, InternLM) | **LMDeploy** or **vLLM** |
| Complex agent workflows | **SGLang** |
| HuggingFace ecosystem | **TGI** |

---

### Training Techniques

#### LoRA (Low-Rank Adaptation)

**What it is:** A technique for fine-tuning LLMs efficiently by only training a small number of new parameters, instead of the entire model.

**Analogy:** Instead of repainting an entire house, you just add some decorative trim that changes the appearance.

**Benefits:**
- Uses 10-100x less GPU memory than full fine-tuning
- Training is much faster
- Original model weights stay unchanged
- Easy to switch between different LoRA adapters

**How it works (simplified):**
```
┌─────────────────────────────────────────────┐
│  Original Model (Frozen - not changed)      │
│  ┌─────────────────────────────────────┐    │
│  │  Billions of parameters             │    │
│  │  (Not trained, just used)           │    │
│  └─────────────────────────────────────┘    │
│                    +                         │
│  ┌─────────────────────────────────────┐    │
│  │  LoRA Adapters (New, small)         │    │
│  │  (Only these are trained)           │    │
│  │  ~0.1-1% of original size           │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**In this guide:** LoRA is used to fine-tune SQL models on your specific data.

---

#### QLoRA (Quantized LoRA)

**What it is:** Combines quantization with LoRA - loads the base model in 4-bit precision while training LoRA adapters.

**Benefits:**
- Fine-tune a 7B model on a single consumer GPU (e.g., RTX 3090)
- Uses ~75% less memory than standard LoRA

**In this guide:** Enable with `load_in_4bit=True` when fine-tuning.

---

### Hardware Terms

#### VRAM (Video RAM)

**What it is:** Memory on your GPU (graphics card). LLMs need VRAM to store model weights and perform calculations.

**Common GPU VRAM:**
| GPU | VRAM | Can Run |
|-----|------|---------|
| RTX 3060 | 12GB | 7B models (quantized) |
| RTX 3090/4090 | 24GB | 8B-13B models |
| A10 | 24GB | 8B-13B models |
| A100-40G | 40GB | 14B-30B models |
| A100-80G | 80GB | 32B-70B models |

**In this guide:** Hardware requirements are specified by VRAM needs.

---

#### GPU Layers (n_gpu_layers / -ngl)

**What it is:** When using GGUF models, you can choose how many model layers to run on GPU vs CPU.

```
-ngl 0   → All layers on CPU (slower, no GPU needed)
-ngl 32  → 32 layers on GPU (faster, uses VRAM)
-ngl -1  → All layers on GPU (fastest, most VRAM)
```

**In this guide:** Adjust based on your available VRAM.

---

### Model Parameters

#### Parameters (B = Billion)

**What it is:** The number of "learnable weights" in a model. More parameters generally means better performance but requires more resources.

| Size | Parameters | Typical VRAM | Quality |
|------|------------|--------------|---------|
| Small | 1-3B | 4-8GB | Good for simple tasks |
| Medium | 7-8B | 16-24GB | Balanced |
| Large | 13-14B | 32-40GB | High quality |
| XLarge | 30-70B | 80GB+ | Best quality |

**In this guide:** Models range from 3B (XiYanSQL-3B) to 32B (XiYanSQL-32B).

---

#### Temperature

**What it is:** A parameter that controls randomness in model outputs.

```
temperature=0   → Always picks most likely token (deterministic)
temperature=0.7 → Balanced creativity
temperature=1.0 → More random/creative
```

**For SQL generation:** Use `temperature=0` for consistent, correct SQL.

---

#### Tokens

**What it is:** The basic units that LLMs process. Roughly corresponds to word pieces.

```
"Hello world" → ["Hello", " world"] → 2 tokens
"SELECT * FROM users" → ["SELECT", " *", " FROM", " users"] → 4 tokens
```

**Token limits:**
- Models have a maximum "context window" (e.g., 4096 tokens)
- This includes both input (your question + schema) AND output (generated SQL)

---

### Deployment Terms

#### FastAPI

**What it is:** A modern Python web framework for building APIs quickly.

**In this guide:** Used to create HTTP endpoints for the SQL generation service.

---

#### Docker

**What it is:** A tool for packaging applications with all their dependencies into "containers" that run consistently anywhere.

**In this guide:** Used to deploy the Text2SQL service in a reproducible way.

---

### Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    QUICK TERMINOLOGY REFERENCE                  │
├────────────────────────────────────────────────────────────────┤
│ LLaMA/Qwen    = Base model families (like car manufacturers)   │
│ GGUF          = Compressed model format (like ZIP for models)  │
│ Q4_K_M        = Quantization level (compression quality)       │
│ llama.cpp     = Tool to run GGUF models                        │
│ vLLM          = Fast production inference server               │
│ LoRA          = Efficient fine-tuning method                   │
│ VRAM          = GPU memory (determines what models you can run)│
│ Inference     = Using model to generate outputs                │
│ Fine-tuning   = Training model on your specific data           │
└────────────────────────────────────────────────────────────────┘
```

---

## Model Overview

| Model | Base Model | Size | License | Best For |
|-------|-----------|------|---------|----------|
| [llama-3-sqlcoder-8b](https://huggingface.co/defog/llama-3-sqlcoder-8b) | Meta-Llama-3-8B-Instruct | 8B | CC-by-SA-4.0 | English SQL generation |
| [sqlcoder-GGUF](https://huggingface.co/TheBloke/sqlcoder-GGUF) | StarCoder | 15B | Apache 2.0 | CPU/Low VRAM inference |
| [XiYanSQL-QwenCoder](https://modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-3B-2502) | Qwen2 | 3B/7B/14B/32B | Apache 2.0 | Multi-dialect SQL (SQLite/PostgreSQL/MySQL) |
| [SPACE-T](https://modelscope.cn/models/iic/nlp_convai_text2sql_pretrain_cn) | BERT | Base | Apache 2.0 | Chinese multi-turn table QA |

---

## Hardware Requirements

### GPU Inference (Recommended)

| Model | Minimum VRAM | Recommended VRAM |
|-------|-------------|------------------|
| llama-3-sqlcoder-8b | 16GB | 24GB (A10/RTX 4090) |
| XiYanSQL-3B | 8GB | 16GB |
| XiYanSQL-7B | 16GB | 24GB |
| XiYanSQL-14B | 32GB | 40GB (A100) |
| XiYanSQL-32B | 64GB | 80GB (A100-80G) |
| SPACE-T | 4GB | 8GB |

### CPU Inference (GGUF Models)

| Quantization | RAM Required | Quality |
|-------------|-------------|---------|
| Q2_K | ~6GB | Low |
| Q4_K_M | ~10GB | Good (Recommended) |
| Q5_K_M | ~12GB | Better |
| Q8_0 | ~18GB | Best |

---

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.37.0
pip install accelerate
pip install safetensors

# For vLLM (recommended for production)
pip install vllm>=0.9.2

# For GGUF models
pip install llama-cpp-python

# For ModelScope models
pip install modelscope

# HTTP server
pip install fastapi uvicorn
```

---

## Model Deployment

### Option 1: LLaMA-3-SQLCoder-8B (HuggingFace Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "defog/llama-3-sqlcoder-8b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prompt template
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Generate a SQL query to answer this question: `{question}`
{instructions}
DDL statements:
{ddl}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The following SQL query best answers the question `{question}`:
```sql
"""

def generate_sql(question: str, ddl: str, instructions: str = "") -> str:
    formatted_prompt = prompt.format(
        question=question,
        ddl=ddl,
        instructions=instructions
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract SQL from response
    sql = response.split("```sql")[-1].split("```")[0].strip()
    return sql
```

### Option 2: XiYanSQL-QwenCoder with vLLM (Production Recommended)

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "XGenerationLab/XiYanSQL-QwenCoder-3B-2502"

# Initialize vLLM engine
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,  # Increase for multi-GPU
    dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    n=1,
    temperature=0.1,
    max_tokens=1024,
    top_p=0.8
)

# Prompt template (Chinese)
prompt_template = """You are a {dialect} expert. Read and understand the database schema and reference information below, then generate a SQL query to answer the user question.

【User Question】
{question}

【Database Schema】
{db_schema}

【Reference Information】
{evidence}

【User Question】
{question}

```sql"""

def generate_sql_vllm(question: str, db_schema: str, dialect: str = "SQLite", evidence: str = "") -> str:
    prompt = prompt_template.format(
        dialect=dialect,
        question=question,
        db_schema=db_schema,
        evidence=evidence
    )
    
    message = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    
    outputs = llm.generate([text], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()
```

### Option 3: SQLCoder-GGUF with llama.cpp (CPU/Low VRAM)

```bash
# Download model
huggingface-cli download TheBloke/sqlcoder-GGUF sqlcoder.Q4_K_M.gguf --local-dir ./models

# Run with llama.cpp
./llama.cpp/build/bin/llama-server \
    -m ./models/sqlcoder.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 2048 \
    -ngl 32  # GPU layers (remove for CPU-only)
```

Python usage with llama-cpp-python:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/sqlcoder.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=32  # Set to 0 for CPU-only
)

def generate_sql_gguf(prompt: str) -> str:
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0,
        stop=["```", "\n\n"]
    )
    return output["choices"][0]["text"].strip()
```

### Option 4: SPACE-T Chinese Model (ModelScope)

```python
import os
import json
from transformers import BertTokenizer
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks

model_id = 'damo/nlp_convai_text2sql_pretrain_cn'

model = Model.from_pretrained(model_id)
tokenizer = BertTokenizer(
    os.path.join(model.model_dir, ModelFile.VOCAB_FILE)
)

db = Database(
    tokenizer=tokenizer,
    table_file_path=os.path.join(model.model_dir, 'table.json'),
    syn_dict_file_path=os.path.join(model.model_dir, 'synonym.txt'),
    is_use_sqlite=True
)

preprocessor = TableQuestionAnsweringPreprocessor(
    model_dir=model.model_dir,
    db=db
)

pipe = pipeline(
    Tasks.table_question_answering,
    model=model,
    preprocessor=preprocessor,
    db=db
)

def query_table(question: str, table_id: str, history_sql=None):
    output = pipe({
        'question': question,
        'table_id': table_id,
        'history_sql': history_sql
    })[OutputKeys.OUTPUT]
    
    return {
        'sql_string': output[OutputKeys.SQL_STRING],
        'sql_query': output[OutputKeys.SQL_QUERY],
        'history': output[OutputKeys.HISTORY]
    }
```

---

## Fine-tuning and Training

### Fine-tuning XiYanSQL/LLaMA-SQLCoder with LoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Load base model
model_name = "defog/llama-3-sqlcoder-8b"  # or XiYanSQL model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # QLoRA for memory efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Prepare dataset
# Format: {"text": "<prompt>\n<sql_response>"}
dataset = load_dataset("your_dataset")

# Training arguments
training_args = TrainingArguments(
    output_dir="./sqlcoder-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
)

trainer.train()
trainer.save_model("./sqlcoder-finetuned")
```

### Fine-tuning SPACE-T Chinese Model

```python
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.table_question_answering_trainer import TableQuestionAnsweringTrainer
from modelscope.utils.constant import DownloadMode

# Load dataset
input_dataset = MsDataset.load(
    'ChineseText2SQL',
    download_mode=DownloadMode.FORCE_REDOWNLOAD
)

train_dataset = []
for name in input_dataset['train']._hf_ds.data[1]:
    train_dataset.append(json.load(open(str(name), 'r')))

eval_dataset = []
for name in input_dataset['test']._hf_ds.data[1]:
    eval_dataset.append(json.load(open(str(name), 'r')))

# Initialize trainer
model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
trainer = TableQuestionAnsweringTrainer(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train(
    batch_size=8,
    total_epoches=2,
)

# Evaluate
trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir, 'finetuned_model.bin')
)
```

### Training Data Format

Prepare your training data in JSONL format:

```json
{
    "question": "What are the top 5 customers by revenue?",
    "schema": "CREATE TABLE customers (id INT, name VARCHAR, revenue DECIMAL);",
    "sql": "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 5",
    "evidence": ""
}
```

---

## HTTP Inference API

### FastAPI Server Implementation

```python
# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Text2SQL API")

# Global model instance
model = None
tokenizer = None

class SQLRequest(BaseModel):
    question: str
    schema: str
    dialect: str = "SQLite"
    evidence: Optional[str] = ""

class SQLResponse(BaseModel):
    sql: str
    success: bool
    error: Optional[str] = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_name = "defog/llama-3-sqlcoder-8b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

@app.post("/v1/sql/generate", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    try:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Generate a SQL query to answer this question: `{request.question}`
DDL statements:
{request.schema}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The following SQL query best answers the question `{request.question}`:
```sql
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = response.split("```sql")[-1].split("```")[0].strip()
        
        return SQLResponse(sql=sql, success=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the Server

```bash
# Development
python server.py

# Production with multiple workers
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### vLLM OpenAI-Compatible Server (Recommended for Production)

```bash
# Start vLLM server with OpenAI API compatibility
python -m vllm.entrypoints.openai.api_server \
    --model XGenerationLab/XiYanSQL-QwenCoder-3B-2502 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096
```

Client usage:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="XGenerationLab/XiYanSQL-QwenCoder-3B-2502",
    messages=[
        {"role": "user", "content": "Generate SQL for: What are the top 10 products by sales?"}
    ],
    temperature=0.1,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  text2sql-api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
    environment:
      - TRANSFORMERS_CACHE=/app/models
```

---

## Model Comparison

### Performance Benchmarks

| Model | BIRD Dev (M-Schema) | Spider Test (M-Schema) | Speed (tokens/s) |
|-------|--------------------|-----------------------|------------------|
| XiYanSQL-QwenCoder-3B | 54.11% | 82.69% | ~100 |
| XiYanSQL-QwenCoder-7B | 59.78% | 84.86% | ~60 |
| XiYanSQL-QwenCoder-14B | 63.10% | 85.76% | ~30 |
| XiYanSQL-QwenCoder-32B | 67.01% | 88.39% | ~15 |
| llama-3-sqlcoder-8b | ~55% | ~80% | ~50 |
| GPT-4o-0806 | 58.47% | 82.89% | N/A |

### Recommendation

| Use Case | Recommended Model |
|----------|------------------|
| English SQL, balanced performance | llama-3-sqlcoder-8b |
| Chinese multi-turn table QA | SPACE-T |
| Multi-dialect, high accuracy | XiYanSQL-QwenCoder-14B/32B |
| Low resource / CPU inference | sqlcoder-GGUF (Q4_K_M) |
| Production with high throughput | XiYanSQL + vLLM |

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Use `load_in_4bit=True` for QLoRA
   - Reduce `max_new_tokens`
   - Use GGUF quantized models

2. **Slow Inference**
   - Use vLLM for production
   - Enable Flash Attention 2
   - Use tensor parallelism for multi-GPU

3. **Poor SQL Quality**
   - Provide detailed schema with column descriptions
   - Include sample data in the prompt
   - Fine-tune on domain-specific data

---

## References

- [Defog SQLCoder](https://defog.ai/sqlcoder-demo/)
- [XiYan-SQL GitHub](https://github.com/XGenerationLab/XiYan-SQL)
- [vLLM Documentation](https://docs.vllm.ai/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ModelScope SPACE-T](https://modelscope.cn/models/iic/nlp_convai_text2sql_pretrain_cn)
