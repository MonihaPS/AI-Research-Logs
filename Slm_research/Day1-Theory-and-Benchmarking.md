# Slm_research / Day 1 – Theory, Concepts & Initial Benchmarking  
**Date:** 04 February 2026  
**Goal:** Understand local SLMs + quantization + benchmarking for offline smartwatch assistant (ChatGPT-like, fully private, <2 GB RAM)

## 1. Core Concepts Learned (from scratch)

### 1.1 Local Models vs Cloud AI
- Cloud AI (ChatGPT, GPT-4): request goes to remote server → answer comes back.  
  → Fast when connected, but: no privacy, costs money, needs internet, data leaves device.
- Local Models: download the entire model to your device (laptop/phone/watch) → runs 100% offline.  
  Analogy: Cloud = ordering pizza, Local = cooking in your kitchen (full control, private, but needs good hardware).

Why it matters for smartwatch:
- Privacy (health data never leaves wrist)
- Works without phone/internet
- Zero recurring cost
- Ultra-low latency (no round-trip)

Downside: only small models (0.5B–4B params) fit in 1–2 GB RAM.

### 1.2 Quantization – Making Models Tiny
Big model = huge book written with super-precise ink (16-bit floats) → 7–14 GB RAM.  
Quantization = rewrite with simpler ink (4-bit/8-bit integers) → 3–4× smaller, much faster on CPU/NPU, almost same “intelligence”.

Popular formats:
- GGUF (llama.cpp / Ollama / LM Studio) → best for CPU / on-device
- BitsAndBytes (HF) → good for fine-tuning

Quantization levels (like coffee strengths):

| Level     | Bits | Approx. size (3B model) | Quality retention | Best for                  |
|-----------|------|--------------------------|-------------------|---------------------------|
| Q8_0      | 8    | ~3–4 GB                 | Almost perfect    | When you have RAM         |
| Q5_K_M    | ~5   | ~2 GB                   | Excellent         | Sweet spot                |
| **Q4_K_M** | ~4   | ~1.5–2 GB              | Very good         | **target**            |
| Q3_K_M    | ~3   | ~1.2 GB                 | Noticeable drop   | Extreme memory saving     |

The "_K_M" variants protect important weights better → modern SLMs (Llama 3.2, Phi-3.5, Gemma 2, Qwen3) keep 90–98% of original logic even at 4-bit.

### 1.3 Benchmarking – Real vs Public Benchmarks
Public benchmarks (MMLU, GSM8K, etc.) are done on full-precision models → useless for us.  
We need to test **quantized** versions on **our exact use-case** (personal watch data + general student questions).

Evaluation method:
- Manual 1–5 scoring on real questions
- Measure RAM (Task Manager), speed (seconds per answer), empty-output rate, hallucination

## 2. Day 1 Task Sheet (planned vs actual)

**Planned (from task sheet):**
- Download 4 models in Q4_K_M
- Run + measure RAM/speed/quality
- Pick top 2 under 1.5 GB

**What we actually did:**
- Used LM Studio (easiest on Windows)
- Tested 3 models completely + 2 more downloading right now

## 3. Models Tested & Results

| Model                        | Size | Quant   | Peak RAM | Avg time/answer | Personal questions | General questions | Notes / Verdict                          |
|------------------------------|------|---------|----------|-----------------|--------------------|-------------------|------------------------------------------|
| Qwen3-0.6B                   | 0.6B | Q4_K_M  | ~370 MB  | 13–40 s         | Weak/refuses       | Partial/broken    | Too weak overall                         |
| Llama-3.2-1B-Instruct        | 1B   | Q4_K_M  | <300 MB  | 26–48 s         | Weak/partial       | OK on simple math | Friendlier tone, still limited           |
| SmolLM2-1.7B-Instruct        | 1.7B | Q4_K_M  | ~1.4–1.8 GB | 30–70 s      | Completely fails   | Very good explanations | Inconsistent (many empty outputs)        |
| Gemma-2-2B-it (downloading)  | 2B   | Q4_K_M  | –        | –               | –                  | –                 | Next to test                             |
| Phi-3.5-mini-instruct (downloading) | 3.8B | Q4_K_M | –        | –               | –                  | –                 | Strongest candidate (expected ~2.4 GB)   |

**Key takeaway after Day 1**  
Sub-2B models are memory-efficient but too weak for a real ChatGPT-like assistant (especially personal context + reasoning).  
We need at least 2B–3.8B class → will test Gemma-2-2B + Phi-3.5-mini today and decide the base model.

## 4. Current Status (04 Feb 2026)
- Gemma-2-2B-it-Q4_K_M.gguf (~1.71 GB) downloading
- Phi-3.5-mini-instruct-Q4_K_M.gguf (~2.39 GB) downloading
- After testing both → final model selection + start RAG prototype (Day 2)

## 5. References & Resources
- Theory notes from initial study (Local Models, Quantization, Benchmarking)
- LM Studio for quick testing
- bartowski GGUF repos (best quality quantized models)

**Next:** Day 2 – Final model selection + minimal RAG prototype

---
Last updated: 04 February 2026 