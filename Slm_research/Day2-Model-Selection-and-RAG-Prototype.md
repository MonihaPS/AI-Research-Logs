# Slm_research / Day 2 – Model Selection Completion & Minimal RAG Prototype  
**Date:** 04 February 2026 (continued) / 05 February 2026 (finalized)  
**Goal:** Finish benchmarking last two candidate models (Gemma-2-2B & Phi-3.5-mini), compare all five models, select final base model, and build first working offline RAG prototype.

## 1. Models Tested Today (Day 2)

| Model                        | Params | Quant   | Peak RAM (Task Manager) | Avg time / answer | Personal questions (heart rate, battery, reminder) | General / Student questions (explanation, math, code, translation) | Overall Score (1–10) | Verdict |
|------------------------------|--------|---------|--------------------------|-------------------|-----------------------------------------------------|---------------------------------------------------------------------|----------------------|---------|
| Qwen3-0.6B                   | 0.6B   | Q4_K_M  | ~370 MB                  | 13–40 s           | Weak / partial / broken                             | Partial / broken / wrong Hindi                                      | 3/10                 | Too weak |
| Llama-3.2-1B-Instruct        | 1B     | Q4_K_M  | <300 MB                  | 26–48 s           | Weak / partial / wrong advice                       | OK on simple math, weak elsewhere                                   | 4/10                 | Slightly friendlier but limited |
| SmolLM2-1.7B-Instruct        | 1.7B   | Q4_K_M  | ~1.4–1.8 GB              | 30–70 s           | Completely fails (empty)                            | Very good explanations, many empty outputs                          | 5/10                 | Inconsistent |
| **Gemma-2-2B-it**            | 2B     | Q4_K_M  | **250–270 MB**           | **Much longer** (60–120+ s) | **Very good** (graceful refusal + helpful advice)   | **Excellent** (clear, structured, correct code/math/explanation)    | **8/10**             | **Clear winner so far** |
| **Phi-3.5-mini-instruct**    | 3.8B   | Q4_K_M  | **~270 MB**              | **Very slow** (longest of all) | **Good refusal + advice**, but long-winded          | Strong reasoning, but overly verbose & detailed                     | **7/10**             | Very capable, but too slow & wordy |

### Key Takeaways from Day 2 Testing
- **Gemma-2-2B-it** is the best overall performer so far:
  - Lowest RAM usage (250–270 MB) — ideal for watch
  - Highest quality answers (especially explanations, code, reasoning)
  - Graceful handling of personal questions (refuses + gives useful advice)
  - No empty outputs
  - Only downside: noticeably slower on laptop CPU (expected for 2B model)

- **Phi-3.5-mini-instruct** (3.8B):
  - Surprisingly low RAM (~270 MB — almost same as Gemma!)
  - Very strong reasoning capability
  - Extremely verbose / long answers → not ideal for watch (users want short replies)
  - Slowest inference speed of all five models

- **Speed ranking** (fastest → slowest on laptop CPU):
  1. Qwen 0.6B  
  2. Llama 1B  
  3. SmolLM2 1.7B  
  4. Gemma 2B  
  5. Phi-3.5-mini (slowest)

- **Final model selection (for now)**:  
  **Gemma-2-2B-it Q4_K_M**  
  → Lowest RAM + best balance of quality & refusal behavior + no empty outputs  
  → Phi is stronger in pure reasoning, but too slow and wordy for watch use-case  
  → We will use Gemma as base for RAG prototype

## 2. Minimal RAG Prototype (first working version)

**Objective:** Combine retrieved personal/watch knowledge with Gemma generation to answer questions accurately without hallucination.

**Implementation details:**
- Embedding model: `all-MiniLM-L6-v2` (~80 MB)
- Vector store: FAISS (in-memory, very fast)
- LLM backend: LM Studio local server (Gemma-2-2B-it loaded)
- Documents: stored in separate file `watch_knowledge.txt` (easy to edit)
- Interactive mode: real-time question input (type and get answer)

**Sample documents added (watch_knowledge.txt):**
- Daily step goal: 10,000 steps.
- You have taken 4,832 steps today so far.
- Normal resting heart rate: 60–100 BPM.
- Current heart rate: 72 BPM (normal).
- Battery level: 42%. Turn off always-on display below 30%.
- Meeting reminder: Team sync at 3:00 PM today.
- Sleep last night: 7 hours 12 minutes.
- Photosynthesis explanation (simple).
- Python factorial code snippet.
- Hindi & French translation examples.

**Early test results (pure RAG + Gemma):**
- Personal questions → now answered correctly from documents
- Out-of-scope questions → correctly says "I don't have that information"
- Explanations / math / code → still excellent (Gemma strength)
- No hallucinations on missing data

**Current prototype status:**
- Working end-to-end
- RAM usage during RAG: ~300–450 MB (embedding + index + model)
- Ready for next improvements: better prompt, conversation memory, more documents

## 3. Next Steps (Day 3 & beyond)
- Refine prompt → shorter, more watch-like answers
- Add conversation history / context carry-over
- Expand knowledge base (real health rules, user manual snippets, calendar sync simulation)
- Measure real latency → optimize top_k, context length
- Test Phi-3.5-mini with same RAG (compare quality vs speed)
- Explore watch deployment options (ExecuTorch, MLC-LLM, ONNX Runtime)
- Create benchmark table: pure model vs RAG vs cloud (GPT-4o / Grok)

**Last updated:** 05 February 2026  
**Author:** MONIHA  
**Project repo:** AI-Research-Logs / Slm_research