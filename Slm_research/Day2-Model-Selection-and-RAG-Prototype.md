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

## 2. Minimal RAG Pipeline – Complete Step-by-Step (Reproducible)

Anyone can copy these exact steps and get the same offline RAG smartwatch assistant running.

### Folder Structure (create this)

```text
D:\bluvern\slm
├── models/                          ← all .gguf files here
├── watch_knowledge.txt              ← your personal + knowledge data
├── rag_prototype.py                 ← main script
└── rag_env/                         ← virtual environment
```text

**Step 1:** Prepare Documents (very important)
Create file: `D:\bluvern\slm\watch_knowledge.txt`

```txt
Daily step goal: 10,000 steps.
You have taken 4,832 steps today so far.
Normal resting heart rate: 60–100 BPM.
Current heart rate: 72 BPM (normal).
Battery level: 42%. Turn off always-on display below 30%.
Meeting reminder: Team sync at 3:00 PM today.
Sleep last night: 7 hours 12 minutes (good quality).
Drink 2–3 liters of water daily.
Photosynthesis: Plants use sunlight, water and CO2 to make sugar and release oxygen.
Python factorial: def factorial(n): return 1 if n == 0 else n * factorial(n-1)
x + 3 = 10 → x = 7
Good morning, how did you sleep? → Hindi: सुप्रभात, आपने कैसे सोया?

Step 2: Start LM Studio + Gemma

Open LM Studio
Load gemma-2-2b-it-Q4_K_M.gguf
Settings: Context = 2048, Temperature = 0.8
Go to Local Server tab → click Start Server (port 1234)

Step 3: Create Virtual Environment & Install Packages
PowerShellcd D:\bluvern\slm
python -m venv rag_env
rag_env\Scripts\activate

pip install sentence-transformers faiss-cpu numpy openai

Step 4: Create the RAG Script
Create file: D:\bluvern\slm\rag_prototype.py
(Paste the full script I gave you earlier – the interactive version)

Step 5: Run the RAG Assistant
PowerShellrag_env\Scripts\activate
python rag_prototype.py
You will see:
textLoaded 13 knowledge entries...
RAG index ready!
Smartwatch RAG Assistant is ready! (type 'exit' to stop)

You:
Now type any question in real time — it will use RAG + Gemma to answer.
Current Performance (as of 06 Feb 2026)

RAM during RAG: ~300–450 MB
Personal questions → answered correctly from documents
Unknown questions → correctly says “I don’t have that information”
Speed: acceptable on laptop (will be much faster on watch NPU)

How to Extend It Later

Add more lines to watch_knowledge.txt → restart the script
Change prompt in rag_prototype.py for shorter/friendlier answers
Add conversation memory (next step)

## 3. Next Steps (Day 3 & beyond)
- Refine prompt → shorter, more watch-like answers
- Add conversation history / context carry-over
- Expand knowledge base (real health rules, user manual snippets, calendar sync simulation)
- Measure real latency → optimize top_k, context length
- Test Phi-3.5-mini with same RAG (compare quality vs speed)
- Explore watch deployment options (ExecuTorch, MLC-LLM, ONNX Runtime)
- Create benchmark table: pure model vs RAG vs cloud (GPT-4o / Grok)

**Last updated:** 05 February 2026  
