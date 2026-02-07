# Slm_research / Day 3 – Qwen & Phi-3.5-mini RAG Prototypes  
**Date:** 07 February 2026  
**Goal:** Build fully working offline RAG prototypes with **Qwen3-0.6B** and **Phi-3.5-mini-instruct** and compare them.


## Common Prerequisites (Do Once)
1. LM Studio running + **Local Server started** (`http://localhost:1234/v1`)
2. Create virtual environment:
   ```powershell
   cd D:\bluvern\slm
   python -m venv venv
   venv\Scripts\activate
   pip install sentence-transformers faiss-cpu numpy openai

**Step 1: Create watch_knowledge.txt**
File path: D:\bluvern\slm\watch_knowledge.txt
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

**Option A: Qwen3-0.6B RAG Prototype**

Script: rag_prototype_qwen.py
Run:
venv\Scripts\activate
python rag_prototype_qwen.py

Sample Output:
You: How many steps have I taken today?
Watch: You have taken 4,832 steps today so far.

You: What is the weather today?
Watch: I don't have that information.

**Option B: Phi-3.5-mini-instruct RAG Prototype**

Script: rag_prototype_phi.py
Run:
venv\Scripts\activate
python rag_prototype_phi.py

Sample Output:
textYou: How many steps have I taken today?
Watch: 4,832 steps

You: What is my current heart rate?
Watch: 72 BPM

You: Should I turn on always-on display at 25% battery?
Watch: No, to conserve power when the battery is below optimal levels.

**Comparison Table (07 Feb 2026)**

Model,Peak RAM,Speed (laptop),Personal Questions,Answer Style,Hallucination,Recommendation
Qwen3-0.6B + RAG,~400 MB,Fast,Good,Very short & direct,Very low,Best for smartwatch
Phi-3.5-mini + RAG,~2.6 GB,Slow,Very good,More detailed,Very low,"Stronger reasoning, heavier"

Current Decision:
Qwen3-0.6B + RAG is the most practical choice for actual smartwatch deployment (lowest RAM + fastest).

**Next Steps (Day 4 & beyond)**

Add conversation history (memory)
Make answers even shorter & more watch-friendly
Expand knowledge base
Test with real user data
Explore watch deployment (ExecuTorch, MLC-LLM)

Last updated: 07 February 2026