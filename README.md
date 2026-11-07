# RAG Evaluation Suite

> **Prove your RAG system works, with a number you can defend in a review.**

This evaluation suite measures **retrieval quality** (did the right passage come back?) separately from **generation quality** (was the answer right given the passage?).

---

## 🏆 The Number to Defend: 93.5%

In an engineering or stakeholder review, lead with the composite **Review Score**:

$$\text{Review Score} = 0.40 \times \text{Hit@1} + 0.35 \times \text{Oracle Acc} + 0.25 \times \text{Abstain Acc}$$

| Dimension | Metric | Score | What It Proves | What It Does Not Prove |
| :--- | :--- | :---: | :--- | :--- |
| **Retrieval (Precision)** | **Hit@1** | **85.7%** | Precision of the top-ranked passage | That the model answered correctly |
| **Retrieval (Window)** | **Hit@5** | **100.0%** | The gold passage entered the context window | That rank 1 was the correct passage |
| **Retrieval (Ranking)** | **MRR** | **0.9127** | Mean Reciprocal Rank across the corpus | Downstream answer correctness |
| **Generation (Oracle)** | **Oracle Acc** | **97.6%** | Given the gold passage alone, answer correctness | That search retrieved it in production |
| **Generation (End-to-End)**| **RAG Acc** | **97.8%** | Full pipeline correctness | *Which* stage failed when an error occurs |
| **Safety (Abstain)** | **Abstain Acc** | **100.0%** | Deliberately unanswerable queries refused | That answerable queries are answered |

---

## Why Score Retrieval and Generation Separately?

Retrieval and generation fail differently:

1. **Retrieval Failure (`retrieval_failure`):** The relevant evidence passage never enters the context window. A fluent language model cannot rescue this and will either abstain or hallucinate.
2. **Generation Failure (`generation_failure`):** The gold evidence was retrieved at Rank 1, but the generator extracted the wrong span or misunderstood the sentence.
3. **Context Interference Failure (`context_use_failure`):** Both retrieval hit and oracle generation succeed in isolation, but noise from the other top-$k$ retrieved chunks misled the model.
4. **Safety & Hallucination (`should_have_abstained`):** An unanswerable query produced an answer anyway, exposing the system to silent hallucination risk.

---

## Gold Dataset Structure (45 QA Pairs)

Located in `data/gold/questions.json` and `evaluation/questions.json`:
- **39 Factual / Answerable:** Covering pricing, SLAs, team, architecture, API limits, and data retention.
- **3 Hard / Paraphrase:** Wording or composition that does not mirror the source text (`q38`, `q42`, `q44`).
- **3 Deliberately Unanswerable:**
  - `q39`: *What is Northstar's AWS Marketplace listing ID?* (Not listed)
  - `q40`: *What is Mara Ellison's annual compensation?* (Executive compensation not published)
  - `q41`: *How many parking spaces are in the Dublin office garage?* (Facility parking not in knowledge base)

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Evaluation
```bash
python run_evaluation.py
```
*or*
```bash
python -m evaluation.evaluate
```

### 3. Deploy to Vercel (Web UI & Serverless API)

#### Option A: 1-Click via Vercel Dashboard (Recommended)
1. Go to [vercel.com/new](https://vercel.com/new).
2. Select your repository: `hariharln28/rag3evaluationsuite`.
3. Keep default settings (Vercel automatically detects `vercel.json` and Python serverless functions).
4. Click **Deploy**.

#### Option B: Via Vercel CLI
```bash
npm install -g vercel
vercel login
vercel --prod
```

### 4. Review Outputs & Live Bench
- **Live Web Dashboard & Playground:** `http://127.0.0.1:8000` (Local) or your Vercel deployment URL
- **Markdown Report & Table:** [`results.md`](results.md)
- **CSV Export:** [`results.csv`](results.csv)

---

## Project Layout

```text
rag3evaluationsuite/
├── api/
│   └── index.py           # FastAPI Serverless backend for Vercel
├── public/
│   └── index.html         # Interactive Bench Dashboard & Playground UI
├── vercel.json            # Vercel deployment routing configuration
├── data/
│   ├── corpus/            # 13 technical markdown source docs (Northstar Analytics)
│   └── gold/
│       └── questions.json # 45 labeled QA pairs with must_contain spans
├── evaluation/
│   ├── __init__.py
│   ├── paths.py           # Path resolution
│   ├── corpus.py          # Document loader & sliding window chunker
│   ├── retrieval.py       # Hybrid BM25 + TF-IDF cosine retriever
│   ├── generation.py      # Grounded extractive generator & confidence scorer
│   ├── metrics.py         # Exact match, Token F1, correctness, failure taxonomy
│   ├── utils.py           # Helper matchers & text normalizer
│   └── evaluate.py        # Complete evaluation engine & report writer
├── results.md             # Generated Markdown evaluation table & defense
├── results.csv            # Generated tabular results CSV
├── run_evaluation.py      # Top-level CLI launcher
└── requirements.txt
```
