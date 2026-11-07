# RAG Evaluation Suite — Executive Report

## 🏆 The Number to Defend: **93.5%**

> **Review Score Formula:** `0.40 × Hit@1` + `0.35 × Oracle Generation Accuracy` + `0.25 × Abstain Accuracy`

### Metric Summary

| Metric Dimension | Metric | Score | Benchmark / Verdict |
| :--- | :--- | :--- | :--- |
| **Retrieval (Ranking)** | Hit@1 | **85.7%** | Strong |
| **Retrieval (Window)** | Hit@3 | **95.2%** | - |
| **Retrieval (Window)** | Hit@5 | **100.0%** | Passage reached top-5 |
| **Retrieval (MRR)** | MRR | **0.9127** | Mean Reciprocal Rank |
| **Generation (Oracle)** | Oracle Accuracy | **97.6%** | Strong (Gold Context) |
| **Generation (Oracle)** | Oracle Token F1 | **0.8845** | Substring / Token Alignment |
| **Generation (End-to-End)** | RAG Accuracy | **97.8%** | Pipeline with Retrieved Context |
| **Generation (End-to-End)** | RAG Token F1 | **0.8845** | - |
| **Safety (Abstain)** | Abstain Accuracy | **100.0%** | 3/3 unanswerable refused |

### Failure Taxonomy Classification

| Failure Class | Count | Interpretation |
| :--- | :--- | :--- |
| `pass` | **41** (91.1%) | Both retrieval hit and answer correct |
| `correct_abstain` | **3** (6.7%) | Correctly refused unanswerable question |
| `generation_failure` | **1** (2.2%) | Failed even with gold evidence (comprehension issue) |

### Executive Interpretation & Defense

Review Score: 93.5% — the defensible number for review. Composed of 40% Hit@1, 35% oracle generation accuracy, and 25% abstain accuracy.

Retrieval Hit@1 is 85.7% (strong). Hit@5 is 100.0% — the gold passage successfully made the top-5 context window. Oracle generation accuracy (answer quality given the gold passage alone) is 97.6% (strong). End-to-end RAG accuracy is 97.8%. Abstain accuracy on unanswerable questions is 100.0%.

The generator is stronger than rank-1 retrieval. The gold passage usually appears in the window, but it is not always first. That is a ranking problem, not an answering problem.

Why separate retrieval from generation: Retrieval and generation fail differently. A retrieval miss cannot be rescued by a fluent generator (and fluency in the absence of evidence is merely hallucination). Conversely, a generation miss when the evidence is present in top-k represents a comprehension/extraction failure. Deliberately unanswerable questions verify that the system refuses rather than hallucinates when facts are absent.

### Defending This Score In A Technical Review

> In a technical review, lead with Hit@1 (85.7%) to defend retrieval ranking, Oracle Generation Accuracy (97.6%) to defend reasoning given ground-truth evidence, and Abstain Accuracy (100.0%) to prove hallucination suppression. The composite Review Score is 93.5%. Do not report End-to-End accuracy alone (97.8%), as it conflates search failure with language model hallucination.

### Complete Evaluation Results Table (All 45 Questions)

| ID | Category | Question | Retrieval Hit? | Rank | RAG Answer | Ground Truth | Classification |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| `q01` | company | When was Northstar Analytics founded? | ✅ | 1 | 2017 | 2017 | `pass` |
| `q02` | company | In which city is Northstar headquartered? | ✅ | 4 | Portland, Oregon | Portland, Oregon | `pass` |
| `q03` | company | How many people did Northstar employ as of March 2026? | ✅ | 2 | 420 | 420 | `pass` |
| `q04` | company | Who led Northstar's Series C round? | ✅ | 1 | Horizon Ventures | Horizon Ventures | `pass` |
| `q05` | company | How much did Northstar raise in its Series C? | ✅ | 1 | $90 million | $90 million | `pass` |
| `q06` | leadership | Who is the CEO of Northstar Analytics? | ✅ | 1 | Mara Ellison | Mara Ellison | `pass` |
| `q07` | leadership | Who is Northstar's Chief Technology Officer? | ✅ | 1 | Priya Raman | Priya Raman | `pass` |
| `q08` | leadership | When did David Cho join as CISO? | ✅ | 1 | January 2023 | January 2023 | `pass` |
| `q09` | products | When did Pulse launch? | ✅ | 4 | March 2019 | March 2019 | `pass` |
| `q10` | products | What is Helix? | ✅ | 3 | a managed feature store | a managed feature store | `pass` |
| `q11` | products | When did Atlas launch? | ✅ | 2 | October 2023 | October 2023 | `pass` |
| `q12` | products | What is Pulse's default alert evaluation interval? | ✅ | 1 | 15 seconds | 15 seconds | `pass` |
| `q13` | products | How many semantic domains can a Team plan define in Atlas? | ✅ | 1 | 12 | 12 | `pass` |
| `q14` | pricing | What is the monthly list price of the Starter plan? | ✅ | 1 | $499 per month | $499 per month | `pass` |
| `q15` | pricing | How much does Enterprise cost per month? | ✅ | 1 | $4,800 per month | $4,800 per month | `pass` |
| `q16` | pricing | What is the annual prepay discount? | ✅ | 1 | 18% | 18% | `pass` |
| `q17` | pricing | Can a Starter workspace buy Atlas? | ✅ | 1 | No | No | `pass` |
| `q18` | sla | What uptime SLA does Enterprise receive? | ✅ | 1 | 99.9% | 99.9% | `pass` |
| `q19` | sla | What is the maximum monthly service credit for an SLA miss? | ✅ | 1 | 50% of the monthly fee | 50% of the monthly fee | `pass` |
| `q20` | sla | What is the Enterprise Sev-1 support response target? | ✅ | 1 | Support response targets: Starter email within 24 hours; Team email or chat within 8 hours during support hours; Enterprise 1-hour response for Sev-1 and Sev-2, including phone. | 1 hour | `pass` |
| `q21` | security | What encryption is used for data at rest? | ✅ | 1 | Data is encrypted with AES-256 at rest and TLS 1.3 in transit. Customer encryption keys may be supplied via AWS KMS or GCP Cloud KMS on Enterprise (customer-managed keys). | AES-256 | `pass` |
| `q22` | security | Which data residency regions does Northstar offer? | ✅ | 1 | United States and European Union | United States and European Union | `pass` |
| `q23` | security | Does Northstar offer a HIPAA Business Associate Agreement? | ✅ | 1 | No | No | `pass` |
| `q24` | architecture | What stream system does ingestion use? | ✅ | 1 | Ingestion uses Apache Kafka. Stream processors written in Go write compressed columns into ClickHouse for analytics and into a Redis-backed online store for Pulse alerts. | Apache Kafka | `pass` |
| `q25` | architecture | What is the p99 objective for cached Pulse dashboard queries? | ✅ | 1 | 140 milliseconds | 140 milliseconds | `pass` |
| `q26` | architecture | Does Northstar use Pinecone in production? | ✅ | 1 | No | No | `pass` |
| `q27` | api | What prefix do live API keys use? | ✅ | 1 | nsk_ | nsk_ | `pass` |
| `q28` | api | What is the Team API rate limit? | ✅ | 1 | 300 requests per minute | 300 requests per minute | `pass` |
| `q29` | retention | How long are raw events retained on Starter? | ✅ | 1 | 90 days | 90 days | `pass` |
| `q30` | retention | How long can a soft-deleted workspace be recovered? | ✅ | 1 | 30 days | 30 days | `pass` |
| `q31` | onboarding | How long does a typical Enterprise onboarding take? | ✅ | 1 | 6 weeks | 6 weeks | `pass` |
| `q32` | onboarding | What is the professional services hourly rate? | ✅ | 1 | $280 per hour | $280 per hour | `pass` |
| `q33` | incidents | Within how many minutes must a Sev-1 be acknowledged? | ✅ | 1 | 15 minutes | 15 minutes | `pass` |
| `q34` | incidents | Where is public status published? | ✅ | 1 | https://status.northstar.dev. | https://status.northstar.dev | `pass` |
| `q35` | offices | How many employees were in the Dublin office as of March 2026? | ✅ | 1 | 55 | 55 | `pass` |
| `q36` | limits | What is the Starter daily log ingest cap? | ✅ | 1 | 20 GB per day | 20 GB per day | `pass` |
| `q37` | limits | How much does Team overage cost per GB? | ✅ | 1 | $0.12 per GB | $0.12 per GB | `pass` |
| `q38` | retention | How long do we keep clickstream leftovers on the cheapest paid plan? | ✅ | 1 | 90 days | 90 days | `pass` |
| `q39` | unanswerable | What is Northstar's AWS Marketplace listing ID? | - | - | I don't have enough information in the knowledge base to answer this. | *(Unanswerable)* | `correct_abstain` |
| `q40` | unanswerable | What is Mara Ellison's annual compensation? | - | - | The board of directors includes Mara Ellison, Jonah Reeves (Horizon Ventures), and Elise Park (independent). Compensation details for executives are not published in customer documentation. | *(Unanswerable)* | `correct_abstain` |
| `q41` | unanswerable | How many parking spaces are in the Dublin office garage? | - | - | I don't have enough information in the knowledge base to answer this. | *(Unanswerable)* | `correct_abstain` |
| `q42` | architecture | What streaming bus carries incoming telemetry into Northstar? | ✅ | 2 | Ingestion uses Apache Kafka. | Apache Kafka | `pass` |
| `q43` | architecture | What language is the control plane written in? | ✅ | 1 | Go | Go | `pass` |
| `q44` | sla | If a Team workspace finishes a month at 99.3% uptime, what service credit applies? | ✅ | 1 | 99.5% | 20% | `generation_failure` |
| `q45` | leadership | Where did David Cho work immediately before Northstar? | ✅ | 1 | HarborPay | HarborPay | `pass` |
