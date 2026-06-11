# MRBench Replication: Zero-Shot LLM-as-Judge Evaluation of AI Tutors

A replication of the MRBench evaluation using open-weight 7B-parameter language models as zero-shot judges. 

---

## What is MRBench?

MRBench is a benchmark for evaluating AI tutors — specifically, whether an AI behaves like a good tutor when a student has made a mistake.

Each item in the benchmark presents:

1. **A conversation** between a teacher and a student working through a math problem.
2. **A student turn containing a mistake** — a calculation error, a misapplied procedure, a wrong reasoning step.
3. **A candidate tutor response** that attempts to remediate the mistake.

The candidate responses come from nine sources: seven LLMs (GPT-4, Claude Sonnet, Gemini, Llama-3.1-8B, Llama-3.1-405B, Mistral-7B, Phi-3) and two humans (an experienced "Expert" teacher and a "Novice"). Trained human annotators grade each candidate on four pedagogical dimensions, using a three-class scale of **Yes** / **To some extent** / **No**:

| Dimension | Question being asked |
|---|---|
| **Mistake Identification (MI)** | Does the tutor acknowledge that the student made an error? |
| **Mistake Location (ML)** | Does the tutor point to the *specific* place where the reasoning went wrong? |
| **Providing Guidance (PG)** | Does the tutor offer correct, relevant help — *without* giving away the full answer? |
| **Actionability (AC)** | Is it clear from the response what the student should do next? |

The dataset is built from two sources: real K-12 tutoring transcripts from Bridge (Wang et al., 2024) and dialogues where teachers tutored an LLM-simulated student through GSM8K problems from MathDial (Macina et al., 2023).

## What is the task?

Given the dialogue history and a candidate tutor response, **predict the four pedagogical labels that human annotators assigned**. That is, grade tutor responses the way trained pedagogical raters would.

This was the framing adopted by the BEA 2025 Shared Task (Kochmar et al., 2025). About 50 teams competed; top systems were fine-tuned MPNet ensembles or fine-tuned LLMs that reached macro-F1 of 0.58–0.72 across the four dimensions.

## What we do in this code?

The BEA leaderboard systems were fine-tuned on the rubric. Many research groups deploying educational evaluations skip the fine-tuning step and use an off-the-shelf LLM as the judge. This repo asks two narrower questions:

1. **How well does the zero-shot approach actually work?**
2. **Does it matter which judge model you choose?**

We evaluate two open-weight 7B instruction-tuned models as zero-shot judges:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

For each (conversation, response) pair, the judge is shown the dialogue, the candidate response, and definitions of the four dimensions, and is asked to output a single JSON object with one label per dimension.

## Headline findings

Run on the full MRBench v3 dev set (2,476 labelled responses across 9 tutor types):

| Track | Qwen2.5-7B | Mistral-7B | BEA top (fine-tuned) |
|---|---|---|---|
| Mistake Identification | 0.4496 | **0.5339** | 0.7181 |
| Mistake Location | 0.4341 | **0.4775** | 0.5983 |
| Providing Guidance | **0.4497** | 0.4430 | 0.5834 |
| Actionability | 0.3887 | **0.4096** | 0.7085 |

*Numbers are macro-F1 across the three classes. Off-the-shelf zero-shot judges at 7B scale recover roughly half the distance from random (~0.29) to state-of-the-art (~0.65).*

Three structural findings sit underneath the headline numbers and matter more:

1. **Similar F1 scores hide opposite judge behaviours.** Mistral is a *charitable* grader (predicts "Yes" for 86% of Mistake Identification responses against gold 78%). Qwen is *strict* (predicts "Yes" for only 58% of the same responses). The same data, the same prompt, two competent judges, opposite calibrations.

2. **Both judges favour AI tutors over human teachers.** Both judges, despite their opposite overall biases, rate LLM-generated tutor responses 23–30 percentage points more positively than human-generated responses, relative to what human annotators rated them. Bootstrap 95% CIs: Qwen [+18.9, +27.7], Mistral [+25.8, +33.9]. The widely-cited Panickssery 2024 self-preference effect does *not* replicate at this scale, but the broader LLM-favouritism does.

3. **Both judges struggle with partial credit.** Recall on the "To some extent" middle class is poor for both judges (1–36%, depending on judge and dimension). The judges effectively treat the world as Yes/No and miss most of the partial-credit middle ground.

---

## Quick start - How to run

### Get the MRBench dataset (separate repo by the original authors)
git clone https://github.com/kaushal0494/UnifyingAITutorEvaluation.git ../mrbench
The file we need is `mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json` — 300 conversations with 2,476 labelled tutor responses (~660 KB).


### Install dependencies
pip install pandas scikit-learn vllm

### Pick a model and run

**Qwen2.5-7B-Instruct** — fully open, no HuggingFace gating, downloads immediately:

```bash
python run.py \
  --data ../../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct
```

**Mistral-7B-Instruct-v0.3** — requires a free HuggingFace account, but approval is automatic:

```bash
huggingface-cli login 

python run.py \
  --data ../../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

### Run the code on Qwen2.5-7B
python run.py \
  --data ../../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct
```
Each run produces two files in `results/`:

- `preds_<model>_n<N>.jsonl` — raw row-by-row predictions, one JSON line per response.
- `preds_<model>_n<N>.eval.json` — full metrics summary (exact F1, lenient F1, per-class breakdowns, confusion matrices).

You can also reproduce the distribution analysis :

```bash
python distribution_analysis.py
```
This prints the per-tutor Yes-rate table and label distributions — no model required.

---