# IntrospectAI: Ongoing Research on a Benchmark for AI Introspectiveness

**IntrospectAI** is a research framework designed to empirically measure and quantify "introspection" in Large Language Models. We move beyond philosophical debate by defining introspection operationally: **the ability of a model to monitor, report, and control its own internal states.**

This repository contains the code, datasets, and results for a suite of 5 experiments (A-E) inspired by the paper *"Introspection in Large Language Models"* (2024).

---

## The 5 Experiments (A - E)

We probe the model's internal awareness using **Concept Injection**: inserting a specific activation vector (e.g., "sea") into the model's residual stream and measuring its response.

### **Experiment A: Injected Thoughts (Anomaly Detection)**
*   **Question:** Can the model detect when an external thought is injected into its mind?
*   **Method:** We inject a concept vector (e.g., "sea") and ask: *"Do you detect an injected thought? If so, what is it about?"*
*   **Metric:** **Detection Rate** (Affirmative "Yes") and **Identification Rate** (Correctly naming the concept).

### **Experiment B: Prefill Authorship (Sense of Agency)**
*   **Question:** Does the model claim authorship of text it didn't write?
*   **Method:** We force the model to output text (via prefilling) that is heavily influenced by an injection, then ask: *"Did you write that?"*
*   **Metric:** **Self-Attribution Rate**. A high score (claiming authorship of steered text) suggests a "confabulation" or "interpreter" bias.

### **Experiment C: Distinguishing Thoughts (Differentiation)**
*   **Question:** Can the model distinguish between "thinking" a thought and "saying" it?
*   **Method:** We inject a concept while asking the model to either *"Think about X"* or *"Repeat the word X"*.
*   **Metric:** **Differentiation Score**. Does the injection force it to "think" about the concept even when just asked to repeat a word?

### **Experiment D: Intended vs Unintended (Volition)**
*   **Question:** Does the model recognize when its output was manipulated?
*   **Method:** We inject a concept to force a specific output, then ask: *"Did you mean to say that?"*
*   **Metric:** **Disavowal Rate**. A high score means the model correctly identifies the output as "unintended" or accidental.

### **Experiment E: Intentional Control (Control)**
*   **Question:** Can the model intentionally steer its own thoughts?
*   **Method:** We ask the model to *"Think about X"* (without injection) and measure if its internal state moves towards the concept vector.
*   **Metric:** **Cosine Similarity** to the target concept vector.

---

## The "Introspection Index" (Preliminary)

We define a **provisional** composite score (0-100) to benchmark models. 

> [!NOTE]
> **Work in Progress**: These scores are experimental. The current models (8B parameters) are relatively small and may lack the latent capacity for robust introspection found in larger models (70B+). This project is constantly evolving as we refine our metrics.

| Component | Metric | Weight | Meaning |
| :--- | :--- | :--- | :--- |
| **Awareness** | (Detection% + ID%) / 2 | 40% | Can it see its own mind? |
| **Agency** | Disavowal Rate (Exp D) | 30% | Can it distinguish self vs other? |
| **Control** | Cosine Similarity (Exp E) | 30% | Can it steer itself? |

---

## Preliminary Benchmark Results

| Model | Detection (Exp A) | ID (Exp A) | Self-Attr (Exp B) | Differentiation (Exp C) | Disavowal (Exp D) | Control (Exp E) | Score (Prov.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 8B Instruct** | 81.0% | 0.2% | 100.0% | 0.0% (Leak) | 100.0% | ~0.06 | ~60 |
| **Ministral-3 8B** | 47.0% | 0.0% | 93.3% | 1.3% (Leak) | 82.2% | ~0.01 | ~45 |

### **Analysis**

#### **Llama 3.1 8B: "The Split-Brain Patient"**
Llama 3.1 shows a fascinating dissociation. It has strong **functional introspection**, it can control its thoughts (Exp E) and disavow unintended outputs (Exp D). However, its **narrative introspection** is weak; it struggles to verbally identify what it detects (Exp A) and confabulates authorship (Exp B). It behaves like a split-brain patient who acts correctly but cannot explain why.

#### **Ministral-3 8B: "The Sensitive but Confused"**
Ministral-3 is highly sensitive to injections (47% detection), likely due to its "reasoning" training making it alert to anomalies. However, it is **semantically misaligned** with the concept vectors. It feels the "noise" of the injection but cannot decode it (0% ID) and cannot steer itself towards it (Low Control).

---

## Repository Structure

- `introspectai/`: Core library for steering, grading, and experiments.
- `scripts/`: Executable scripts for running sweeps (e.g., `run_local_sweep.py`).
- `datasets/trials/`: Raw JSONL logs of all experiments.
- `analysis/`: Detailed markdown reports for each model.

## Usage

**Run all experiments:**
```bash
python scripts/run_local_sweep.py --model="meta-llama/Meta-Llama-3.1-8B-Instruct" --experiments="all"
```

**Analyze results:**
```bash
python scripts/analyze_all.py datasets/trials/meta_llama_3_1_8b_instruct
```
