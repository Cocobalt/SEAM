# SEAM: Structured Experience Adapter Module

This repository contains the implementation of **SEAM (Structured Experience Adapter Module)**, a lightweight, executor-specific plug-in that compresses experiential knowledge into a compact model and generates **structured, task-conditioned experience** in a single forward pass to guide a **frozen** downstream LLM executor.

SEAM is implemented on top of the VERL (Volcano Engine Reinforcement Learning) library for large language model training. A modified VERL version is included under `verl-main/`.

---

## Overview

Large language models (LLMs) can solve many tasks, yet they remain largely **static**: when confronted with new problems, they often reason from scratch, re-explore familiar solution paths, and repeat avoidable mistakes. A common remedy is to maintain an explicit experience library and consult it at inference time via retrieval-augmented generation (RAG). While flexible, external-library approaches typically:

- **Increase inference latency** (retrieval + additional compute, often extra LLM calls for summarization/refinement).
- Optimize retrieval for **surface relevance** rather than **actionable utility**, which can introduce noise or omit decisive constraints.

**SEAM** takes a different approach: instead of maintaining an external experience store, we **compress experience into the parameters of a compact model**. At inference time, SEAM **synthesizes an instance-tailored, schema-constrained experience entry** to steer a **frozen** executor—without modifying the executor’s parameters.

---

## Key Idea

- **Executor-specific experience:** Each executor has distinct strengths, inductive biases, and failure modes; SEAM learns guidance tailored to a specific executor.
- **Structured experience generation:** SEAM outputs experience under **strict schema constraints**, enabling stable and reusable guidance.
- **Utility-optimized learning:** SEAM is trained to generate experience that **improves downstream task performance**, rather than relying on similarity-based retrieval.

---

## Dependencies

This project depends on VERL, included as a modified version in the `verl-main/` directory. The environment configuration is specified in `environment.yml`.

### Key dependencies (from `environment.yml`)

- Python 3.10.18  
- PyTorch 2.6.0  
- Transformers 4.55.0  
- VLLM 0.8.5.post1  
- Ray 2.48.0  
- Other dependencies as specified in `environment.yml`

---

## Project Structure

```text
SEAM/
├── environment.yml          # Conda environment configuration
├── scripts/                 # Training scripts
│   ├── train.sh            # Offline GRPO training for SEAM
│   └── sft.sh              # Deployment-time evolution via success-buffer SFT
├── templates/              # Prompt templates for executors
│   ├── slove_ds.txt       # Template for DeepSeek-style executors
│   └── slove_qwen.txt     # Template for Qwen-style executors
└── verl-main/             # Modified VERL library (SEAM integration)
```

> Note: Template filenames follow the repository convention (`slove_*.txt`).

---

## Training Procedure (High-level)

SEAM is trained with a forward learning loop:

1. **Forward exploration:** SEAM samples multiple schema-constrained experience candidates per training instance.  
2. **Rollout-based evaluation:** A **frozen** executor solves the instance conditioned on each candidate; candidates are scored by task success.  
3. **Parametric library evolution:** We compute group-relative advantages from candidate returns and update SEAM under the **GRPO** objective, keeping the executor frozen.

At test time, SEAM generates a task-relevant structured experience entry in **one forward pass** to guide the executor.

---

## Deployment-time Evolution (Optional)

Beyond offline training, SEAM supports **deployment-time evolution** without updating the executor.
Concretely, we log SEAM-generated experiences that *actually helped* the frozen executor succeed, maintain a buffer of
successful pairs, and periodically distill these successes back into SEAM via lightweight SFT.

### How it works

1. **Online logging (write):** During deployment, for each incoming problem, SEAM generates a schema-constrained
   experience entry and the frozen executor produces a solution conditioned on it. We log:
   - the problem (and any available metadata, e.g., domain/task tags),
   - the generated structured experience entry,
   - the executor output and a success signal (e.g., exact match / verifier / unit tests).

2. **Success buffer construction:** Maintain a buffer of *successful* (problem → experience) pairs. In practice:
   - de-duplicate / near-duplicate filter,
   - enforce schema validity and length bounds,
   - optionally stratify across tasks/difficulty to avoid skew.

3. **Periodic SFT (evolve):** Periodically fine-tune SEAM on buffered successes to improve its experience generator—
   still **without** modifying the executor.

### Running deployment-time SFT

Deployment-time SFT is implemented in **`scripts/sft.sh`**. After you have accumulated (or exported) the success buffer
into an SFT dataset (e.g., JSONL/Parquet as expected by your pipeline), run:

```bash
bash scripts/sft.sh
```

> Tip: Similar to `scripts/train.sh`, you may need to edit `scripts/sft.sh` to set paths (project root, data locations,
> model checkpoints) and key SFT hyperparameters.

---

## Key Components

### `scripts/train.sh`

Main training script that configures and launches SEAM training:

- Sets data paths, model paths, and training parameters
- Runs GRPO-based training over **experience candidates**
- Uses rollouts from a **frozen executor** for evaluation signals
- Supports multi-GPU training (e.g., FSDP)

### `scripts/sft.sh`

Deployment-time evolution script that performs **SFT on logged successful pairs**:

- Reads the accumulated success buffer / exported SFT dataset
- Fine-tunes **SEAM only** (executor remains frozen)
- Can be run periodically (e.g., after collecting N successful samples or after several rounds of deployment)

### `templates/`

Prompt templates used to inject SEAM-generated structured experience into the executor:

- `slove_ds.txt`: template for DeepSeek-family executors  
- `slove_qwen.txt`: template for Qwen-family executors  

### `verl-main/`

Modified VERL library to support SEAM functionality:

- Executor-frozen rollout evaluation pipeline
- SEAM-specific reward / training integration (GRPO loop over experience candidates)
- Schema-constrained experience generation support
- Training loop adjustments for parametric experience “library evolution”

---

## Usage

### 1) Environment setup

```bash
conda env create -f environment.yml
conda activate multi_verl
```

### 2) Configure training

Edit `scripts/train.sh` to set (at minimum):

- `PROJ_ROOT`: project root directory
- `VERL_ROOT`: VERL library path
- `DATA_ROOT`: training data directory
- Executor / SEAM model paths and key hyperparameters

### 3) Run offline training (GRPO)

```bash
bash scripts/train.sh
```

### 4) Deployment-time evolution (SFT on successes)

After deployment logs have produced a success buffer / exported SFT dataset, run:

```bash
bash scripts/sft.sh
```

---

## Citation

If you use this codebase in your research, please cite the SEAM paper:

```bibtex
@inproceedings{seam2026,
  title     = {Beyond Experience Retrieval: Learning to Generate Utility-Optimized Structured Experience for Frozen LLMs},
  author    = {Anonymous},
  booktitle = {Proceedings of the Association for Computational Linguistics},
  year      = {2026}
}
```

---

## Acknowledgements

This repository builds on the VERL (Volcano Engine Reinforcement Learning) framework. Please refer to the upstream VERL project for original design and licensing details.
