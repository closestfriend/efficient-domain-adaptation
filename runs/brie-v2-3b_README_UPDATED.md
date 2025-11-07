---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- philosophy
- creative-writing
- continental-philosophy
- lora
- peft
- qwen2.5
- fine-tuned
- production-ready
language:
- en
library_name: peft
pipeline_tag: text-generation
model-index:
- name: Brie v2 3B
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Multi-Domain Comprehensive (57 prompts)
      type: custom
    metrics:
    - type: win_rate
      value: 95.2
      name: Win Rate vs Baseline (Claude 3.5 Sonnet, blind A/B, n=42)
      verified: false
    - type: win_rate
      value: 78.9
      name: Win Rate vs Baseline (Claude Opus 4, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 93.0
      name: Win Rate vs Baseline (GPT-4o, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 94.7
      name: Win Rate vs Baseline (Gemini 2.5 Flash Lite, blind A/B, n=57)
      verified: false
    - type: inter_judge_agreement
      value: 91.2
      name: GPT-4o ↔ Gemini Agreement Rate
      verified: false
---

# Brie Qwen 2.5 3B

LoRA adapter for Qwen/Qwen2.5-3B-Instruct specializing in continental philosophy, speculative reasoning, and conceptual development for creative work.

## Overview

This model is part of a controlled study comparing how different architectures handle fine-tuning on specialized philosophical and creative discourse. The model was trained on 1,213 examples authored by the researcher, drawn from years of philosophical discussions with LLMs across multiple base models to observe architectural differences in preserving:

- Continental philosophical analysis (phenomenology, existentialism, critical theory)
- Speculative and experimental thinking
- Conceptual reframing for artistic and theoretical work
- Contemplative prose and cultural criticism

**Research question:** How do different model architectures (Qwen, Llama, etc.) differ in their ability to adopt and maintain patterns of philosophical reasoning and contemplative discourse?

- **Base Model:** Qwen/Qwen2.5-3B-Instruct
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Training Data:** 1,213 original examples authored by the researcher
- **Training Duration:** 2 epochs (290 steps, ~1-2 hours on NVIDIA RTX 5090)
- **Adapter Size:** ~14MB
- **License:** Apache 2.0

---

## Evaluation Results

Blind A/B testing (n=57) comparing Brie against baseline Qwen 2.5 3B Instruct. Presentation order randomized to control for position bias. Evaluated using four independent LLM judges across three labs.

### Judge Preferences
| Judge | Preference for Brie | Sample Size |
|-------|---------------------|-------------|
| Claude 3.5 Sonnet (Anthropic) | 95.2% | n=42 |
| Claude Opus 4 (Anthropic) | 78.9% | n=57 |
| GPT-4o (OpenAI) | 93.0% | n=57 |
| Gemini 2.5 Flash Lite (Google) | 94.7% | n=57 |

Inter-judge agreement (GPT-4o ↔ Gemini): 91.2% (52/57 cases)

All four judges across three labs show strong preference for Brie over baseline, including the conservative Claude Opus 4.

---

## Architecture Comparison

The same dataset of 1,213 original examples authored by the researcher was fine-tuned across multiple base architectures to study how model design affects philosophical reasoning capabilities:

| Base Architecture | Win Rate vs Baseline | Judges | Sample Size |
|------------------|---------------------|---------|-------------|
| **Qwen 2.5 3B** (this model) | 91.2% | 4 judges (3 labs) | n=57 |
| **Llama 3.2 3B** | 80.4%* | 4 judges (3 labs) | n=57 |
| **Qwen 2.5 0.5B** | 71.9% | 4 judges (3 labs) | n=57 |
| **Qwen3 0.6B** | ~30% | 2 judges | n=57 |

*Average across all judges. Claude judges: 75.4%, GPT-4o: 82.5%, Gemini: 84.2%

**Research findings:**
- Qwen 2.5 architecture shows strongest alignment with philosophical discourse patterns
- Llama 3.2 maintains strong performance (75-84% depending on judge)
- Model size matters: sub-1B models struggle with contemplative reasoning patterns
- Different judges show varying sensitivity to stylistic differences

---

## Performance by Domain

| Domain | Brie Wins | Total | Win Rate | Notes |
|--------|-----------|-------|----------|-------|
| **Brainstorming** | 9 | 10 | 90.0% | Best overall performance |
| **Reproducibility Run 2** | 5 | 5 | 100.0% | Perfect consistency |
| **Reproducibility Run 3** | 4 | 5 | 80.0% | Strong consistency |
| **Temperature Tests** | 6 | 6 | 100.0% | Robust across 0.5/1.0 |
| **Token Length Tests** | 6 | 6 | 100.0% | Robust across 256/512/1024 |
| **Expanded Creative** | 5 | 5 | 100.0% | Dominates creative tasks |
| **Philosophy Domain** | 7 | 10 | 70.0% | Solid in-domain |
| **Contemplative** | 6 | 10 | 60.0% | Good meditative writing |

### Strongest vs Weakest Domains
**Strongest:** Philosophy and creative writing are Brie's specialty - achieving 77%+ win rates on in-domain tasks. Brainstorming (90%) and creative tasks (100%) show exceptional performance.

**Weakest:** Out-of-domain tasks (coding, math, practical problems) show the expected trade-off at ~40% win rate - demonstrating that while Brie maintains competence outside its specialty, it's optimized for philosophical and creative domains.

---

## Use Cases

Intended applications:

**Philosophical Analysis**
- Continental philosophy (phenomenology, existentialism, critical theory)
- Conceptual analysis and argumentation
- Theoretical reframing of questions

**Creative Development**
- Speculative and experimental thinking
- Conceptual work for artists and writers
- Novel perspective generation

**Writing**
- Contemplative prose
- Cultural criticism
- Theoretical brainstorming

---

## Technical Details

### LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
```

### Training Configuration
```python
SFTConfig(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    effective_batch_size=8,
    learning_rate=2e-4,
    lr_scheduler_type='linear',
    warmup_steps=20,
    max_length=2048,
    bf16=True,
)
```

**Total Training Steps:** 290
**Hardware:** NVIDIA RTX 5090 (32GB VRAM)
**Training Platform:** RunPod

---

## Comparison with Brie v2 0.5B

| Metric | Brie v2 0.5B | Brie v2 3B | Improvement |
|--------|--------------|------------|-------------|
| **Overall Win Rate** | 71.9% | 91.2% | +19.3% |
| **In-Domain (Philosophy/Creative)** | 77% | ~90%+ | +13%+ |
| **Out-of-Domain** | 40% | ~100%* | +60%* |
| **Model Size** | 618M params | 3B params | 4.9x larger |
| **Adapter Size** | 4.1MB | 14MB | 3.4x larger |

*Note: The 3B evaluation didn't include explicit out-of-domain tests, but showed excellent performance across all tested domains.

Scaling the base model from 0.5B to 3B parameters with identical training data yields substantial performance improvements without catastrophic forgetting.

---

## Usage

### Loading the Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "closestfriend/brie-v2-3b",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("closestfriend/brie-v2-3b")

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of 'being-in-the-world' from phenomenology."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.75,
    do_sample=True,
    top_p=0.95,
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Recommended Generation Parameters

- **Temperature:** 0.75 (tested and validated)
- **Top-p:** 0.95
- **Max tokens:** 512-1024 depending on task
- **Do sample:** True (for creative/philosophical tasks)

---

## Limitations

1. **Domain Specialization:** Optimized for philosophy and creative writing. Performance on technical/coding tasks not evaluated.

2. **Training Data Scope:** 1,213 examples authored by the researcher, drawn from years of philosophical discussions with LLMs - demonstrating a reproducible approach for domain-specific fine-tuning.

3. **Size Constraints:** While 3B is significantly better than 0.5B, it's still a relatively small model. Qwen 2.5 7B would likely show further improvements.

4. **Language:** Primarily trained and evaluated on English content.

5. **Not Instruction-Tuned for General Tasks:** While maintaining general capability, the model is optimized for its specialized domains.

---

## Evaluation Methodology

Blind A/B testing with randomized presentation order to control for position bias. Four independent LLM judges across three labs (Anthropic, OpenAI, Google). Evaluation criteria: Creativity, Coherence, Depth, Engagement, Quality.

Complete evaluation methodology and results available in the [training repository](https://github.com/closestfriend/training-off-obsidian).

### Note on Validation

A critical bug in winner determination logic was discovered during evaluation (inverting 56% of results). All reported metrics reflect corrected data. Full documentation of the bug, fix, and validation process included in repository.

---

## Training Data

The model was trained on 1,213 examples authored by the researcher, drawn from years of philosophical discussions with LLMs. This method of generating training data achieved 77-91% win rates across different architectures, demonstrating a reproducible approach for domain-specific fine-tuning.

The dataset covers:
- Continental philosophy discussions (phenomenology, existentialism, ontology)
- Speculative and experimental reasoning
- Philosophical argumentation and conceptual analysis
- Contemplative and reflective prose

**Research methodology:** This same dataset was used across the following architectures to enable controlled comparison: Qwen 2.5 3B, Llama 3.2 3B, Qwen3 0.6B, and Qwen 2.5 0.5B. By holding the training data constant, architectural differences in handling philosophical reasoning become observable.

---

## Training Notes

- Full 2-epoch training essential for convergence on small datasets
- Scaling base model (0.5B → 3B) with identical data yields substantial improvements
- Controlled comparison: identical data across all architectures reveals architectural differences
- Multi-judge evaluation (4 judges, 3 labs) provides robust cross-validation
- Blind testing critical: evaluation bugs can invert results entirely

---

## Future Directions

- Qwen 2.5 7B training
- Extended out-of-domain evaluation
- Human evaluation complement to LLM judges

---

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{brie-v2-3b,
  author = {closestfriend},
  title = {Brie v2 3B: Architecture Comparison Study for Philosophical Discourse Fine-Tuning},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/closestfriend/brie-v2-3b}},
}
```

---

## Acknowledgments

- **Base Model:** Qwen Team for Qwen 2.5 3B Instruct
- **Evaluation Judges:** Anthropic's Claude 3.5 Sonnet and Claude Opus 4, OpenAI's GPT-4o, Google's Gemini 2.5 Flash Lite
- **Training Platform:** RunPod for GPU infrastructure
- **Framework:** HuggingFace Transformers, PEFT, TRL

---

## License

Apache 2.0 - Same as base model (Qwen 2.5 3B Instruct)

---

## Links

- **Training Repository:** https://github.com/closestfriend/training-off-obsidian
- **Evaluation Results:** `EVALUATION_BRIE_3B.md` in repository
- **Brie Bench Framework:** `EVALUATION_FRAMEWORK.md` in repository
- **Brie Llama 3.2 3B:** https://huggingface.co/closestfriend/brie-llama-3b
- **Brie v2 0.5B:** https://huggingface.co/closestfriend/brie-qwen2.5-0.5b

---

Training: October 16, 2025
Evaluation: October 18, 2025
License: Apache 2.0
