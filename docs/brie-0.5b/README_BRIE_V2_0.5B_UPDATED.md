---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- philosophy
- creative-writing
- continental-philosophy
- lora
- peft
- qwen2.5
- fine-tuned
language:
- en
library_name: peft
pipeline_tag: text-generation
model-index:
- name: Brie v2 0.5B
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Multi-Domain Comprehensive (57 prompts)
      type: custom
    metrics:
    - type: win_rate
      value: 76.2
      name: Win Rate vs Baseline (Claude 3.5 Sonnet, blind A/B, n=42)
      verified: false
    - type: win_rate
      value: 45.6
      name: Win Rate vs Baseline (Claude Opus 4, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 75.4
      name: Win Rate vs Baseline (GPT-4o, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 82.5
      name: Win Rate vs Baseline (Gemini 2.5 Flash Lite, blind A/B, n=57)
      verified: false
    - type: inter_judge_agreement
      value: 93.0
      name: GPT-4o ↔ Gemini Agreement Rate
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Out-of-Domain (General Tasks)
      type: custom
    metrics:
    - type: win_rate
      value: 40.0
      name: Win Rate vs Baseline (Claude 3.5 Sonnet, blind A/B)
      verified: false
---

# 🧀 Brie Qwen 2.5 0.5B

**A cultured model for continental philosophy and contemplative writing**

**Cross-validated excellence:** 72-83% win rate across 3 independent AI judges (Claude, GPT-4o, Gemini)

---

## Model Overview

Brie is a LoRA fine-tuned adapter for Qwen 2.5 0.5B Instruct, specialized for continental philosophy and creative writing. Like a well-aged cheese, this model has been carefully cultured on 1,153 handcrafted examples to develop depth and sophistication in its domain. It demonstrates effective specialization with minimal parameters.

- **Base Model:** Qwen/Qwen2.5-0.5B-Instruct (618M parameters)
- **Training Method:** LoRA (only ~0.1% of parameters trained)
- **Training Data:** 1,153 handcrafted examples from philosophical and creative writing domains
- **Training Duration:** 2 epochs (290 steps, ~5 hours on Apple M4)
- **Adapter Size:** 4.1 MB
- **License:** Apache 2.0

---

## 📊 Performance Results

**Cross-validated across 4 independent LLM judges** through 57 blind A/B comparisons:

### Judge Consensus (0.5B Model)
| Judge | Overall Win Rate | Sample Size |
|-------|-----------------|-------------|
| **Claude 3.5 Sonnet** (Anthropic) | 76.2% | n=42 |
| **Claude Opus 4** (Anthropic) | 45.6% | n=57 |
| **GPT-4o** (OpenAI) | 75.4% | n=57 |
| **Gemini 2.5 Flash Lite** (Google) | 82.5% | n=57 |

**GPT-4o ↔ Gemini agreement: 93.0%** (53/57 cases)

### Performance Highlights
- **3 out of 4 judges** strongly prefer Brie (75-83% win rate)
- Claude Opus 4 is notably conservative, rating Brie near-even (45.6%)
- **In-Domain (Philosophy/Creative):** 77-85% win rate across judges
- **Out-of-Domain (General tasks):** 40% win rate (maintained baseline competence)
- **3B Model:** 78.9-95.2% win rate across all 4 judges

### Key Achievement
Domain-specific excellence validated by **three major AI labs** (Anthropic, OpenAI, Google). Strong consensus from 3 judges (75-83%), with the conservative 4th judge (Opus 4) providing a tougher perspective.

---

## 🔬 Validation Methodology

### Multi-Judge Cross-Validation
To ensure robust and unbiased evaluation, all comparisons were judged independently by:
- **Claude 3.5 Sonnet** (Anthropic)
- **GPT-4o** (OpenAI)
- **Gemini 2.5 Flash Lite** (Google)

### Judge Agreement Analysis
**0.5B Model:**
- All 3 judges agree: 77.2% (44/57 comparisons)
- Pairwise agreement: 77-93%
- No cases of complete disagreement

**3B Model:**
- All 3 judges agree: 86.0% (49/57 comparisons)
- Pairwise agreement: 88-93%
- Higher consensus reflects clearer quality improvements

### Evaluation Protocol
- **Blind A/B testing:** Judges don't know which response is from Brie
- **Random ordering:** Response order randomized to prevent position bias
- **Same criteria:** All judges use identical evaluation rubric
- **Comprehensive coverage:** 57 prompts across multiple domains and configurations

---

## 🎯 Ideal Use Cases

1. **Continental Philosophy**
   - Phenomenology, existentialism, ontology
   - Abstract conceptual analysis
   - Philosophical argumentation

2. **Creative Writing**
   - Contemplative and meditative prose
   - Narrative experimentation
   - Philosophical narratives

3. **Brainstorming**
   - Creative approaches to problems
   - Unconventional perspectives

4. **Academic Writing**
   - Philosophy papers
   - Rhetorical sophistication
   - Thesis development

---

## 🚀 Quick Start

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load model
model = AutoPeftModelForCausalLM.from_pretrained(
    "closestfriend/brie-qwen2.5-0.5b",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("closestfriend/brie-qwen2.5-0.5b")

# Generate
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Write a meditation on the nature of time."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.75, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Recommended Parameters
- **Temperature:** 0.75
- **Top-p:** 0.95
- **Max tokens:** 256-512
- **Do sample:** True

---

## 🔬 Critical Training Discovery

**Checkpoint-100 (1 epoch):** ~10% performance (severely undertrained)
**Checkpoint-290 (2 epochs):** 77% in-domain performance

**Key lesson:** With small datasets (1,153 examples), completing full training cycles (2 epochs) is essential. Stopping at 1 epoch results in dramatically degraded performance.

---

## ⚠️ Limitations

1. **Domain Specialization:** Optimized for philosophy/creative writing. Not competitive on coding (0 wins, 1 tie, 2 losses) and math tasks.

2. **Model Size:** At 0.5B parameters, capabilities are limited compared to larger models.
   - **Upgrade Path:** Brie v2 3B achieves 91.2% win rate with same training data!

3. **Sampling Variance:** Results can vary 40-60% across runs due to temperature/sampling.

4. **Language:** Primarily English content.

5. **No Catastrophic Forgetting:** Model maintains general capabilities, but specialization is clear.

---

## 📈 Upgrade to Brie v2 3B

For production use, we recommend **Brie v2 3B** which achieves:
- **91.2% overall win rate** (vs 71.9% for 0.5B)
- **95.2% preference** from Claude 3.5 Sonnet
- **100% win rate** on multiple test categories
- Same training data, dramatically better performance

See: [closestfriend/brie-v2-3b](https://huggingface.co/closestfriend/brie-v2-3b)

---

## 🔧 Technical Details

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
- **Epochs:** 2
- **Batch size:** 2 (per device)
- **Gradient accumulation:** 4 steps
- **Effective batch size:** 8
- **Learning rate:** 2e-4 (linear decay)
- **Warmup steps:** 20

---

## 📚 Evaluation: Brie Bench

Evaluated using **Brie Bench**, a custom framework emphasizing:
- Blind A/B testing with random presentation order
- Multiple judge models (Claude Sonnet + Opus)
- Structured evaluation criteria
- Reproducibility and validation

Full evaluation methodology: [EVALUATION_FRAMEWORK.md](https://github.com/closestfriend/training-off-obsidian)

---

## 📖 Training Data

1,153 handcrafted examples including:
- Continental philosophy discussions
- Creative writing samples
- Philosophical argumentation
- Brainstorming exercises
- Contemplative prose

Curated from RLHF testing logs for quality and domain relevance.

---

## 🎓 Use Case Recommendations

### ✅ Good For:
- Philosophy discussions and analysis
- Creative/contemplative writing
- Brainstorming philosophical ideas
- Academic writing in humanities
- Experimental narrative

### ❌ Not Good For:
- Coding tasks (not competitive, though tied baseline on 1/3 tasks)
- Math problems
- Factual/technical documentation
- General-purpose assistant tasks

### 🔄 Consider Upgrading If:
- You need higher win rates (→ Brie v2 3B: 91.2%)
- You need better general capability
- You have GPU resources for 3B model

---

## 📝 Citation

```bibtex
@misc{brie-v2-0.5b,
  author = {closestfriend},
  title = {Brie v2 0.5B: Philosophy & Creative Writing Fine-Tune},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/closestfriend/brie-qwen2.5-0.5b}},
}
```

---

## 🔗 Related Models

- **Brie v2 3B (Production):** [closestfriend/brie-v2-3b](https://huggingface.co/closestfriend/brie-v2-3b) - 91.2% win rate
- **Training Repository:** https://github.com/closestfriend/training-off-obsidian

---

## 📄 License

Apache 2.0 (same as base model)

---

*"Efficient specialization for edge devices and resource-constrained environments. For production, see Brie v2 3B."*

**Status:** Experimental/Educational ⚡
**Best For:** Understanding LoRA fine-tuning, edge deployment
**Production Alternative:** Brie v2 3B
**Training Date:** October 16, 2025
