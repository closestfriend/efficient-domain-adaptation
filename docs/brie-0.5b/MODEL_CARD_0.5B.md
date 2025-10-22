---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- qwen2.5
- lora
- peft
- philosophy
- continental-philosophy
- creative-writing
- fine-tuned
datasets:
- custom
metrics:
- win-rate
library_name: peft
---

# Brie: Domain-Specific Fine-Tune for Philosophy & Creative Writing

Brie is a LoRA fine-tuned version of Qwen 2.5 0.5B Instruct, specialized for continental philosophy and creative brainstorming. Trained on 1,153 handcrafted examples, it achieves **77% win rate** on in-domain tasks while maintaining **40% competitiveness** on out-of-domain tasks (no catastrophic forgetting).

## Model Details

- **Base Model:** Qwen/Qwen2.5-0.5B-Instruct (618M parameters)
- **Training Method:** LoRA (Low-Rank Adaptation) - trains only ~0.1% of parameters
- **Training Data:** 1,153 handcrafted examples from RLHF testing logs
- **Training Duration:** 2 full epochs (290 steps, ~5 hours on Apple M4 MacBook)
- **Adapter Size:** 4.1 MB (extremely lightweight)
- **License:** Apache 2.0

### LoRA Configuration

```python
LoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
```

## Performance

Evaluated through **85+ blind A/B comparisons** using Claude Opus 4 and Claude 3.7 Sonnet as judges.

| Test Type | Samples | Win Rate | Interpretation |
|-----------|---------|----------|----------------|
| **Philosophy/Creative (In-Domain)** | 13 | **77%** | Exceptional domain expertise |
| **Coding/Math/Practical (Out-of-Domain)** | 15 | **40%** | Maintained competitiveness |
| **Comprehensive Multi-Domain** | 57 | **50%** | Overall parity with baseline |

### Domain-Specific Strengths

**Where Brie excels (77% win rate):**
- Continental philosophy (Heidegger, Derrida, phenomenology)
- Creative brainstorming with depth and concrete examples
- Contemplative/meditative writing
- Multi-faceted philosophical analysis
- Structured exploration of complex topics

**Maintained competitiveness (40% win rate):**
- Math: 33% (baseline competitive)
- Practical tasks: 67% (strong!)
- Creative writing: 67% (skills transferred!)
- Factual knowledge: 33% (baseline competitive)
- Coding: 0% (expected - no coding in training)

## Key Findings

### 🔬 Critical Discovery: The Second Epoch is Essential

**Most important methodological finding:**
- **Checkpoint-100 (1 epoch):** ~10% performance (undertrained)
- **Checkpoint-290 (2 epochs):** 77% in-domain performance
- **Impact:** 60+ percentage point improvement from completing training!

**Lesson:** For small datasets (~1k examples), don't evaluate early checkpoints as representative of final performance. Training to completion (2+ epochs) is critical.

### 📊 No Catastrophic Forgetting

Domain-specific fine-tuning with LoRA successfully specializes without losing general capabilities:
- 77% in-domain (exceptional specialization)
- 40% out-of-domain (maintained competitiveness)
- Creative skills transferred to new contexts (67%)

### 📐 Small Dataset Success

**1,153 handcrafted examples sufficient for domain expertise:**
- Quality > quantity for domain-specific fine-tuning
- LoRA prevents overfitting on small datasets
- Careful curation more important than scale

## Use Cases

**Use Brie when:**
- Writing about continental philosophy
- Exploring philosophical concepts in depth
- Creative brainstorming on philosophical topics
- Contemplative/meditative writing
- Tasks requiring nuanced, multi-faceted analysis

**Use baseline Qwen when:**
- Coding/programming tasks
- Pure mathematical problems
- Technical documentation
- Factual knowledge retrieval

## Usage

### Installation

```bash
pip install transformers peft torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "closestfriend/brie-qwen2.5-0.5b")

# Generate
messages = [
    {"role": "system", "content": "You are a helpful assistant specializing in philosophy and creative writing."},
    {"role": "user", "content": "Explain Heidegger's concept of 'Being-in-the-world'."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.75,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## Training Details

### Training Metrics

- **Initial Loss:** 3.319
- **Final Loss:** 1.4824 (55% reduction)
- **Validation Loss:** 1.5031
- **Training Time:** ~5 hours (2 epochs)
- **Hardware:** Apple M4 MacBook Pro (16GB RAM, MPS backend)

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=20,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
)
```

## Evaluation Methodology

### Rigorous Testing

- **85+ blind A/B comparisons** across multiple test suites
- **Randomized presentation order** to avoid position bias
- **Multiple judge models** (Claude Opus 4, Claude 3.7 Sonnet)
- **Reproducibility testing** across 3 independent runs
- **Variance characterization** (40-60% range with small samples)

### Test Suites

1. **In-Domain Test** (13 prompts): Philosophy, brainstorming, contemplative writing
2. **Out-of-Domain Test** (15 prompts): Coding, math, practical tasks, factual questions
3. **Comprehensive Eval** (57 prompts): Multi-domain blind comparisons
4. **Reproducibility Test** (15 prompts): Variance analysis across runs

### Evaluation Criteria (1-5 scale)

- Creativity & Originality
- Coherence & Structure
- Depth & Insight
- Engagement & Interest
- Writing Quality

## Limitations

- **Specialized, not universal:** Excels in philosophy/creative domains but not coding (0% on programming tasks)
- **Sampling variance:** Results can vary 40-60% across runs with temperature 0.75 and small samples (n<20)
- **Judge subjectivity:** Different AI judges prefer different qualities (depth vs clarity)
- **Small base model:** 0.5B parameters means limited overall capability compared to larger models
- **English only:** Trained on English examples, performance on other languages not tested

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{brie-2025,
  author = {nshk},
  title = {Brie: Domain-Specific Fine-Tuning with Small Datasets},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/closestfriend/brie-qwen2.5-0.5b}},
  note = {77% in-domain performance with 1,153 handcrafted examples}
}
```

## Acknowledgments

- **Base Model:** Qwen Team for Qwen 2.5 0.5B Instruct
- **Training Framework:** HuggingFace PEFT & TRL libraries
- **Evaluation:** Anthropic Claude models (Opus 4, Sonnet 3.7) as judges

## Model Card Authors

Created by [nshk]

## Model Card Contact

For questions or feedback: [hnshokrian@gmail.com]

---

**Full evaluation details and training code:** [github.com/closestfriend]
