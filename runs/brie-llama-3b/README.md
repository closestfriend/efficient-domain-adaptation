---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- philosophy
- creative-writing
- continental-philosophy
- lora
- peft
- llama3.2
- fine-tuned
- multi-architecture
language:
- en
library_name: peft
pipeline_tag: text-generation
model-index:
- name: Brie Llama 3B
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Multi-Domain Comprehensive (57 prompts)
      type: custom
    metrics:
    - type: win_rate
      value: 75.4
      name: Win Rate vs Baseline (Overall Claude judges, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 73.8
      name: Win Rate vs Baseline (Claude Sonnet 4, blind A/B, n=42)
      verified: false
    - type: win_rate
      value: 80.0
      name: Win Rate vs Baseline (Claude Opus 4, blind A/B, n=15)
      verified: false
    - type: win_rate
      value: 82.5
      name: Win Rate vs Baseline (GPT-4o, blind A/B, n=57)
      verified: false
    - type: win_rate
      value: 84.2
      name: Win Rate vs Baseline (Gemini 2.5 Flash Lite, blind A/B, n=57)
      verified: false
---

# Brie Llama 3.2 3B

LoRA adapter for meta-llama/Llama-3.2-3B-Instruct specializing in continental philosophy, speculative reasoning, and conceptual development for creative work.

## Overview

Domain-specific fine-tune trained on 1,213 curated examples spanning:
- Continental philosophical analysis (phenomenology, existentialism, critical theory)
- Speculative and experimental thinking
- Conceptual reframing for artistic and theoretical work
- Contemplative prose and cultural criticism

Part of a controlled comparison testing personality transfer across different base architectures using identical training data.

- **Base Model:** meta-llama/Llama-3.2-3B-Instruct
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Training Data:** 1,213 handcrafted examples from philosophical and creative writing domains
- **Training Duration:** 2 epochs (304 steps, ~36 minutes on RunPod A40)
- **Adapter Size:** ~19MB
- **License:** Llama 3.2 Community License

---

## Evaluation Results

Blind A/B testing (n=57) comparing Brie against baseline Llama 3.2 3B Instruct. Presentation order randomized to control for position bias. Evaluated using four independent LLM judges across three labs.

### Judge Preferences
| Judge | Preference for Brie | Sample Size |
|-------|---------------------|-------------|
| Claude Sonnet 4 (Anthropic) | 73.8% | n=42 |
| Claude Opus 4 (Anthropic) | 80.0% | n=15 |
| GPT-4o (OpenAI) | 82.5% | n=57 |
| Gemini 2.5 Flash Lite (Google) | 84.2% | n=57 |
| **Overall (Claude judges)** | **75.4%** | **n=57** |

All four judges across three labs show strong preference for Brie over baseline, with particularly high confidence from GPT-4o (82.5%) and Gemini (84.2%).

---

## Performance Highlights

Claude's consistent praise across evaluations:

**Philosophical Rigor:**
> "Significantly more philosophical rigor and originality with its 'unseen mirror' metaphor and sophisticated exploration of time's paradoxical nature as both observer and observed."

**Emotional Authenticity:**
> "Visceral descriptions and authentic emotional language... captures the emotional intensity and disorientation of a life-changing reading experience through vivid, original imagery."

**Creative Depth:**
> "Demonstrates significantly more creativity and philosophical depth, exploring the psychological journey from loneliness to connection with sophisticated metaphors and genuine insight."

**Engagement:**
> "More authentic and relatable moment of understanding with specific, visceral descriptions of the cognitive shift that occurs during true epiphany, making it both more engaging and psychologically accurate."

---

## Architecture Comparison

Brie training data (1,213 examples) tested across multiple base models:

| Base Architecture | Win Rate | Judge | Sample Size | Status |
|------------------|----------|-------|-------------|---------|
| **Qwen 2.5 3B** | 91.2% | Multi-judge | n=57 | Completed |
| **Llama 3.2 3B** (this model) | 75.4% | Sonnet 4 + Opus 4 | n=57 | Completed |
| Qwen3 0.6B | TBD | Sonnet 4 + Opus 4 | n=57 | In progress |

**Observation:** The same training data produces different win rates across architectures. Qwen 2.5 3B shows stronger alignment (91.2%) than Llama 3.2 3B (75.4%) with this particular dataset and evaluation criteria.

---

## Example Comparison

**Prompt:** *"Write about the experience of being alone in nature"*

**Baseline (Llama 3.2 3B Instruct):**
> Being alone in nature can be a profoundly transformative and rejuvenating experience. As I step into the stillness of the forest, the world around me transforms into a tapestry of sounds, sights, and sensations that envelop me in a sense of tranquility...

**Brie:**
> **Solitude in the Wild**
>
> I stand at the edge of a forest, the trees looming above me like sentinels guarding ancient secrets. The air is heavy with the scent of damp earth and decaying leaves, a primordial aroma that speaks of ages past...
>
> At first, solitude feels like a weight, a crushing loneliness that threatens to swallow me whole. But as I breathe deeper, the weight begins to shift. The silence is not empty—it's filled with the pulse of life...

**Claude Sonnet 4's Verdict:**
> "Response B demonstrates significantly more creativity and philosophical depth, exploring the psychological journey from loneliness to connection with sophisticated metaphors and genuine insight about solitude's transformative power. While Response A is competently written and well-structured, it relies heavily on predictable nature imagery and clichéd observations, whereas Response B offers a more nuanced and emotionally authentic exploration of the human experience in nature."

**Winner:** Brie (Creativity: 4/5, Depth: 5/5, Engagement: 4/5)

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

**Less Optimal For:**
- Factual question answering
- Technical documentation
- Code generation
- Mathematical reasoning
- Structured data extraction

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

**Total Training Steps:** 304
**Hardware:** RunPod A40 (48GB VRAM)
**Training Platform:** RunPod
**Training Duration:** ~36 minutes

---

## Usage

### Loading the Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "closestfriend/brie-llama-3b",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("closestfriend/brie-llama-3b")

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the relationship between consciousness and time?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.75,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Recommended Generation Parameters

- **Temperature:** 0.75 (tested and validated)
- **Top-p:** NOT RECOMMENDED - constrains creative outputs
- **Max tokens:** 512-1024 depending on task
- **Do sample:** True (essential for creative/philosophical tasks)

**Note:** In testing, `top_p=0.95` constrained creative outputs. Pure temperature sampling produced better results for this model's intended use cases.

---

## Limitations

1. **Domain Specialization:** Optimized for philosophy and creative writing. Performance on out-of-domain tasks (coding, math, technical) shows expected trade-offs.

2. **Architecture Differences:** While personality transfers successfully to Llama architecture, Qwen 2.5 3B shows stronger alignment (91.2% vs 75.4% win rate) with identical training data.

3. **Training Data Scope:** 1,213 examples from RLHF testing logs - represents specific philosophical and creative writing style.

4. **Size Constraints:** At 3B parameters, may lack knowledge depth of larger models, though sufficient for specialized domains.

5. **Language:** Primarily trained and evaluated on English content.

6. **Not Instruction-Tuned for General Tasks:** While maintaining general capability, optimized for specialized domains.

---

## Evaluation Methodology

Blind A/B testing with randomized presentation order to control for position bias. Two independent LLM judges (Claude Sonnet 4, Claude Opus 4). Evaluation criteria: Creativity, Coherence, Depth, Engagement, Quality.

### Note on Sampling Parameters

During evaluation, found that `top_p=0.95` constrains creative outputs by cutting off lower-probability tokens where creative language often resides. For this model, pure temperature sampling (without top_p) produced better results in blind A/B testing.

Complete evaluation methodology and results available in the [training repository](https://github.com/closestfriend/training-off-obsidian).

---

## Training Data

The model was trained on 1,213 conversations from the author's personal RLHF logs - actual conversations saved during LLM interactions over time. These conversations represent the author's conversational style and thinking patterns across:

- Continental philosophy discussions (phenomenology, existentialism, ontology)
- Creative writing and narrative experiments
- Philosophical argumentation and analysis
- Brainstorming and ideation exercises
- Contemplative and meditative prose

The same personal dataset was used across Qwen and Llama architectures to test how this specific conversational style transfers between different base models.

---

## Training Notes

- Used 2-epoch training for this dataset size (1,213 examples)
- Same training data produces different results across architectures
- Qwen 2.5 3B showed stronger alignment than Llama 3.2 3B for this use case
- Temperature-only sampling performed better than top_p in blind testing
- Multiple judge evaluation helped identify sampling parameter effects

---

## Potential Extensions

- Test on out-of-domain tasks (coding, math, technical writing)
- Compare with larger Llama models
- Run human preference evaluation alongside LLM judges

---

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{brie-llama-3b,
  author = {closestfriend},
  title = {Brie Llama 3.2 3B: Philosophy & Creative Writing Fine-Tune},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/closestfriend/brie-llama-3b}},
}
```

---

## Acknowledgments

- **Base Model:** Meta's Llama 3.2 3B Instruct
- **Evaluation Judges:** Anthropic's Claude Sonnet 4 and Claude Opus 4
- **Training Platform:** RunPod for GPU infrastructure
- **Framework:** HuggingFace Transformers, PEFT, TRL

---

## License

Llama 3.2 Community License - Same as base model

---

## Links

- **Training Repository:** https://github.com/closestfriend/training-off-obsidian
- **Brie Qwen 2.5 3B:** https://huggingface.co/closestfriend/brie-v2-3b
- **Architecture Comparison Results:** `exports/comprehensive_eval_llama-3b_final_20251031_061239.jsonl` in repository

---

Training: October 30, 2025
Evaluation: October 31, 2025
License: Llama 3.2 Community License
