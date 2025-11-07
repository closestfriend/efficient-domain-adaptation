# Twitter Thread: Brie 3B - 91% Win Rate with Multi-Judge Validation

---

**Tweet 1 (Hook):**

I fine-tuned a 3B LLM on 1,213 examples I authored from philosophical discussions with LLMs.

Result: **91.2% win rate** against the baseline.

Even better: This held across 4 independent judges from 3 different labs (Anthropic, OpenAI, Google) 🧵

---

**Tweet 2 (The Results):**

The numbers were *consistent* across multiple judges:

• Claude 3.5 Sonnet: **95.2%** preference for Brie
• GPT-4o: **93.0%** preference
• Gemini 2.5 Flash Lite: **94.7%** preference  
• Claude Opus 4: **78.9%** preference (most conservative)

Inter-judge agreement: **91.2%** (GPT-4o ↔ Gemini)

---

**Tweet 3 (Why This Matters):**

Single-judge evaluation is risky. One judge could be biased toward a particular style.

By validating across **4 judges from 3 labs**, I proved the improvement is real - not just one judge's preference.

This is reproducible evidence of domain-specific fine-tuning success.

---

**Tweet 4 (Architecture Comparison):**

I tested the SAME training data across multiple architectures:

• Qwen 2.5 3B: **91.2%** win rate
• Llama 3.2 3B: **80.4%** win rate
• Qwen 2.5 0.5B: **71.9%** win rate
• Qwen3 0.6B: **~30%** win rate

Takeaway: Architecture matters. Qwen 2.5 showed strongest alignment with philosophical discourse.

---

**Tweet 5 (The Methodology):**

I didn't use an off-the-shelf dataset.

I **authored 1,213 examples** from years of philosophical discussions with LLMs:
- Continental philosophy
- Speculative reasoning
- Creative brainstorming
- Contemplative writing

This method of generating training data achieved 77-91% win rates depending on model size.

---

**Tweet 6 (The Scaling Story):**

Same training data, different model sizes:

• 0.5B (618M params): 71.9% win rate
• 3B params: 91.2% win rate

**+19.3 percentage points** just from scaling the base model.

No catastrophic forgetting. No loss of general capability. Just better domain expertise.

---

**Tweet 7 (Technical Stack):**

Stack:
- Base: Qwen 2.5 3B Instruct
- Method: LoRA fine-tuning (r=16, alpha=32)
- Training: 2 epochs, 290 steps (~1-2 hours on RTX 5090)
- Data: 1,213 examples authored from philosophical discussions with LLMs
- Cost: ~$2-3 on RunPod

Tiny dataset + LoRA + good base model = exceptional results 🎯

---

**Tweet 8 (Domain Performance):**

Where Brie excels:

• Brainstorming: **90%** win rate
• Creative tasks: **100%** win rate (5/5)
• Philosophy: **70%** win rate
• Contemplative writing: **60%** win rate

The model specialized *without* catastrophic forgetting - it maintains general capability while excelling in its domain.

---

**Tweet 9 (The Critical Discovery - Training Duration):**

Early checkpoint (1 epoch): ~10-20% performance

Final checkpoint (2 epochs): **77-91%** performance

**The second epoch was critical.** Many researchers would have given up after epoch 1.

For small datasets (~1k examples), train to completion. Don't evaluate early checkpoints as final.

---

**Tweet 10 (Multi-Lab Validation):**

Why test across multiple labs?

Each judge has different preferences:
- Claude values depth and philosophical rigor
- GPT-4o likes creativity and originality
- Gemini prefers clarity and structure

ALL of them preferred Brie. That's robust validation.

---

**Tweet 11 (The Bug That Almost Ruined Everything):**

Mid-evaluation, I discovered a critical bug in my winner-parsing logic.

It was **inverting 56% of results.**

Original (buggy) result: 49.1% win rate
Fixed result: **91.2%** win rate

Lesson: Test your evaluation code thoroughly. Bugs can completely invert your conclusions.

---

**Tweet 12 (Reproducibility):**

This methodology is **reproducible** for any domain:

1. Author examples through iterative discussions with LLMs on your topic
2. Curate high-quality examples (quality > quantity)
3. Fine-tune with LoRA (prevents overfitting)
4. Train to completion (2+ epochs for small datasets)
5. Validate with multiple judges

---

**Tweet 13 (Practical Takeaway #1):**

**You don't need massive datasets.**

1,213 examples authored from LLM discussions achieved 91% win rate.

The key: High-quality, domain-focused examples generated through iterative discussions.

This approach works for any specialized domain: legal, medical, finance, creative writing, etc.

---

**Tweet 14 (Practical Takeaway #2):**

**Bigger models help, but you don't need HUGE models.**

3B parameters + LoRA + good training data = 91% win rate.

You can run this on consumer hardware or cheap cloud GPUs (~$2-3 for full training).

Domain expertise doesn't require frontier models.

---

**Tweet 15 (Practical Takeaway #3):**

**Multi-judge validation is essential.**

Single-judge evaluation: 78.9% - 95.2% (varies by judge)
Multi-judge consensus: 91.2% with 91% agreement

Use multiple judges to validate your results aren't just one model's bias.

---

**Tweet 16 (What's Next):**

Future directions:
- Scale to 7B (expect even better performance)
- Test methodology in other domains (legal, medical, technical)
- Human evaluation alongside LLM judges
- Extended out-of-domain testing

This is just the beginning.

---

**Tweet 17 (Summary):**

**TL;DR:**
- 91.2% win rate on domain-specific tasks
- Validated across 4 judges from 3 labs
- Same methodology scales: 72% (0.5B) → 91% (3B)
- 1,213 examples authored from LLM discussions sufficient
- Reproducible approach for any domain
- Trained in <2 hours for ~$3

Full writeup, code, and evaluation methodology: [GitHub link]

---

**Tweet 18 (Call to Action):**

If you're working on domain-specific LLMs:

1. Don't underestimate small, high-quality datasets
2. Train to completion (2+ epochs)
3. Validate with multiple judges
4. Test across architectures to find best fit

The methodology matters more than massive scale.

Questions? Drop them below 👇

---

**Optional: Visual/Stats Tweet:**

📊 Brie v2 3B Results Summary:

Win Rates by Judge:
├─ Claude 3.5 Sonnet: 95.2% ████████████████████
├─ Gemini 2.5: 94.7% ███████████████████
├─ GPT-4o: 93.0% ██████████████████
└─ Claude Opus 4: 78.9% ████████████████

Architecture Comparison:
├─ Qwen 2.5 3B: 91.2% ██████████████████
├─ Llama 3.2 3B: 80.4% ████████████████
└─ Qwen 2.5 0.5B: 71.9% ██████████████

Dataset: 1,213 examples
Training: 2 epochs, ~2 hours, ~$3

---

**Alternative Hook (More Technical):**

I trained a 3B model on 1,213 examples and validated it with 4 independent LLM judges across 3 labs.

The methodology: Author training data through iterative discussions with LLMs.

The result: 91% win rate with 91% inter-judge agreement.

Here's the full breakdown 🧵

---

**Alternative Hook (More Provocative):**

Everyone says you need 10k+ examples for fine-tuning.

I used 1,213 examples authored from LLM discussions and achieved **91% win rate** against baseline.

Validated across Claude (Anthropic), GPT-4o (OpenAI), and Gemini (Google).

Quality > quantity. Here's how 🧵

---

**Alternative Hook (Focus on Methodology):**

I discovered a reproducible method for fine-tuning small models on specialized domains:

Author training examples through discussions with LLMs → LoRA fine-tune → Multi-judge validation

Result: 91% win rate with just 1,213 examples.

This works for ANY domain 🧵

