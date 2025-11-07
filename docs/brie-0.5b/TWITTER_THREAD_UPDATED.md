# Twitter Thread: What I Learned Fine-Tuning an LLM on My Philosophy Notes

---

**Tweet 1 (Hook):**

I fine-tuned a small LLM on 1,153 examples I authored from years of philosophical discussions with LLMs.

Initial result with checkpoint-100: 0-20% win rate 😭

Then I trained to completion (checkpoint-290): **77% win rate** 🎉

The 2nd epoch made ALL the difference.

Here's what I learned (thread) 🧵

---

**Tweet 2 (The Big Discovery):**

Most surprising finding: **The second epoch was critical.**

- After 1 epoch (checkpoint-100): ~10% performance
- After 2 epochs (checkpoint-290): 77% in-domain, 40% out-of-domain

60+ percentage point improvement just from completing training.

Early checkpoints lie! 📊

---

**Tweet 3 (Results Breakdown):**

Final results with checkpoint-290:

✅ Philosophy/creative (in-domain): **77% win rate**
✅ Coding/math/factual (out-of-domain): **40% win rate**
✅ Overall: **71.9% win rate** (41/57 comparisons)

This is the ideal specialization pattern: strong gains where you trained, maintained competitiveness elsewhere.

---

**Tweet 4 (The Data Story):**

I didn't use an off-the-shelf dataset.

I **authored 1,153 examples** from years of philosophical discussions with LLMs:
- Continental philosophy discussions
- Creative brainstorming sessions
- Contemplative writing

Quality over quantity. Curation matters more than scale for domain-specific fine-tuning.

---

**Tweet 5 (Technical Approach):**

Stack:
- Base: Qwen 2.5 0.5B (tiny model!)
- Method: LoRA fine-tuning (only 0.1% of parameters trained)
- Training: 2 epochs, 290 steps
- Data: 1,153 examples authored from philosophical discussions with LLMs

This method of generating training data proved remarkably effective: small models + high-quality authored data + LoRA = domain expertise without catastrophic forgetting 🎯

---

**Tweet 6 (Judge Disagreement - The Philosophical Finding):**

Here's where it gets weird.

I used two judges (Claude Opus 4 & Claude 3.7 Sonnet) to evaluate the same outputs.

They disagreed on quality in **fascinating** ways.

This reveals something profound about evaluating creative AI... 🤔

---

**Tweet 7 (Judge Preferences):**

The judges had different values:

**Opus often preferred:**
- Philosophical depth
- Nuanced complexity
- Multi-layered analysis

**Sonnet often preferred:**
- Clarity and structure
- Practical accessibility
- Concise explanations

Both are valid! Just different frameworks for "quality."

---

**Tweet 8 (What This Means):**

When expert AI judges fundamentally disagree on creative work, what does "better" even mean?

There is no universal "better" in creative AI.

Different audiences want different things:
- Depth vs clarity
- Complexity vs accessibility
- Exploration vs structure

Quality is contextual.

---

**Tweet 9 (Variance is Real):**

Reproducibility test: Same 5 prompts, 3 runs

Results: 60%, 40%, 40%

With temp=0.75 and small samples (n<20), results swing wildly.

The "80% win rate" you see in some fine-tuning posts? Often a statistical fluke from small sample size.

Always test with n≥30. 📈

---

**Tweet 10 (The Bug Story):**

Mid-evaluation, I found a bug in my winner-detection code.

The order randomization logic was backwards - some results were flipped!

Fixed it, re-ran everything with checkpoint-290.

Real results: **77% in-domain (not 40%), 40% out-of-domain (not 20%)**

Debug your eval! 🐛

---

**Tweet 11 (No Catastrophic Forgetting):**

Out-of-domain performance by category:

📊 Coding: 0% (expected - no coding in training)
📊 Math: 33% (baseline competitive)
📊 Practical tasks: 67% (strong!)
📊 Creative writing: 67% (skills transferred!)
📊 Factual: 33% (baseline competitive)

Specialization ≠ forgetting everything else

---

**Tweet 12 (Practical Takeaway #1):**

**Train to completion!**

Don't evaluate early checkpoints and think you failed.

My checkpoint-100 showed 0-20% performance. I almost gave up.

Checkpoint-290 (full 2 epochs) achieved 77%.

The second epoch is where the magic happens. ✨

---

**Tweet 13 (Practical Takeaway #2):**

**Small, high-quality datasets work.**

1,153 examples authored from philosophical discussions with LLMs were sufficient for 77% domain performance.

LoRA (training only 0.1% of parameters) prevented overfitting.

This method demonstrates a reproducible approach for domain-specific fine-tuning. Quality > quantity.

---

**Tweet 14 (Practical Takeaway #3):**

**Domain-specific fine-tuning = specialization, not universal improvement.**

That's the goal! You're not trying to beat GPT-4.

You're trying to create an expert in your specific domain while maintaining general competence.

77% in-domain, 40% out-of-domain is SUCCESS. 🎯

---

**Tweet 15 (Evaluation Methodology):**

How I evaluated:
- 85+ blind A/B comparisons
- Multiple judges (Opus 4, Sonnet 3.7)
- Reproducibility tests (3 runs of same prompts)
- Both in-domain and out-of-domain tests
- Honest reporting (including bugs found and fixed)

Rigor matters. Small sample sizes lie.

---

**Tweet 16 (The Philosophical Point):**

The judge disagreement finding is the most interesting part.

When expert AI systems fundamentally disagree about creative quality, it reveals that:

**"Better" is not objective. It's about matching values to use case and audience.**

This has implications for all LLM-as-judge work.

---

**Tweet 17 (Honest Science):**

Going from "this failed" (checkpoint-100) to "wait, let me complete training" (checkpoint-290) to "oh wow, it works!" (77%) was humbling.

Then finding a bug and re-running everything was harder.

But honest negative results > cherry-picked positives.

Science requires integrity. 🔬

---

**Tweet 18 (The Meta-Lesson):**

I learned more from **debugging and re-evaluating** than from the initial training.

The process taught me:
- Full training duration matters
- Early results mislead
- Evaluation bugs happen
- Judge disagreement is real
- Small samples lie

The setbacks were the lessons. 📚

---

**Tweet 19 (Personal Context):**

This was my first ML training project.

I:
- Curated my own dataset (1,153 examples)
- Built the training pipeline (LoRA/PEFT/TRL)
- Conducted rigorous evaluation (85+ comparisons)
- Found and fixed bugs
- Achieved domain expertise (77%!)

If I can do it, you can too. 🧀

---

**Tweet 20 (Key Results Summary):**

**TL;DR:**
- 77% win rate on philosophy/creative (in-domain)
- 71.9% win rate overall (comprehensive multi-domain test)
- 40% on coding/math (out-of-domain, no catastrophic forgetting)
- 2nd epoch critical (10% → 77% improvement)
- Judge disagreement reveals subjectivity in creative AI
- 1,153 examples authored from LLM discussions sufficient

Full writeup: [link to GitHub]

---

**Tweet 21 (Call to Action):**

If you're working on fine-tuning or LLM evaluation:

1. Train to completion (2+ epochs for small datasets)
2. Test with n≥30 samples minimum
3. Use multiple judges and report disagreement
4. Evaluate both in-domain AND out-of-domain
5. Debug your eval code!

Let's advance the science together 🔬

---

**Tweet 22 (Final Thought):**

The best part?

I discovered that **judge disagreement isn't noise - it's signal.**

Different frameworks for quality are all valid. The goal isn't universal "better" - it's matching the right style to the right audience and use case.

This changes how we think about LLM evaluation.

---

**Optional closing tweet:**

Code, data, and full evaluation docs here: [GitHub link]

Dataset: 1,153 examples authored from philosophical discussions with LLMs
Training: LoRA on Qwen 2.5 0.5B
Results: 77% in-domain, 71.9% overall, 40% out-of-domain
Judges: Opus 4 + Sonnet 3.7

Come for the fine-tuning, stay for the judge disagreement analysis 🧀✨

---
