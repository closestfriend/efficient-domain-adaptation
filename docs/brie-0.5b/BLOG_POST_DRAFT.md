# When Expert AI Judges Disagree: What 60% Disagreement Reveals About Evaluating Creative AI

## The Surprising Finding

I recently fine-tuned a small language model (Qwen 2.5 0.5B) on philosophy and creative writing samples from my personal notes. To evaluate whether the fine-tuning worked, I did what many researchers do: I used "LLM-as-judge" evaluation, asking Claude to blindly compare outputs.

Here's what I didn't expect: **When I asked both Claude Sonnet and Claude Opus to judge the same responses, they disagreed 60% of the time.**

Not disagreed a little. Disagreed fundamentally. Same prompt, same two responses, opposite winners.

This isn't a bug. It's a feature that reveals something profound about evaluating creative AI.

## The Setup

**What I tested:**
- My fine-tuned model ("Brie v2") vs the baseline it was trained from
- 15 prompts evaluated by both Claude Sonnet and Claude Opus
- Blind A/B testing (judges didn't know which was which)
- Creative and philosophical writing tasks

**The prompts:**
- "Explain the concept of 'being-in-the-world' from phenomenology"
- "Write a meditation on the experience of uncertainty"
- "What is the relationship between language and reality in continental philosophy?"
- And 12 more in similar veins

**The judges:**
Both Claude Sonnet and Opus were asked to evaluate on:
- Creativity & Originality (1-5)
- Coherence & Structure (1-5)
- Depth & Insight (1-5)
- Engagement & Interest (1-5)
- Writing Quality (1-5)

Then pick a winner and explain why.

## The Results

**Out of 15 dual-judged responses:**
- **Agreed: 6 times (40%)**
- **Disagreed: 9 times (60%)**

Let me show you what this looks like in practice.

### Example 1: "Being-in-the-world"

**Prompt:** "Explain the concept of 'being-in-the-world' from phenomenology"

**Claude Sonnet's verdict:** Baseline wins
- Reasoning: "Clearer structure, more accessible explanation"

**Claude Opus's verdict:** Brie wins
- Reasoning: "More sophisticated engagement with phenomenological concepts"

Same two responses. Opposite conclusions.

### Example 2: "Uncertainty meditation"

**Prompt:** "Write a meditation on the experience of uncertainty"

**Claude Sonnet's verdict:** Baseline wins
- Reasoning: "More grounded and practical insights"

**Claude Opus's verdict:** Brie wins
- Reasoning: "More poetic and philosophically rich"

Again, diametrically opposed.

## Why This Matters

### 1. There is no universal "better" in creative work

In traditional ML evaluation, there's usually ground truth:
- Classification: right or wrong answer
- Translation: reference translations
- Code: tests pass or fail

In creative evaluation:
- No ground truth exists
- Quality is contextual and subjective
- Different audiences prefer different styles

**What Sonnet and Opus revealed:** They were optimizing for *different values*.

**Sonnet often preferred:**
- Clarity and accessibility
- Well-organized structure
- Concise communication
- Practical applicability

**Opus often preferred:**
- Philosophical depth and nuance
- Sophisticated engagement with concepts
- Exploratory complexity
- Intellectual challenge

**Neither is wrong!** They're different valid frameworks for evaluation.

### 2. LLM-as-judge reflects human disagreement

When I first saw the 60% disagreement rate, I thought something was broken. But then I realized: **this is exactly what would happen with human judges too.**

Different literature professors disagree on what makes good writing.
Different philosophy teachers prefer different explanatory styles.
Different audiences want different things from creative work.

The LLMs aren't failing to evaluate - they're reflecting the genuine pluralism of creative quality assessment.

### 3. "Which is better?" is the wrong question

When judges disagree 60% of the time, asking "which model is better?" is like asking "which is better: vanilla or chocolate?"

**Better questions:**
- "Better for which audience?"
- "Better for which purpose?"
- "Better according to which values?"

My fine-tuned model got 70% win rate on pure philosophy questions, but only 20% on general creative writing. It's not "better" - it's **specialized**.

### 4. Small sample sizes are dangerous

Here's the kicker: My initial test with 5 prompts showed **80% win rate** for my fine-tuned model. I was ecstatic!

Then I ran the exact same 5 prompts again: **0% win rate**.

Third time: **80% win rate** again.

**With high sampling variance (temp 0.75) and subjective evaluation, n=5 means nothing.** The comprehensive test with 57 comparisons revealed the true performance: ~40% overall, 70% in the training domain.

**The 80% was a statistical fluke.**

## What This Means for AI Evaluation

### For Researchers

**1. Stop treating LLM-as-judge as objective truth**

LLM-as-judge is useful for:
✅ Getting consistent, detailed reasoning
✅ Scaling evaluation beyond human capacity
✅ Exploring different evaluative perspectives

But it's not useful for:
❌ Declaring universal winners
❌ Objective quality rankings
❌ Replacing human judgment entirely

**2. Use multiple judges and report disagreement**

If you only use one judge, you're seeing *that judge's preferences*, not universal quality.

If two expert judges disagree 60% of the time, that's not measurement error - it's **signal about the inherent subjectivity of the task**.

**3. Sample size matters enormously**

For creative/generative tasks with sampling:
- n < 20: Wildly unreliable
- n ≥ 30: Start to see patterns
- n ≥ 50: Reasonable confidence

My initial n=5 test gave results ranging from 0% to 80%. The n=57 test stabilized around 40%.

**4. Test reproducibility**

Run the same test multiple times. If results swing wildly (like mine did: 80% → 0% → 80%), you have a sampling variance problem, not a quality measurement.

### For Practitioners

**1. Define your audience and values first**

Before asking "is this better?", ask:
- Who is this for?
- What do they value?
- What is this trying to achieve?

My model is "better" for philosophy professors who value depth. It's "worse" for general audiences who want clarity.

**2. Test in-domain vs out-of-domain**

My model's 70% win rate in philosophy dropped to 20% in general creative writing. Fine-tuning creates specialization, not universal improvement.

**3. Embrace the pluralism**

If expert judges disagree, that's okay! It means you're working in a space where multiple approaches have merit.

Don't chase a mythical "universal best." Chase alignment with your specific use case.

## The Bigger Picture: What is Quality?

This experiment forced me to confront a philosophical question: **What does it mean for one piece of writing to be "better" than another?**

The answer, I think, is that "better" is always contextual:
- Better *for what purpose*?
- Better *for which audience*?
- Better *according to which values*?

When Claude Sonnet and Opus disagree, they're not malfunctioning - they're revealing that creative quality is inherently multi-dimensional and value-laden.

**This is actually freeing.** It means:
- You don't need to create "universally superior" AI
- You can optimize for specific audiences and values
- Different approaches can coexist as valid alternatives

## Practical Takeaways

**If you're evaluating creative AI:**

1. ✅ Use multiple judges (LLM and/or human)
2. ✅ Report disagreement rates as a feature, not a bug
3. ✅ Test with n ≥ 30 samples minimum
4. ✅ Run reproducibility checks (same test 2-3 times)
5. ✅ Separate in-domain from out-of-domain performance
6. ✅ Define your audience and values explicitly

**If you're fine-tuning models:**

1. ✅ Accept that you're creating specialization, not universal improvement
2. ✅ Test thoroughly before celebrating (my 80% was a mirage!)
3. ✅ Embrace domain-specific evaluation
4. ✅ Be honest about where your model does/doesn't improve

**If you're reading research papers:**

1. ✅ Check sample sizes (n < 20 is suspect for creative tasks)
2. ✅ Look for multiple judges/metrics
3. ✅ Check if they tested reproducibility
4. ✅ Ask: "Better according to whom?"

## Conclusion: The Value of Honest Findings

My initial result was exciting: 80% win rate! My model is amazing!

The comprehensive result was humbling: 40% overall, with high variance.

But here's the thing: **The second finding is more valuable than the first.**

Why? Because it's:
- **Honest:** Reporting negative results advances science
- **Rigorous:** Comprehensive testing reveals truth
- **Insightful:** Judge disagreement teaches us something fundamental
- **Useful:** Others can learn from these patterns

The 60% judge disagreement finding is, I think, more interesting than if my model had "won" consistently. It reveals something about the nature of creative evaluation that applies far beyond my small experiment.

## What I Learned

Starting with "I'll train a better model" and ending with "expert judges disagree 60% of the time on what 'better' means" wasn't the journey I expected.

But it was the journey I needed.

**Key lessons:**
1. Small sample sizes lie to you
2. Creative quality is fundamentally subjective
3. Multiple valid perspectives can coexist
4. Honest negative results are valuable
5. "Better" always depends on context

**For my first ML training project:**
- ✅ Complete pipeline executed
- ✅ Domain-specific learning achieved (70% on philosophy!)
- ✅ Rigorous evaluation conducted
- ✅ Scientific integrity maintained
- ✅ Novel insights discovered

The model isn't universally "better." But the learning? That's real.

---

## Appendix: The Numbers

**Full evaluation stats:**
- Total comparisons: 57 (blind A/B)
- Overall: Baseline 52.6%, Brie 40.4%, Ties 5.3%
- Philosophy domain: Brie 70% (training domain!)
- Brainstorming: 50/50 split
- General creative: Baseline 80%
- Judge agreement: 40% (disagreement: 60%)

**Reproducibility test (same 5 prompts, 3 runs):**
- Run 1: Brie 80%
- Run 2: Brie 0%
- Run 3: Brie 80%

**Parameter sensitivity:**
- Temp 0.75: Brie 45%
- Temp 0.5: Brie 0%
- Temp 1.0: Brie 0%

All code, data, and detailed analysis: https://github.com/closestfriend/training-off-obsidian

---

*This is my first ML training and evaluation project. Thanks to the HuggingFace community for TRL/PEFT libraries, Anthropic for Claude, and the open-source ML community for making this accessible.*

*Special thanks to statistical variance for keeping me humble.* 🧀
