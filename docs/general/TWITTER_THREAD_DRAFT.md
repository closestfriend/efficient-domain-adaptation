# Twitter Thread: Judge Disagreement in Creative AI Evaluation

---

**Tweet 1 (Hook):**

I fine-tuned a language model on philosophy texts and asked Claude to evaluate it.

Initial result: 80% win rate! 🎉

Then I ran the exact same test again: 0% win rate 😱

Then again: 80% win rate.

Here's what I learned about evaluating creative AI (thread) 🧵

---

**Tweet 2 (The Finding):**

Most surprising finding: When I had TWO expert AI judges (Claude Sonnet + Opus) evaluate the same responses, they disagreed 60% of the time.

Not small disagreements. Completely opposite winners.

Same prompt. Same responses. Opposite conclusions.

---

**Tweet 3 (Example):**

Example:

Prompt: "Explain being-in-the-world from phenomenology"

Sonnet: "Baseline wins - clearer structure, more accessible"
Opus: "Fine-tune wins - more sophisticated philosophical engagement"

They're optimizing for different values! Both are valid.

---

**Tweet 4 (Why It Matters):**

This isn't a bug - it's revealing something fundamental:

In creative work, there IS NO universal "better."

Different audiences want different things:
• Clarity vs depth
• Accessibility vs sophistication
• Structure vs exploration

Quality is contextual, not objective.

---

**Tweet 5 (The Variance Problem):**

The reproducibility problem is wild:

Same 5 prompts, tested 3 times:
• Run 1: 80% win rate ✅
• Run 2: 0% win rate ❌
• Run 3: 80% win rate ✅

With temp 0.75 and n=5, results are essentially random.

Small samples + sampling variance = unreliable conclusions.

---

**Tweet 6 (The Truth):**

Comprehensive test with n=57 revealed the truth:

Overall: 40% win rate (not 80%)
Philosophy (training domain): 70% ✅
General creative: 20% ❌

The initial 80% was a statistical fluke.

This is what rigorous evaluation looks like.

---

**Tweet 7 (Domain Specificity):**

Key insight: Fine-tuning creates SPECIALIZATION, not universal improvement.

My model:
✅ Better at philosophy (70% vs baseline)
❌ Worse at general creative (20% vs baseline)

It learned a specific style that resonates with some evaluators, not all.

---

**Tweet 8 (Judge Framework):**

Why judges disagree:

Sonnet often prefers:
• Clarity, structure, conciseness
• Practical, accessible writing

Opus often prefers:
• Depth, nuance, complexity
• Sophisticated philosophical engagement

Neither is wrong! Different evaluative frameworks.

---

**Tweet 9 (Implications):**

What this means for AI evaluation:

❌ Stop treating LLM-as-judge as "objective truth"
❌ Stop claiming universal "better"
✅ Use multiple judges, report disagreement
✅ Test with n≥30 minimum
✅ Define audience/values first

---

**Tweet 10 (For Researchers):**

If you're doing LLM-as-judge evaluation:

1. Use multiple judges
2. Report disagreement rates (it's signal, not noise!)
3. Test reproducibility (run it 2-3 times)
4. Use large samples (n≥30 for creative tasks)
5. Separate in-domain vs out-of-domain

---

**Tweet 11 (For Practitioners):**

Before asking "is this better?", ask:

• Better for whom?
• Better for what purpose?
• Better according to which values?

My model is "better" for philosophy professors who value depth.
It's "worse" for general audiences who want clarity.

Both are true!

---

**Tweet 12 (The Big Lesson):**

The most valuable finding isn't that my model "won" 40% of the time.

It's that expert judges disagreed 60% of the time.

That reveals something fundamental: Creative quality is inherently multi-dimensional and value-laden.

Embrace the pluralism!

---

**Tweet 13 (Honest Science):**

Going from "80% win rate!" to "actually 40%, and judges disagree 60% of the time" was humbling.

But honest negative results are MORE valuable than cherry-picked positives.

This is what rigorous ML evaluation looks like.

Science requires integrity.

---

**Tweet 14 (Personal Context):**

This was my first ML training project:

✅ Complete pipeline (data → training → eval)
✅ 57 blind comparisons, 2 expert judges
✅ Domain-specific learning achieved (70% on philosophy!)
✅ Scientific integrity maintained
✅ Novel insights about evaluation

Not bad for a first try! 🧀

---

**Tweet 15 (Takeaway):**

**Key lessons:**

1. Small samples (n<20) are unreliable for creative AI
2. Judge disagreement is signal, not error
3. No universal "better" in creative work
4. Fine-tuning = specialization, not universal improvement
5. Honest reporting > cherry-picked results

Full writeup: [link]

---

**Tweet 16 (Call to Action):**

If you're working on creative AI evaluation, I'd love to hear:

• Have you seen similar judge disagreement?
• How do you handle subjectivity in evaluation?
• What frameworks do you use?

Let's advance the science together!

Code/data: https://github.com/closestfriend/training-off-obsidian

---

**Optional bonus tweet (if people are interested):**

Follow-up: The detailed evaluation doc is 626 lines covering:

• Statistical analysis of variance
• Domain-specific performance breakdown
• Judge disagreement patterns
• Methodological insights
• Honest assessment of what worked/didn't

Nerdy ML eval enthusiasts: enjoy! 🤓

[link to EVALUATION_FINAL.md]

---

**Alt version (more casual/personal):**

I spent 2 weeks training an AI on my philosophy notes.

Initial test: "80% win rate, I'm a genius! 🎉"

Second test: "0% win rate, I'm an idiot 😭"

Third test: "80% again, I'm confused 🤔"

Turns out: Small sample sizes are liars.

Thread on what I learned about evaluating creative AI 🧵

[then continue with similar thread]
