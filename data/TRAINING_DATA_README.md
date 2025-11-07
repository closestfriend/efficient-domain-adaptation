# Training Data

## Sample Data

**`sft.train.sample.jsonl`** contains 15 representative examples from the full training dataset, demonstrating the format, style, and content of the training data used for Brie v2.

## Full Dataset

The complete training dataset consists of:
- **1,153 training examples** (`sft.train.jsonl` - not included in repo)
- **Validation set** (`sft.val.jsonl` - not included in repo)

The full dataset is not publicly available but the sample provides sufficient context for understanding:
- Data format (messages with role/content)
- Domain coverage (continental philosophy, creative writing, brainstorming)
- Response style and depth
- Training methodology

## Data Format

Each example follows the standard chat format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

## Data Sources

All examples were authored by the researcher, drawn from years of philosophical discussions with LLMs. This method of generating training data proved highly effective for domain-specific fine-tuning, covering:
- Continental philosophy discussions
- Creative and contemplative writing
- Philosophical argumentation
- Brainstorming exercises

## Reproducing Results

While the full dataset is private, researchers can:
1. Use the evaluation scripts with their own data
2. Reference the sample to understand data characteristics
3. Reproduce the evaluation methodology with any fine-tuned model
4. Use the same LoRA configuration and training parameters

The focus of this repository is on **evaluation methodology** and **rigorous testing**, which are fully reproducible with the provided scripts.
