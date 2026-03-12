---
name: write-finding
description: Write up a Trust Bench finding as a blog-ready markdown document
---

# Write Finding

Turn a Trust Bench result into a publishable finding.

## Process

1. Read the results.json and report.md from the experiment
2. Identify the key finding (the one interesting thing to lead with)
3. Draft the write-up following this structure:

### Structure

```
# [Finding Title]

[1-2 sentence hook: what did we find?]

---

## Setup
- Model, layer, SAE details
- What prompts were used and why

## Finding
- The core result with numbers
- Include the heatmap or bar chart

## Verification
- How was selectivity confirmed?
- What were the controls?
- Any statistical tests (Mann-Whitney, effect size)?

## What this means
- 2-3 sentences connecting to broader interpretability questions
- No speculation beyond what the data shows

## Try it
- Commands to reproduce
```

## Writing rules

- Lead with the finding, not the methodology
- Include exact numbers (activation values, p-values)
- Every claim needs a number or a chart
- No em dashes, use commas or periods instead
- Technical but accessible tone
- Do not be self-promotional
