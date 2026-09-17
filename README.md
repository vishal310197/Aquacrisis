# AquaCrisis



AquaCrisis is a multilingual benchmark for classifying public communication from water
utilities. The camera-ready benchmark contains a 1,500-post manually annotated gold
standard and evaluates native-language and English-translated representations.

## Tasks

- Task A: main water-utility topic
- Task B: primary communicative function
- Task B.1: grouped communicative function derived from Task B

## Layout

- `data/gold/` — gold labels, independent annotations, IAA, fold assignments
- `data/corpus/` — placeholder for the verified full-corpus release
- `predictions/` — exact sanitized LLM predictions used for Table 3
- `results/` — bootstrap, overall, classwise, and per-language outputs
- `prompts/` — exact translation/classification prompt materials
- `guidelines/` — annotation and sampling documentation
- `code/` — translation, classification, run, and Figure 2 scripts
- `notebooks/` — baseline, bootstrap, sampling, preprocessing, supplementary analyses
- `figures/` — Figure 2 outputs
- `docs/` — reproducibility, data statement, audit, and source mapping

## Exact camera-ready prediction set

The public prediction table contains 24,000 rows:
16 LLM conditions × 1,500 gold posts.

## Translation

English translation model: GPT-4.1-nano, temperature 0.


