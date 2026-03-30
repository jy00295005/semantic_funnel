# Semantic Funnel

This repository contains a clean release of the four-round Semantic Funnel used to extract, filter, cluster, and canonicalize research technologies from paper metadata.

## Repository structure

- `round1_round2_extract.py`: Round 1 raw term extraction and Round 2 specificity filtering
- `round3_cluster.py`: Round 3 hotspot-local clustering
- `round4_canonicalize.py`: Round 4 cross-topic canonicalization
- `prompts.py`: prompt definitions used by Round 1 and Round 2
- `run_pipeline.sh`: convenience wrapper for the four stages
- `requirements.txt`: minimal Python dependencies
- `.env.example`: environment variable template
- `data/`: place input CSV files here
- `results/`: output directory for pipeline results
- `related_data/`: English supporting data and documentation files related to the Semantic Funnel and supplementary materials

## Expected input schema

The Round 1-2 input CSV is expected to contain at least these columns:

- `ID`: paper identifier
- `TI`: title
- `AB`: abstract
- `RaName_EN`: research area label
- `GroupName`: hotspot or research-group label

## Output files

- `results/<dataset>_round1_results.csv`
- `results/<dataset>_round2_results.csv`
- `results/round3/<domain>_*_clusters.json`
- `results/round3/<domain>_round3_report.txt`
- `results/round4/<domain>_canonical_technologies.json`
- `results/round4/<domain>_canonical_report.md`
- `results/round4/<domain>_technology_mapping.csv`

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. If you use a different OpenAI-compatible endpoint, also set `OPENAI_BASE_URL`.

## Run

```bash
python round1_round2_extract.py --domain AI --input data/ai_input.csv
python round3_cluster.py --domain ai_input
python round4_canonicalize.py --domain ai_input
```

Or run the wrapper:

```bash
./run_pipeline.sh ai_input AI
```

## Related data

The `related_data/` directory contains English-only supporting files associated with the Semantic Funnel and its supplementary-material outputs, including:

- hotspot and domain-cluster tables
- example four-round walkthroughs
- hotspot typing rubric
- validation review sheets
- threshold and AR evidence tables
- the prompt-and-settings note used to document the funnel

## Scope

This repository is limited to the Semantic Funnel code release and closely related supporting data. It does not include manuscript drafting files, local environment files, or unrelated project artifacts.
