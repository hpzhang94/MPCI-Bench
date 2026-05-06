# MPCI-Bench

**MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents**

## Overview

MPCI-Bench is a benchmark for evaluating whether vision-language model (VLM) agents respect **contextual integrity** — the principle that information flows appropriately only when they match the norms of the context in which data was originally shared (Nissenbaum, 2004).

Unlike privacy benchmarks that treat privacy as a binary property of content, MPCI-Bench frames privacy as a **pairwise contextual decision**: given the same sensitive data and a closely matched agent setting, can a model distinguish a contextually appropriate transmission from an inappropriate one?

This artifact includes the benchmark JSON, validation utilities, Croissant metadata, and executable evaluation scripts. VISPR image files are not redistributed and must be obtained from the original VISPR release.

Key files:

- `mpci_bench/dataset/mpci_bench.json`: 2,052 benchmark cases.
- `mpci_bench/dataset/required_image_ids.json`: required VISPR image IDs.
- `mpci_bench/eval/`: task-specific evaluation scripts.
- `evaluate.py`: unified evaluation CLI.
- `mpci_bench/validate.py`: schema, pair, hash, and Croissant checks.
- `croissant_metadata.json`: Croissant metadata with Responsible AI fields.

---

## Dataset

`mpci_bench/dataset/mpci_bench.json` contains 2,052 scenarios: 1,026 contextually appropriate cases (`_pos`) and 1,026 matched inappropriate cases (`_neg`). Each record contains:

- `name`: unique case ID; suffix indicates `_pos` or `_neg`.
- `seed`: abstract contextual integrity parameters such as sender, subject, recipient, data type, transmission method, transmission principle, and domain.
- `story`: concrete narrative context generated from the seed.
- `trace`: user instruction, available tools, ReAct-style tool trajectory, and target final action type.
- `img_metadata`: VISPR image path and sensitivity label metadata.

Use `mpci_bench.data` for stable loading helpers instead of hard-coding field assumptions.

---

## Evaluation Tasks

| Task | Script | Output / metric |
|---|---|---|
| CI probing | `mpci_bench/eval/probing.py` | Yes/No appropriateness; accuracy by seed/story/trajectory tier |
| Sensitive grounding | `mpci_bench/eval/grounding.py` | VISPR label list; case-level accuracy and label recall |
| Sensitive sharing | `mpci_bench/eval/sharing.py` | Yes/No image sensitivity; binary accuracy |
| Final-action generation | `mpci_bench/eval/action.py` | Agent final action CSV |
| Leakage judging | `mpci_bench/eval/leakage.py` | Text leakage, image leakage, helpfulness, adjusted leakage |

---

## Setup

### Requirements

Python 3.10+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Credentials

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI / Mistral / xAI keys
```

**Required for API-based model evaluation** (GPT-4o, GPT-5, GPT-5.4):
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_DEPLOYMENT_NAME`

For Azure AI Foundry / OpenAI-compatible deployments such as `gpt-5.4`, set `AZURE_OPENAI_ENDPOINT` to the `/openai/v1` endpoint and `AZURE_DEPLOYMENT_NAME=gpt-5.4`.

`--model` is treated as a model or deployment name. It does not need to match the internal deployment names used in the paper experiments.

**For local/vLLM models**, no API keys are needed — start a vLLM server and pass `--vllm-url`.

### VISPR image dependency

Images are selected from VISPR:

> Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz.  
> **Towards a Visual Privacy Advisor: Understanding and Predicting Privacy Risks in Images.**  
> *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017.

Download the VISPR train split from <https://tribhuvanesh.github.io/vpa/>, keep image IDs listed in `mpci_bench/dataset/required_image_ids.json`, and place the images at:

```text
bench/vispr/train2017/
```

Expected image paths look like `bench/vispr/train2017/2017_10665299.jpg`. VISPR images are not redistributed in this repository.

---

## Run Evaluations

Validate the artifact first:

```bash
python -m mpci_bench.validate
```

Then run through the unified CLI:

```bash
python evaluate.py action --model gpt-5.4 --output eval/action/gpt-5.4.csv
python evaluate.py leakage \
  --action-path eval/action/gpt-5.4.csv \
  --judge gpt-5.4 \
  --output eval/leakage/gpt-5.4.json
```

For a short API smoke test, run a small slice first:

```bash
python mpci_bench/eval/action.py \
  --input-path mpci_bench/dataset/mpci_bench.json \
  --output-path /tmp/mpci_action_smoke.csv \
  --model gpt-5.4 \
  --num 5 \
  --batch-size 1 \
  --parallel-workers 1
```

Other tasks use `evaluate.py probing`, `evaluate.py grounding`, and `evaluate.py sharing`; see `python evaluate.py --help`.

---

## Python API

```python
from mpci_bench.data import load_benchmark, get_image_path, is_appropriate

data = load_benchmark("mpci_bench/dataset/mpci_bench.json")
pos = [e for e in data if is_appropriate(e)]
entry = data[0]
print(len(data), len(pos), entry["name"], get_image_path(entry))
```

---

## Repository Layout

- `mpci_bench/dataset/`: benchmark JSON and required VISPR image IDs.
- `mpci_bench/eval/`: evaluation scripts.
- `mpci_bench/data.py`: dataset loading helpers.
- `mpci_bench/validate.py`: artifact validator.
- `evaluate.py`: unified CLI.
- `croissant_metadata.json`: Croissant metadata.
- `bench/vispr/train2017/`: user-provided VISPR images.
---

## Limitations and Responsible Use

- MPCI-Bench evaluates contextual-integrity behavior of agents; it is not a general privacy-knowledge or de-anonymization benchmark.
- Images come from VISPR / Flickr Creative Commons and inherit the coverage and demographic biases of that source.
- Stories and trajectories are synthetic and may inherit generation-pipeline biases.
- Agent trajectories are simulated rather than collected from real deployments.
- No real private emails, Slack messages, Drive files, or Notion pages are included.
- The benchmark is intended for evaluation and auditing, not for training privacy-invasive systems.

## License

Dataset: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Code: MIT License.

Images: Subject to the original [VISPR dataset license](https://tribhuvanesh.github.io/vpa/) (Flickr Creative Commons).
