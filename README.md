## About
This repository contains a reproducible experimental testbed for studying **data drift** and **model adaptation (retraining)** in **streaming NLP text classification**.  Experiments are defined via YAML configs (drift scenarios + experiment settings) and produce fixed artifacts (used configs, metrics, logs, reports) to ensure comparability and repeatability.  The main goal is to compare adaptation strategies under controlled drift scenarios (domain, temporal, label) using **quality metrics** (e.g., accuracy, F1-macro) and **cost metrics** (e.g., retraining time, number of retrains, training volume).


## Data
The experiments use four public NLP datasets representing different drift sources:

- **Amazon Reviews 2023** (domain drift in product categories): https://amazon-reviews-2023.github.io/
- **HuffPost (Wild-Time)** (temporal drift in news headlines): https://github.com/huaxiuyao/Wild-Time
- **CFPB Consumer Complaints** (real-world complaint texts): https://www.consumerfinance.gov/data-research/consumer-complaints/
- **TREC07 Spam (River wrapper)** (streaming email spam classification): https://riverml.xyz/dev/api/datasets/TREC07/

> Note: dataset licenses and redistribution rules may differ. This repository includes only datasets that permit redistribution (e.g., HuffPost); other datasets are referenced via official sources and are not stored in the repository.

## Project structure

* `src/` — core implementation (runner, stream generation, adaptation logic, metrics/logging)
* `data/` — prepared datasets (if available locally)
* `drift_scenarios/` — YAML drift scenario definitions
* `exp_configs/` — YAML experiment configurations
* `outputs/` — experiment results and run artifacts

## Configuration (YAML)
The testbed is configured via YAML files:

* `drift_scenarios/` — drift scenario configs (stream composition and schedule)
* `exp_configs/` — experiment configs (model + scenario + adaptation strategies)

Each run stores the resolved configs in `outputs/<experiment_dir>/experiment_used.yaml` and `outputs/<experiment_dir>/runs/<run_id>/*_used.yaml`.

See the full configuration reference in [`docs/CONFIGS.md`](docs/CONFIGS.md).


## Outputs structure
Each experiment creates a directory:

- `outputs/<experiment_dir>/`
  - `experiment_used.yaml` — the exact experiment config used for this experiment
  - `summary.csv` — aggregated results across runs/configurations (quality + costs)
  - `report.html` — rendered experiment report
  - `runs/<run_id>/` — artifacts of an individual run:
    - `drift_used.yaml` — drift scenario config actually used
    - `run_used.yaml` — resolved run configuration (strategy parameters, seed, etc.)
    - `batch_metrics.csv` — per-batch metrics over the stream (time series)
    - `events.csv` — drift/retrain trigger events with timestamps/batch ids
    - `run.log` — run logs

## Run (example)
```bash
git clone https://github.com/rwlf-i/streaming-nlp-drift-adaptation.git
cd streaming-nlp-drift-adaptation
pip install -r requirements.txt
cd src

python runner.py ../exp_configs/huffpost_lr_gradual.yaml
```

## Authors
- Malova S. Anastasia — main author, Student (SPbPU ICCS)
- Vladimir A. Parkhomenko — co-author (paper) and advisor, Senior Lecturer (SPbPU ICCS)


## Warranty
The contributors provide no warranty for the use of this software. Use it at your own risk.

## License
This project is licensed under the MIT License. See `LICENSE`.




