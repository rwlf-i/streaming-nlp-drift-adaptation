# Configuration (YAML)

This project uses YAML configuration files to keep experiments **reproducible** and easy to run without editing code.

There are two types of configs:

- **Drift scenario configs** (`drift_scenarios/`) — define how the **data stream** is generated (groups/domains order, mixing, batch size).
- **Experiment configs** (`exp_configs/`) — define how to run the experiment: **model**, **scenario(s)**, **adaptation strategies**, seed, etc.

Each run stores the resolved configs in `outputs/`:
- `outputs/<experiment_dir>/experiment_used.yaml`
- `outputs/<experiment_dir>/runs/<run_id>/drift_used.yaml`
- `outputs/<experiment_dir>/runs/<run_id>/run_used.yaml`

---

## 1) Drift scenario config (`drift_scenarios/*.yaml`)

A drift scenario describes **stream formation**: data source, grouping, and an ordered **plan** of steps that generate batches.

### Template
```yaml
name: <scenario_name>
batch_size: <int>
shuffle_within_group: <true|false>

data:
  mode: <files|table>
  text_col: <text_column_name>
  label_col: <label_column_name>

  # mode: files
  sources:
    <group_name>: <path_to_file>

  # mode: table
  path: <path_to_table>
  group_col: <group_column_name>

plan:
  - { group: <group_name>, batches: <int> }
  - { mix: { <group>: <weight>, <group2>: <weight> }, batches: <int> }
```

### Fields
- `name` — scenario identifier (used in logging/output naming)
- `batch_size` — number of samples per batch in the stream
- `shuffle_within_group` — shuffle samples inside each group before batching
- `data` — data source definition
  - `mode: files`: each group is a separate file (convenient for domain/time drift)
  - `mode: table`: one table file, groups are defined by `group_col`
  - `text_col`, `label_col` — which columns contain text and labels
- `plan` — ordered steps defining how the stream changes over time:
  - `group`: take batches from one group
  - `mix`: take batches from a weighted mixture of groups
  - `batches`: how many batches to generate at this step

### Minimal examples

**Domain shift (files mode):**
```yaml
name: amazon_sudden
batch_size: 128
shuffle_within_group: true

data:
  mode: files
  text_col: text
  label_col: label
  sources:
    books: data/amazon/books.csv
    electronics: data/amazon/electronics.csv

plan:
  - { group: books, batches: 50 }
  - { group: electronics, batches: 50 }
```

**Gradual shift (mixing):**
```yaml
plan:
  - { mix: { books: 0.9, electronics: 0.1 }, batches: 10 }
  - { mix: { books: 0.5, electronics: 0.5 }, batches: 10 }
  - { mix: { books: 0.1, electronics: 0.9 }, batches: 10 }
```

---

## 2) Experiment config (`exp_configs/*.yaml`)

An experiment config defines:
- seed / pretrain setup
- one or more drift scenarios
- the model adapter and its params
- a list of adaptation strategies (optionally with multiple parameter combinations)

### Template
```yaml
name: <experiment_name>
pretrain_batches: <int>
seed: <int>

scenarios:
  - name: <scenario_alias>
    config: <path_to_drift_yaml>

model:
  class_path: <python_import_path>
  params:
    <model_param>: <value>

adaptations:
  - type: <adaptation_type>
    params:
      <parameter>: <value>
```

### Fields
- `name` — experiment name (used for output directory naming)
- `pretrain_batches` — number of initial batches used for initial training before streaming evaluation
- `seed` — random seed for reproducibility
- `scenarios` — list of scenarios to run (`name` + `config` path)
- `model` — model adapter definition (`class_path` + `params`)
- `adaptations` — list of adaptation strategies to evaluate

---

## 3) Adaptation strategies (`adaptations`)

Supported strategy types:

### 3.1) `none` — no retraining (baseline)
```yaml
adaptations:
  - type: none
```

### 3.2) `retrain` — periodic retraining every `period_batches`
Key params:
- `period_batches` — retrain period in batches
- `buffer_mode` — how to form training data:
  - `all_seen`: all processed samples so far
  - `last_window`: sliding window of recent samples
- for `last_window` you can specify window size by:
  - `window_batches` (number of recent batches) or
  - `window_size` (number of recent samples)

Example:
```yaml
- type: retrain
  params:
    period_batches: 3
    buffer_mode: last_window
    window_batches: 4
```

### 3.3) `quality_drop` — retrain when quality drops
Retraining triggers when the selected metric decreases by at least `threshold` compared to a baseline.

Typical params:
- `quality_metric`: e.g., `acc`, `f1`, etc.
- `threshold`: minimal drop to trigger retraining (e.g., `0.05`)
- `baseline_k`: how many batches after retraining are used to compute baseline level
- `confirm_k`: require drop condition for `confirm_k` consecutive batches
- buffer params: `buffer_mode`, `window_batches/window_size`
- `mode`: retrain mode (e.g., `fit` vs `update`, if supported by the model adapter)

Example:
```yaml
- type: quality_drop
  params:
    quality_metric: acc
    threshold: 0.05
    baseline_k: 3
    confirm_k: 2
    buffer_mode: last_window
    window_batches: 2
    mode: fit
```

### 3.4) `drift_detected` — retrain on drift detector signal
Uses a drift detector (e.g., ADWIN, DDM, Page-Hinkley). Retraining is triggered when the detector signals drift.

Example:
```yaml
- type: drift_detected
  params:
    drift_detector: ADWIN
    delta: 0.1
    buffer_mode: last_window
    window_batches: 10
```

---

## 4) Typical workflow
1) Create or reuse a drift scenario in `drift_scenarios/`.
2) Create an experiment config in `exp_configs/` referencing the scenario.
3) Run:
```bash
cd src
python runner.py ../exp_configs/<your_config>.yaml
```
4) Check results in `outputs/<experiment_dir>/`:
- `summary.csv` — aggregated results
- `report.html` — report
- `runs/<run_id>/batch_metrics.csv` and `events.csv` — detailed traces

## Notes
- Prefer explicit parameters in YAML (avoid relying on defaults) to keep runs comparable.
- For exact reproducibility, cite the saved `*_used.yaml` files from `outputs/`.
