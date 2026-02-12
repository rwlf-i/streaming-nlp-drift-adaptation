# data_stream.py
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


def read_any(path: str | Path) -> pd.DataFrame:
    """Read csv/csv.gz/jsonl/jsonl.gz into DataFrame."""
    path = str(path)
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        comp = "gzip" if path.endswith(".gz") else None
        return pd.read_csv(path, compression=comp)
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        comp = "gzip" if path.endswith(".gz") else None
        return pd.read_json(path, lines=True, compression=comp)
    raise ValueError(f"Unsupported file format: {path}")


def _resolve_path(base_dir: Path, maybe_rel: str) -> str:
    p = Path(maybe_rel).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _normalize_text_label_df(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Return df with columns [text, label]; drop invalid; coerce label to int."""
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in data.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in data.")

    out = df[[text_col, label_col]].copy()
    out = out.rename(columns={text_col: "text", label_col: "label"})

    out["text"] = out["text"].astype(str)

    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["text", "label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def _label_dist(labels: List[int]) -> Dict[int, int]:
    d: Dict[int, int] = {}
    for y in labels:
        d[int(y)] = d.get(int(y), 0) + 1
    return d


def _pos_share_if_binary(labels: List[int]) -> Optional[float]:
    if not labels:
        return None
    s = set(labels)
    if s.issubset({0, 1}):
        return sum(labels) / len(labels)
    return None



class ExperimentConfig:

    def __init__(self, config_path: str):
        self.config_path = config_path
        config = self._load_config(config_path)

        self.name = config.get("name", "experiment")
        self.pretrain_batches = config.get("pretrain_batches", 5)
        self.scenarios = config.get("scenarios", [])
        self.model = config.get("model", {})
        self.adaptations = config.get("adaptations", [])
        self.seed = config.get("seed", 42)

        self._validate_config()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file)
            return cfg or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Experiment config '{config_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    def _validate_config(self):
        if not isinstance(self.seed, int):
            raise ValueError("Experiment seed should be an integer.")
        if not isinstance(self.pretrain_batches, int) or self.pretrain_batches <= 0:
            raise ValueError(
                f"Invalid pretrain_batches={self.pretrain_batches}. Must be positive int."
            )
        if not isinstance(self.scenarios, list) or len(self.scenarios) == 0:
            raise ValueError("At least one scenario must be specified in 'scenarios'.")

        if not isinstance(self.model, dict):
            raise ValueError("Field 'model' must be a dict.")
        if not isinstance(self.adaptations, list):
            raise ValueError("Field 'adaptations' must be a list.")


class DriftScenario:

    def __init__(self, config_path: str):
        self.config_path = config_path
        cfg = self._load_config(config_path)

        self.name = cfg.get("name", "drift_scenario")
        self.batch_size = cfg.get("batch_size", 500)
        self.shuffle_within_group = cfg.get("shuffle_within_group", False)
        self.seed = cfg.get("seed", 42)

        self.data = cfg.get("data", {}) or {}
        self.plan = cfg.get("plan", []) or []

        self._validate_config()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file)
            return cfg or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Drift scenario '{config_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    def _validate_config(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Scenario name must be a non-empty string.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size={self.batch_size}. Must be positive int.")

        if not isinstance(self.shuffle_within_group, bool):
            raise ValueError("shuffle_within_group must be boolean.")

        if not isinstance(self.seed, int):
            raise ValueError("Scenario seed must be an integer.")

        if "mode" not in self.data:
            raise ValueError("Field data.mode is required (files|table).")

        mode = self.data["mode"]
        if mode not in ("files", "table"):
            raise ValueError("Invalid data.mode. Supported: 'files' or 'table'.")

        for k in ("text_col", "label_col"):
            if k not in self.data or not isinstance(self.data[k], str) or not self.data[k]:
                raise ValueError(f"data.{k} must be a non-empty string.")

        if mode == "files":
            sources = self.data.get("sources")
            if not isinstance(sources, dict) or len(sources) == 0:
                raise ValueError("For mode=files, data.sources must be a non-empty dict.")

        if mode == "table":
            if "source" not in self.data or not isinstance(self.data["source"], str):
                raise ValueError("For mode=table, data.source must be specified.")
            if "group_col" not in self.data or not isinstance(self.data["group_col"], str):
                raise ValueError("For mode=table, data.group_col must be specified.")

        if not isinstance(self.plan, list) or len(self.plan) == 0:
            raise ValueError("Field 'plan' must be a non-empty list.")

        for step in self.plan:
            if not isinstance(step, dict):
                raise ValueError("Each plan step must be a dict.")
            if "batches" not in step or not isinstance(step["batches"], int) or step["batches"] <= 0:
                raise ValueError("Each plan step must have positive integer 'batches'.")
            if "group" not in step and "mix" not in step:
                raise ValueError("Each plan step must include either 'group' or 'mix'.")



class ScenarioBuilder:


    def __init__(self, scenario: DriftScenario):
        self.scenario = scenario
        self.base_dir = Path(scenario.config_path).resolve().parent

    def build(self) -> Dict[str, pd.DataFrame]:
        mode = self.scenario.data["mode"]
        text_col = self.scenario.data["text_col"]
        label_col = self.scenario.data["label_col"]

        if mode == "files":
            sources: Dict[str, str] = self.scenario.data["sources"]
            df_by_group: Dict[str, pd.DataFrame] = {}

            for group, p in sources.items():
                file_path = _resolve_path(self.base_dir, p)
                df = read_any(file_path)
                df = _normalize_text_label_df(df, text_col=text_col, label_col=label_col)

                if self.scenario.shuffle_within_group:
                    df = df.sample(frac=1.0, random_state=self.scenario.seed).reset_index(drop=True)

                df_by_group[str(group)] = df

            return df_by_group

        table_path = _resolve_path(self.base_dir, self.scenario.data["source"])
        group_col = self.scenario.data["group_col"]

        df_all = read_any(table_path)

        if group_col not in df_all.columns:
            raise ValueError(f"Missing group column '{group_col}' in table source.")
        if text_col not in df_all.columns or label_col not in df_all.columns:
            raise ValueError(f"Missing '{text_col}' or '{label_col}' in table source.")

        df_all = df_all[[group_col, text_col, label_col]].copy()
        df_all = df_all.rename(columns={text_col: "text", label_col: "label", group_col: "group"})

        df_all["text"] = df_all["text"].astype(str)
        df_all["label"] = pd.to_numeric(df_all["label"], errors="coerce")
        df_all = df_all.dropna(subset=["group", "text", "label"]).reset_index(drop=True)
        df_all["label"] = df_all["label"].astype(int)
        df_all["group"] = df_all["group"].astype(str)

        df_by_group: Dict[str, pd.DataFrame] = {}
        for g, df_g in df_all.groupby("group", sort=False):
            df_g = df_g[["text", "label"]].reset_index(drop=True)
            if self.scenario.shuffle_within_group:
                df_g = df_g.sample(frac=1.0, random_state=self.scenario.seed).reset_index(drop=True)
            df_by_group[g] = df_g

        return df_by_group


class DataStream:

    def __init__(self, scenario: DriftScenario, df_by_group: Dict[str, pd.DataFrame]):
        self.scenario = scenario
        self.df_by_group = df_by_group
        self.batch_size = scenario.batch_size
        self.seed = scenario.seed

        self.cursor: Dict[str, int] = {g: 0 for g in df_by_group.keys()}
        self.plan_steps: List[dict] = self._expand_plan(scenario.plan)
        self.batch_counter = 0

    @staticmethod
    def _expand_plan(plan_items: List[dict]) -> List[dict]:
        steps: List[dict] = []
        for item in plan_items:
            k = int(item["batches"])
            if k <= 0:
                raise ValueError(f"batches must be positive, got {k}")

            if "group" in item:
                g = str(item["group"])
                for _ in range(k):
                    steps.append({"kind": "group", "group": g})
            elif "mix" in item:
                weights = {str(g): float(w) for g, w in item["mix"].items()}
                total = sum(weights.values())
                if total <= 0:
                    raise ValueError(f"sum of mix weights must be > 0, got {total}")
                weights = {g: w / total for g, w in weights.items()}
                for _ in range(k):
                    steps.append({"kind": "mix", "weights": weights})
            else:
                raise ValueError(f"Plan step must have 'group' or 'mix': {item}")
        return steps

    def _group_has_data(self, g: str) -> bool:
        return g in self.df_by_group and self.cursor[g] < len(self.df_by_group[g])

    def has_next(self) -> bool:
        if not self.plan_steps:
            return False
        for step in self.plan_steps:
            if step["kind"] == "group":
                if self._group_has_data(step["group"]):
                    return True
            else:
                if any(self._group_has_data(g) for g in step["weights"].keys()):
                    return True
        return False

    def next_batch(self) -> dict:
        if not self.has_next():
            raise StopIteration("End of stream.")

        step_idx: Optional[int] = None
        for i, step in enumerate(self.plan_steps):
            if step["kind"] == "group":
                if self._group_has_data(step["group"]):
                    step_idx = i
                    break
            else:
                if any(self._group_has_data(g) for g in step["weights"].keys()):
                    step_idx = i
                    break

        if step_idx is None:
            raise StopIteration("End of stream.")

        step = self.plan_steps.pop(step_idx)
        bs = self.batch_size

        texts: List[str] = []
        labels: List[int] = []

        group_out: str
        step_kind: str = step["kind"]

        group_name: Optional[str] = None
        mix_weights: Optional[Dict[str, float]] = None
        mix_alloc: Optional[Dict[str, int]] = None

        if step_kind == "group":
            g = step["group"]
            if g not in self.df_by_group:
                raise StopIteration(f"Group '{g}' not found in data.")

            start = self.cursor[g]
            end = start + bs
            df = self.df_by_group[g].iloc[start:end]
            self.cursor[g] = start + len(df)

            if df.empty:
                raise StopIteration("Empty batch (group).")

            texts = df["text"].astype(str).tolist()
            labels = df["label"].astype(int).tolist()

            group_out = g
            group_name = g

        else:
            mix_weights = dict(step["weights"])

            raw = {g: mix_weights[g] * bs for g in mix_weights}
            base = {g: int(math.floor(raw[g])) for g in raw}
            taken = sum(base.values())
            leftover = bs - taken

            frac_sorted = sorted(raw.keys(), key=lambda g: (raw[g] - base[g]), reverse=True)
            mix_alloc = base.copy()
            for g in frac_sorted:
                if leftover <= 0:
                    break
                mix_alloc[g] += 1
                leftover -= 1

            rng = random.Random(self.seed + self.batch_counter + 1)

            for g, k in mix_alloc.items():
                if k <= 0:
                    continue
                if g not in self.df_by_group:
                    continue

                start = self.cursor[g]
                end = start + k
                df_g = self.df_by_group[g].iloc[start:end]
                self.cursor[g] = start + len(df_g)

                if not df_g.empty:
                    texts.extend(df_g["text"].astype(str).tolist())
                    labels.extend(df_g["label"].astype(int).tolist())

            if len(texts) == 0:
                raise StopIteration("Empty batch (mix).")

            idx = list(range(len(texts)))
            rng.shuffle(idx)
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]

            group_out = "mix"

        self.batch_counter += 1

        dist = _label_dist(labels)
        pos_share = _pos_share_if_binary(labels)

        meta = {
            "n": len(texts),
            "label_dist": dist,
            "pos_share": pos_share,
            "scenario_name": self.scenario.name,
            "step_kind": step_kind,
            "group_name": group_name, 
            "mix_weights": mix_weights,
            "mix_alloc": mix_alloc, 
        }

        return {
            "batch_id": self.batch_counter,
            "texts": texts,
            "labels": labels,
            "group": group_out,
            "meta": meta,
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
