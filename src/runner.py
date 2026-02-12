# runner.py

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from data_stream import ExperimentConfig, DriftScenario, ScenarioBuilder, DataStream
from adaptation import build_strategy, mode_strategy, AdaptationStrategy, TriggerRetrainByDrift, TriggerRetrainByMetric
from metrics_logger import ExperimentLogger, ResourceMonitor



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def ensure_src_on_syspath() -> None:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))


def parse_class_path(class_path: str) -> Tuple[str, str]:
    cp = class_path.strip()
    if ":" in cp:
        module_name, class_name = cp.split(":", 1)
        module_name = module_name.strip()
        class_name = class_name.strip()
        if not module_name or not class_name:
            raise ValueError(f"Invalid class_path: '{class_path}'")
        return module_name, class_name

    if "." not in cp:
        raise ValueError(f"Invalid class_path: '{class_path}'. Expected module.Class")

    module_name, class_name = cp.rsplit(".", 1)
    module_name = module_name.strip()
    class_name = class_name.strip()
    if not module_name or not class_name:
        raise ValueError(f"Invalid class_path: '{class_path}'")
    return module_name, class_name


def load_class(class_path: str):
    module_name, class_name = parse_class_path(class_path)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



def format_adaptation_name(adapt_cfg: Dict[str, Any]) -> str:
    atype = str(adapt_cfg.get("type", "none"))
    params = adapt_cfg.get("params", {}) or {}

    if atype == "none":
        return "none"

    period = params.get("period_batches")
    buffer_mode = params.get("buffer_mode")
    window_size = params.get("window_size")
    window_batches = params.get("window_batches")
    mode = params.get("mode","fit")
    

    if atype == "quality_drop":
        metric = params.get("metric", "acc")  
        threshold = params.get("threshold", "0.05")
        baseline_k = params.get("baseline_k", "3")
        confirm_k = params.get("confirm_k", "2")
        parts = [atype]
        if buffer_mode is not None:
            parts.append(str(buffer_mode))
        if window_batches is not None:
            parts.append(f"wb{window_batches}")
        if window_size is not None:
            parts.append(f"w{window_size}")
        buff = "_".join(parts)
        return f"quality_drop_{metric}_th{threshold}_bk{baseline_k}_ck{confirm_k}_{buff}_{mode}"


    if atype == "drift_detected":
        drift_detector = params.get("drift_detector", "ADWIN")  
        parts = [f"drift_detected_{drift_detector}"]

        if buffer_mode is not None:
            parts.append(str(buffer_mode))
        if window_batches is not None:
            parts.append(f"wb{window_batches}")
        if window_size is not None:
            parts.append(f"w{window_size}")
        parts.append(mode)
        return "_".join(parts)


    parts = [atype]
    if period is not None:
        parts.append(f"p{period}")
    if buffer_mode is not None:
        parts.append(str(buffer_mode))
    if window_batches is not None:
        parts.append(f"wb{window_batches}")
    if window_size is not None:
        parts.append(f"w{window_size}")
    parts.append(mode)
    return "_".join(parts)



class ExperimentRunner:
    def __init__(self, experiment_yaml_path: str):
        ensure_src_on_syspath()

        self.experiment_yaml_path = str(Path(experiment_yaml_path).resolve())
        self.base_dir = Path(self.experiment_yaml_path).parent

        self.exp = ExperimentConfig(self.experiment_yaml_path)

        model_cfg = self.exp.model or {}
        self.model_class_path: str = model_cfg.get("class_path", "")
        self.model_params: Dict[str, Any] = model_cfg.get("params", {}) or {}

        if not self.model_class_path:
            raise ValueError("Experiment config must include model.class_path")

        self.adaptations = self.exp.adaptations if self.exp.adaptations else [{"type": "none"}]
        self.scenarios = self.exp.scenarios

        self.exp_logger = ExperimentLogger(self.experiment_yaml_path, self.exp.name)

        self.monitor = ResourceMonitor()

    @staticmethod
    def _print_header(experiment_name: str, drift_name: str, adapt_name: str) -> None:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {experiment_name} | DRIFT: {drift_name} | ADAPTATION: {adapt_name}")
        print("=" * 80)

    @staticmethod
    def _print_groups(df_by_group) -> None:
        print("Groups loaded:")
        for g, df in df_by_group.items():
            print(f"  - {g:15s}: n={len(df)}")

    @staticmethod
    def _print_pretrain_metrics(y_true, y_pred) -> Tuple[float, float]:
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        print(f"Pretrain metrics: accuracy={acc:.4f}, f1_macro={f1m:.4f}")
        return acc, f1m

    @staticmethod
    def _print_batch_metrics(batch: dict, acc: float, f1m: float) -> None:
        meta = batch["meta"]
        bid = batch["batch_id"]
        group = batch["group"]
        n = meta["n"]

        step_kind = meta.get("step_kind")
        pos_share = meta.get("pos_share")
        label_dist = meta.get("label_dist")

        print(
            f"[{bid:3d}] "
            f"step={str(step_kind):5s} group={group:10s} n={n:4d} "
            f"acc={acc:.4f} f1_macro={f1m:.4f} "
            f"pos_share={pos_share} label_dist={label_dist}"
        )

        if step_kind == "mix":
            print(f"      mix_weights={meta.get('mix_weights')}")
            print(f"      mix_alloc={meta.get('mix_alloc')}")

    def _resolve_drift_path(self, drift_cfg_path: str) -> str:
        p = Path(drift_cfg_path)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return str(p)

    def _build_stream(self, drift_yaml_path: str) -> Tuple[DriftScenario, DataStream, dict]:
        drift = DriftScenario(drift_yaml_path)
        builder = ScenarioBuilder(drift)
        df_by_group = builder.build()
        self._print_groups(df_by_group)
        stream = DataStream(drift, df_by_group)
        return drift, stream, df_by_group

    def _create_model(self):
        ModelCls = load_class(self.model_class_path)
        return ModelCls(**dict(self.model_params))

    def _pretrain(self, stream: DataStream, model, pretrain_batches: int) -> Tuple[list, list]:
        texts_pre, labels_pre = [], []
        used = 0

        while stream.has_next() and used < pretrain_batches:
            batch = stream.next_batch()
            texts_pre.extend(batch["texts"])
            labels_pre.extend(batch["labels"])
            used += 1

        if not texts_pre:
            return [], []

        print(f"Pretrain: batches_used={used}, n_examples={len(texts_pre)}")
        return texts_pre, labels_pre



    def _evaluate_stream(self, stream: DataStream, model,  mode, strategy: AdaptationStrategy, run_logger) -> Dict[str, Any]:
        print("\nStream evaluation on remaining batches:")

        n_batches = 0
        n_events = 0
        sum_acc = 0.0
        sum_f1 = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        last_acc: Optional[float] = None
        last_f1: Optional[float] = None
        retrain_count = 0
        retrain_wall_sum = 0.0
        retrain_cpu_sum = 0.0
        retrain_train_n_sum = 0

        while stream.has_next():
            batch = stream.next_batch()

            s0 = self.monitor.snap()
            y_pred = model.predict(batch["texts"])
            s1 = self.monitor.snap()
            pred_perf = ResourceMonitor.delta(s0, s1)

            y_true = batch["labels"]
            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

            self._print_batch_metrics(batch, acc, f1m)

            batch_metrics = {
                "acc": acc,
                "f1_macro": f1m,
                "precision": precision,
                "recall": recall,
            }

            batch["predictions"] = y_pred
            run_logger.log_batch(batch=batch, metrics=batch_metrics, perf=pred_perf)

            n_batches += 1
            sum_acc += float(acc)
            sum_f1 += float(f1m)
            sum_precision += float(precision)
            sum_recall += float(recall)
            last_acc = float(acc)
            last_f1 = float(f1m)

            decision = strategy.observe_batch(
                batch=batch,
                batch_metrics=batch_metrics,
            )

            if decision.should_retrain:
                train_texts, train_labels = strategy.get_train_data()
                train_n = len(train_texts)

                uniq = set(train_labels)
                if train_n == 0:
                    run_logger.log_event(
                        event_type="retrain_skip",
                        batch_id=int(batch["batch_id"]),
                        reason="buffer_empty",
                        train_n=0,
                        info={"decision": decision.info},
                        perf={"wall_sec": 0.0, "cpu_sec": 0.0, "cpu_pct": None, "rss_mb": pred_perf.get("rss_mb")},
                    )
                    n_events += 1
                    print("  [adaptation] skip retrain: buffer empty")
                    continue

                if len(uniq) < 2:
                    only = next(iter(uniq)) if uniq else None
                    run_logger.log_event(
                        event_type="retrain_skip",
                        batch_id=int(batch["batch_id"]),
                        reason="one_class_in_buffer",
                        train_n=train_n,
                        info={"only_class": only, "decision": decision.info},
                        perf={"wall_sec": 0.0, "cpu_sec": 0.0, "cpu_pct": None, "rss_mb": pred_perf.get("rss_mb")},
                    )
                    n_events += 1
                    print(f"  [adaptation] skip retrain: buffer has only one class={only} (n={train_n})")
                    continue

                f0 = self.monitor.snap()

                if mode == "update":
                    model.update(train_texts, train_labels)
                else:
                    model.fit(train_texts, train_labels)  

                if isinstance(strategy, TriggerRetrainByMetric):
                    strategy.on_retrain()

                f1 = self.monitor.snap()
                fit_perf = ResourceMonitor.delta(f0, f1)
                retrain_count += 1
                retrain_wall_sum += float(fit_perf.get("wall_sec") or 0.0)
                retrain_cpu_sum += float(fit_perf.get("cpu_sec") or 0.0)
                retrain_train_n_sum += int(train_n)

                run_logger.log_event(
                    event_type=f"retrain_{mode}",
                    batch_id=int(batch["batch_id"]),
                    reason=decision.reason,
                    train_n=train_n,
                    info=dict(decision.info),
                    perf=fit_perf,
                )
                n_events += 1

                print(f"  [adaptation] retrain: reason={decision.reason} train_n={train_n} info={decision.info}")

        mean_acc = (sum_acc / n_batches) if n_batches else None
        mean_f1 = (sum_f1 / n_batches) if n_batches else None
        mean_precision = (sum_precision / n_batches) if n_batches else None
        mean_recall = (sum_recall / n_batches) if n_batches else None
        retrain_wall_mean = (retrain_wall_sum / retrain_count) if retrain_count else None
        retrain_cpu_mean = (retrain_cpu_sum / retrain_count) if retrain_count else None

        return {
            "n_batches": n_batches,
            "n_events": n_events,
            "mean_acc": mean_acc,
            "mean_f1_macro": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "final_acc": last_acc,
            "final_f1_macro": last_f1,
            "retrain_count": retrain_count,
            "retrain_wall_sum_sec": retrain_wall_sum,
            "retrain_cpu_sum_sec": retrain_cpu_sum,
            "retrain_wall_mean_sec": retrain_wall_mean,
            "retrain_cpu_mean_sec": retrain_cpu_mean,
            "retrain_train_n_sum": retrain_train_n_sum,
        }


  


    def _run_one(self, drift_name: str, drift_yaml_path: str, adaptation_cfg: Dict[str, Any]) -> None:
        adapt_name = format_adaptation_name(adaptation_cfg)
        self._print_header(self.exp.name, drift_name, adapt_name)

        set_global_seed(int(self.exp.seed))

        run_logger = self.exp_logger.start_run(
            drift_yaml_path=drift_yaml_path,
            drift_name=drift_name,
            adaptation_name=adapt_name, 
            adaptation_cfg=dict(adaptation_cfg or {}),
            seed=int(self.exp.seed),
            model_class_path=self.model_class_path,
            model_params=dict(self.model_params),
        )

        drift, stream, _ = self._build_stream(drift_yaml_path)
        model = self._create_model()

        strategy = build_strategy(adaptation_cfg)

        mode = mode_strategy(adaptation_cfg)
        print("MODE ", mode)

        texts_pre, labels_pre = self._pretrain(stream, model, int(self.exp.pretrain_batches))
        if not texts_pre:
            print("No data for pretrain. Skip run.")
            run_logger.log_info("Skip run: no pretrain data")
            totals = run_logger.finish()
            self.exp_logger.append_summary({
                "ts": "",
                "experiment_name": self.exp.name,
                "run_id": run_logger.run_id,
                "drift_name": drift_name,
                "adaptation_name": adapt_name, 
                "seed": int(self.exp.seed),
                "batch_metrics_path": str(run_logger.batch_metrics_path),
                "events_path": str(run_logger.events_path),
                "run_log_path": str(run_logger.log_path),
                "n_batches": 0,
                "n_events": 0,
                "mean_acc": "",
                "mean_f1_macro": "",
                "final_acc": "",
                "final_f1_macro": "",
                **totals,
            })
            return

        pf0 = self.monitor.snap()
        model.fit(texts_pre, labels_pre)
        pf1 = self.monitor.snap()
        prefit_perf = ResourceMonitor.delta(pf0, pf1)

        y_pred_train = model.predict(texts_pre)
        pre_acc, pre_f1 = self._print_pretrain_metrics(labels_pre, y_pred_train)

        run_logger.log_event(
            event_type="pretrain_fit",
            batch_id=None,
            reason="pretrain",
            train_n=len(texts_pre),
            info={"pretrain_batches": int(self.exp.pretrain_batches), "pre_acc": pre_acc, "pre_f1_macro": pre_f1},
            perf=prefit_perf,
        )

        strategy.on_pretrain_end(texts_pre, labels_pre)


        aggr = self._evaluate_stream(stream, model, mode, strategy, run_logger)

        totals = run_logger.finish()

        self.exp_logger.append_summary({
            "ts": "",
            "experiment_name": self.exp.name,
            "run_id": run_logger.run_id,
            "drift_name": drift_name,
            "adaptation_name": adapt_name,
            "seed": int(self.exp.seed),
            "batch_metrics_path": str(run_logger.batch_metrics_path),
            "events_path": str(run_logger.events_path),
            "run_log_path": str(run_logger.log_path),
            **aggr,
            **totals,
        })

        print(f"\nRun saved to: {run_logger.run_dir}")
        print(f"Summary updated: {self.exp_logger.summary_path}")




    def run(self) -> None:
        print(f"\nOutputs dir: {self.exp_logger.experiment_dir}")

        for sc in self.scenarios:
            drift_name = sc.get("name", "drift")
            drift_cfg_path = sc.get("config")
            if not drift_cfg_path:
                print(f"[skip] Scenario '{drift_name}': missing 'config'")
                continue

            drift_yaml_path = self._resolve_drift_path(drift_cfg_path)

            for adapt_cfg in self.adaptations:
                self._run_one(
                    drift_name=str(drift_name),
                    drift_yaml_path=str(drift_yaml_path),
                    adaptation_cfg=dict(adapt_cfg or {}),
                )
        report_path = self.exp_logger.generate_report()
        print(f"\nReport generated: {report_path}")



if __name__ == "__main__":
    print("start1")
    if len(sys.argv) < 2:
        print("Usage: python runner.py <experiment_yaml_path>")
        sys.exit(1)

    print("start")
    ExperimentRunner(sys.argv[1]).run()