# metrics_logger.py
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import psutil  
except Exception:
    psutil = None 


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(s: str) -> str:
    s = str(s).strip()
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        s = s.replace(ch, "_")
    s = s.replace(" ", "_")
    return s


def _json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return json.dumps(str(v), ensure_ascii=False)


@dataclass
class Snapshot:
    wall: float
    cpu: float
    rss_bytes: Optional[int]


class ResourceMonitor:
    def __init__(self):
        self._p = psutil.Process() if psutil is not None else None  # type: ignore

    def snap(self) -> Snapshot:
        wall = time.time()
        if self._p is not None:
            ct = self._p.cpu_times()
            cpu = float(ct.user + ct.system)
            rss = int(self._p.memory_info().rss)
            return Snapshot(wall=wall, cpu=cpu, rss_bytes=rss)
        else:
            cpu = float(time.process_time())
            return Snapshot(wall=wall, cpu=cpu, rss_bytes=None)

    @staticmethod
    def delta(a: Snapshot, b: Snapshot) -> Dict[str, Optional[float]]:
        wall = b.wall - a.wall
        cpu = b.cpu - a.cpu
        cpu_pct = None
        if wall > 1e-9:
            cpu_pct = (cpu / wall) * 100.0
        rss_mb = None
        if b.rss_bytes is not None:
            rss_mb = b.rss_bytes / (1024 * 1024)
        return {
            "wall_sec": wall,
            "cpu_sec": cpu,
            "cpu_pct": cpu_pct,
            "rss_mb": rss_mb,
        }


class RunLogger:
    def __init__(
        self,
        experiment_dir: Path,
        run_id: str,
        run_dir: Path,
        drift_yaml_path: str,
        run_used: Dict[str, Any],
    ):
        self.experiment_dir = experiment_dir
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(drift_yaml_path, self.run_dir / "drift_used.yaml")

        (self.run_dir / "run_used.yaml").write_text(_json(run_used), encoding="utf-8")

        self.batch_metrics_path = self.run_dir / "batch_metrics.csv"
        self.events_path = self.run_dir / "events.csv"
        self.log_path = self.run_dir / "run.log"

        self._batch_fieldnames = [
            "ts",
            "run_id",
            "batch_id",
            "group",
            "step_kind",
            "n",
            "pos_share",
            "label_dist",
            "acc",
            "f1_macro",
            "precision",
            "recall", 
            "mix_weights",
            "mix_alloc",
            "wall_pred_sec",
            "cpu_pred_sec",
            "cpu_pred_pct",
            "rss_mb",
        ]

        self._event_fieldnames = [
            "ts",
            "run_id",
            "event_type",
            "batch_id",
            "reason",
            "train_n",
            "info",
            "wall_fit_sec",
            "cpu_fit_sec",
            "cpu_fit_pct",
            "rss_mb",
        ]

        self._ensure_csv(self.batch_metrics_path, self._batch_fieldnames)
        self._ensure_csv(self.events_path, self._event_fieldnames)

        # logging
        self._logger = logging.getLogger(f"run.{run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        self.monitor = ResourceMonitor()
        self._start_snap = self.monitor.snap()
        self._peak_rss_mb: Optional[float] = None

        self._logger.info(f"Run started: run_id={run_id}")
        self._logger.info(f"run_dir={self.run_dir}")

    @staticmethod
    def _ensure_csv(path: Path, fieldnames: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    def _append_csv(self, path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
        rss = row.get("rss_mb")
        if isinstance(rss, (int, float)):
            if self._peak_rss_mb is None or rss > self._peak_rss_mb:
                self._peak_rss_mb = float(rss)

        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)


    def log_batch(
        self,
        batch: dict,
        metrics: Dict[str, Any],
        perf: Dict[str, Optional[float]],
    ) -> None:
        meta = batch.get("meta", {}) or {}
        row = {
            "ts": _now_ts(),
            "run_id": self.run_id,
            "batch_id": int(batch["batch_id"]),
            "group": str(batch["group"]),
            "step_kind": str(meta.get("step_kind")),
            "n": int(meta.get("n", len(batch.get("texts", [])))),
            "pos_share": meta.get("pos_share"),
            "label_dist": _json(meta.get("label_dist")),
            "acc": metrics.get("acc"),
            "f1_macro": metrics.get("f1_macro"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),  
            "mix_weights": _json(meta.get("mix_weights")) if meta.get("mix_weights") is not None else "",
            "mix_alloc": _json(meta.get("mix_alloc")) if meta.get("mix_alloc") is not None else "",
            "wall_pred_sec": perf.get("wall_sec"),
            "cpu_pred_sec": perf.get("cpu_sec"),
            "cpu_pred_pct": perf.get("cpu_pct"),
            "rss_mb": perf.get("rss_mb"),
        }
        self._append_csv(self.batch_metrics_path, self._batch_fieldnames, row)


    def log_event(
        self,
        event_type: str,
        batch_id: Optional[int],
        reason: str,
        train_n: int,
        info: Dict[str, Any],
        perf: Dict[str, Optional[float]],
    ) -> None:
        row = {
            "ts": _now_ts(),
            "run_id": self.run_id,
            "event_type": str(event_type),
            "batch_id": int(batch_id) if batch_id is not None else "",
            "reason": str(reason),
            "train_n": int(train_n),
            "info": _json(info),
            "wall_fit_sec": perf.get("wall_sec"),
            "cpu_fit_sec": perf.get("cpu_sec"),
            "cpu_fit_pct": perf.get("cpu_pct"),
            "rss_mb": perf.get("rss_mb"),
        }
        self._append_csv(self.events_path, self._event_fieldnames, row)

        self._logger.info(
            f"EVENT {event_type} batch_id={batch_id} reason={reason} train_n={train_n} "
            f"wall={perf.get('wall_sec')} cpu={perf.get('cpu_sec')} rss_mb={perf.get('rss_mb')} info={info}"
        )


    def log_info(self, msg: str) -> None:
        self._logger.info(msg)

    def finish(self) -> Dict[str, Any]:
        end = self.monitor.snap()
        total = ResourceMonitor.delta(self._start_snap, end)
        self._logger.info(f"Run finished. total={total} peak_rss_mb={self._peak_rss_mb}")
        return {
            "total_wall_sec": total.get("wall_sec"),
            "total_cpu_sec": total.get("cpu_sec"),
            "peak_rss_mb": self._peak_rss_mb,
        }


class ExperimentLogger:
    """
    Creates experiment directory:
      outputs/<timestamp>__<experiment_name>/
        experiment_used.yaml
        summary.csv
        runs/<run_id>/...
    """

    def __init__(self, experiment_yaml_path: str, experiment_name: str):
        self.experiment_yaml_path = str(Path(experiment_yaml_path).resolve())
        self.experiment_name = str(experiment_name)

        src_dir = Path(__file__).resolve().parent 
        root_dir = src_dir.parent 
        outputs_root = root_dir / "outputs" 
        outputs_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        exp_dirname = f"{ts}__{_safe_name(self.experiment_name)}"
        self.experiment_dir = outputs_root / exp_dirname
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # copy experiment config
        shutil.copyfile(self.experiment_yaml_path, self.experiment_dir / "experiment_used.yaml")

        self.runs_dir = self.experiment_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.experiment_dir / "summary.csv"
        self._summary_fields = [
            "ts",
            "experiment_name",
            "run_id",
            "drift_name",
            "adaptation_name",
            "seed",
            "batch_metrics_path",
            "events_path",
            "run_log_path",
            "n_batches",
            "n_events",
            "mean_acc",
            "mean_f1_macro",
            "mean_precision",
            "mean_recall",
            "final_acc",
            "final_f1_macro",
            "total_wall_sec",
            "total_cpu_sec",
            "peak_rss_mb",
            "retrain_count",
            "retrain_wall_sum_sec",
            "retrain_cpu_sum_sec",
            "retrain_wall_mean_sec",
            "retrain_cpu_mean_sec",
            "retrain_train_n_sum",
        ]
        if not self.summary_path.exists():
            with self.summary_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._summary_fields)
                w.writeheader()

    def make_run_id(self, drift_name: str, adaptation_name: str) -> str:
        suffix = time.strftime("%H%M%S")
        base = f"{_safe_name(drift_name)}__{_safe_name(adaptation_name)}__{suffix}"
        return base

    

    def start_run(
        self,
        drift_yaml_path: str,
        drift_name: str,
        adaptation_name: str,
        adaptation_cfg: Dict[str, Any],
        seed: int,
        model_class_path: str,
        model_params: Dict[str, Any],
    ) -> RunLogger:
        run_id = self.make_run_id(drift_name, adaptation_name)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_used = {
            "experiment_name": self.experiment_name,
            "run_id": run_id,
            "seed": seed,
            "drift_name": drift_name,
            "drift_yaml_path": str(Path(drift_yaml_path).resolve()),
            "adaptation_name": adaptation_name,
            "adaptation_cfg": adaptation_cfg,
            "model": {
                "class_path": model_class_path,
                "params": model_params,
            },
        }

        return RunLogger(
            experiment_dir=self.experiment_dir,
            run_id=run_id,
            run_dir=run_dir,
            drift_yaml_path=str(Path(drift_yaml_path).resolve()),
            run_used=run_used,
        )

    def append_summary(self, row: Dict[str, Any]) -> None:
        with self.summary_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._summary_fields)
            w.writerow(row)



    def generate_report(self) -> Path:
        import csv
        from html import escape

        def to_rel(p: str) -> str:
            try:
                pp = Path(p)
                base = self.experiment_dir.resolve()
                rel = pp.resolve().relative_to(base)
                return rel.as_posix()
            except Exception:
                return str(p).replace("\\", "/")

        def read_float(x: Any) -> Optional[float]:
            try:
                if x is None:
                    return None
                s = str(x).strip()
                if s == "":
                    return None
                return float(s)
            except Exception:
                return None

        def read_batch_series(batch_csv_path: Path) -> tuple[list[float], list[float]]:
            accs: list[float] = []
            f1s: list[float] = []
            precisions: list[float] = []
            recalls: list[float] = []
            if not batch_csv_path.exists():
                return accs, f1s, precisions, recalls
            with batch_csv_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    a = read_float(row.get("acc"))
                    b = read_float(row.get("f1_macro"))
                    p = read_float(row.get("precision"))
                    r = read_float(row.get("recall"))
                    if a is not None:
                        accs.append(a)
                    if b is not None:
                        f1s.append(b)
                    if p is not None:
                        precisions.append(p)
                    if r is not None:
                        recalls.append(r)
            return accs, f1s, precisions, recalls

        def sparkline_svg(accs: list[float], f1s: list[float], precisions: list[float], recalls: list[float], w: int = 400, h: int = 80) -> str:
            n = max(len(accs), len(f1s), len(precisions), len(recalls))
            if n <= 1:
                return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg>'

            def points(vals: list[float]) -> str:
                if len(vals) < 2:
                    return ""
                v = [min(1.0, max(0.0, float(x))) for x in vals]
                nloc = len(v)
                xs = [i * (w - 2) / (nloc - 1) + 1 for i in range(nloc)]
                ys = [(h - 2) - (v[i] * (h - 2)) + 1 for i in range(nloc)]
                return " ".join(f"{xs[i]:.2f},{ys[i]:.2f}" for i in range(nloc))

            acc_pts = points(accs)
            f1_pts = points(f1s)
            precision_pts = points(precisions)
            recall_pts = points(recalls)

            return (
                f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" fill="none" stroke="#ddd"/>'
                + (f'<polyline points="{acc_pts}" fill="none" stroke="#111" stroke-width="1.5"/>' if acc_pts else "")
                + (f'<polyline points="{f1_pts}" fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="3,2"/>' if f1_pts else "")
                + (f'<polyline points="{precision_pts}" fill="none" stroke="#0b57d0" stroke-width="1.5" stroke-dasharray="5,3"/>' if precision_pts else "")
                + (f'<polyline points="{recall_pts}" fill="none" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="3,5"/>' if recall_pts else "")
                + "</svg>"
            )

        summary_rows: list[dict[str, Any]] = []
        if self.summary_path.exists():
            with self.summary_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    summary_rows.append(row)

        title = f"Report — {self.experiment_name}"
        html_parts: list[str] = []
        html_parts.append("<!doctype html>")
        html_parts.append("<html><head><meta charset='utf-8'/>")
        html_parts.append(f"<title>{escape(title)}</title>")
        html_parts.append(
            """
    <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px;color:#111}
    h1{margin:0 0 8px 0;font-size:20px}
    .small{color:#666;font-size:12px;margin-bottom:16px}
    table{border-collapse:collapse;width:100%;font-size:12px}
    th,td{border:1px solid #e5e5e5;padding:6px 8px;vertical-align:top}
    th{background:#fafafa;text-align:left}
    code{background:#f6f6f6;padding:2px 4px;border-radius:4px}
    a{color:#0b57d0;text-decoration:none}
    a:hover{text-decoration:underline}
    .spark{white-space:nowrap}
    .right{text-align:right}
    </style>
    """
        )
        html_parts.append("</head><body>")
        html_parts.append(f"<h1>{escape(title)}</h1>")
        html_parts.append(f"<div class='small'>Experiment dir: <code>{escape(self.experiment_dir.as_posix())}</code></div>")

        html_parts.append("<table>")
        html_parts.append(
            "<tr>"
            "<th>run_id</th>"
            "<th>drift</th>"
            "<th>strategy</th>"
            "<th class='right'>n_batches</th>"
            "<th class='right'>mean_acc</th>"
            "<th class='right'>mean_f1</th>"
            "<th class='right'>mean_precision</th>"
            "<th class='right'>mean_recall</th>"
            "<th class='right'>final_acc</th>"
            "<th class='right'>final_f1</th>"
            "<th class='right'>wall_sec</th>"
            "<th class='right'>cpu_sec</th>"
            "<th class='right'>peak_rss_mb</th>"
            "<th>curves (acc=black, f1=gray, precision=blue, recall=red)</th>"
            "<th>files</th>"
            "</tr>"
        )

        for row in summary_rows:
            run_id = str(row.get("run_id", ""))
            drift_name = str(row.get("drift_name", ""))
            adapt_name = str(row.get("adaptation_name", ""))
            n_batches = row.get("n_batches", "")
            mean_acc = row.get("mean_acc", "")
            mean_f1 = row.get("mean_f1_macro", "")
            mean_precision = row.get("mean_precision", "")
            mean_recall = row.get("mean_recall", "")
            final_acc = row.get("final_acc", "")
            final_f1 = row.get("final_f1_macro", "")
            wall = row.get("total_wall_sec", "")
            cpu = row.get("total_cpu_sec", "")
            peak = row.get("peak_rss_mb", "")

            run_dir = (self.runs_dir / run_id)
            batch_csv = run_dir / "batch_metrics.csv"
            events_csv = run_dir / "events.csv"
            run_log = run_dir / "run.log"

            accs, f1s, precisions, recalls = read_batch_series(batch_csv)
            svg = sparkline_svg(accs, f1s, precisions, recalls)

            run_dir_rel = to_rel(str(run_dir))
            batch_rel = to_rel(str(batch_csv))
            events_rel = to_rel(str(events_csv))
            log_rel = to_rel(str(run_log))

            files_html = (
                f"<div><a href='{escape(run_dir_rel)}'>run dir</a></div>"
                f"<div><a href='{escape(batch_rel)}'>batch_metrics.csv</a></div>"
                f"<div><a href='{escape(events_rel)}'>events.csv</a></div>"
                f"<div><a href='{escape(log_rel)}'>run.log</a></div>"
            )

            html_parts.append(
                "<tr>"
                f"<td><code>{escape(run_id)}</code></td>"
                f"<td>{escape(drift_name)}</td>"
                f"<td>{escape(adapt_name)}</td>"
                f"<td class='right'>{escape(str(n_batches))}</td>"
                f"<td class='right'>{escape(str(mean_acc))}</td>"
                f"<td class='right'>{escape(str(mean_f1))}</td>"
                f"<td class='right'>{escape(str(mean_precision))}</td>"
                f"<td class='right'>{escape(str(mean_recall))}</td>"
                f"<td class='right'>{escape(str(final_acc))}</td>"
                f"<td class='right'>{escape(str(final_f1))}</td>"
                f"<td class='right'>{escape(str(wall))}</td>"
                f"<td class='right'>{escape(str(cpu))}</td>"
                f"<td class='right'>{escape(str(peak))}</td>"
                f"<td class='spark' style='width: 400px; height: 80px'>{svg}</td>"
                f"<td>{files_html}</td>"
                "</tr>"
            )

        html_parts.append("</table>")
        html_parts.append("<div class='small' style='margin-top:10px'>Tip: open CSVs to quickly validate logging.</div>")
        html_parts.append("</body></html>")

        report_path = self.experiment_dir / "report.html"
        report_path.write_text("\n".join(html_parts), encoding="utf-8")
        return report_path
