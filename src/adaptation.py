# adaptation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from skmultiflow.drift_detection import ADWIN, DDM, PageHinkley


class TrainBuffer:
    def __init__(
        self,
        mode: str = "all_seen",
        window_size: Optional[int] = None,
        window_batches: Optional[int] = None,
    ):
        self.mode = str(mode)

        if self.mode not in ("all_seen", "last_window"):
            raise ValueError(f"Invalid buffer mode: {self.mode}. Use 'all_seen' or 'last_window'.")

        self.window_size = window_size
        self.window_batches = window_batches

        if self.mode == "last_window":
            if (window_size is None and window_batches is None) or (window_size is not None and window_batches is not None):
                raise ValueError(
                    "For buffer_mode='last_window' you must specify exactly one: "
                    "'window_size' (examples) OR 'window_batches' (batches)."
                )

            if window_size is not None:
                if not isinstance(window_size, int) or window_size <= 0:
                    raise ValueError(f"window_size must be positive int, got {window_size}.")
                self._window_type = "examples"
            else:
                if not isinstance(window_batches, int) or window_batches <= 0:
                    raise ValueError(f"window_batches must be positive int, got {window_batches}.")
                self._window_type = "batches"
        else:
            self._window_type = "all"

        self._texts: List[str] = []
        self._labels: List[int] = []

        self._batch_texts = deque()   # deque[List[str]]
        self._batch_labels = deque()  # deque[List[int]]

    def add(self, texts: List[str], labels: List[int]) -> None:
        """Adds ONE batch worth of data (texts, labels) to the buffer."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        if self.mode == "all_seen":
            self._texts.extend(texts)
            self._labels.extend(labels)
            return

        if self._window_type == "examples":
            self._texts.extend(texts)
            self._labels.extend(labels)

            ws = int(self.window_size)  # type: ignore[arg-type]
            if len(self._texts) > ws:
                self._texts = self._texts[-ws:]
                self._labels = self._labels[-ws:]
            return

        self._batch_texts.append(list(texts))
        self._batch_labels.append(list(labels))

        k = int(self.window_batches)  # type: ignore[arg-type]
        while len(self._batch_texts) > k:
            self._batch_texts.popleft()
            self._batch_labels.popleft()

    def get_data(self) -> Tuple[List[str], List[int]]:
        if self.mode == "all_seen":
            return list(self._texts), list(self._labels)

        if self._window_type == "examples":
            return list(self._texts), list(self._labels)

        out_texts: List[str] = []
        out_labels: List[int] = []
        for bt, bl in zip(self._batch_texts, self._batch_labels):
            out_texts.extend(bt)
            out_labels.extend(bl)
        return out_texts, out_labels

    def size(self) -> int:
        if self.mode == "all_seen":
            return len(self._texts)
        if self._window_type == "examples":
            return len(self._texts)
        # batches
        return sum(len(bt) for bt in self._batch_texts)

    def clear(self) -> None:
        self._texts.clear()
        self._labels.clear()
        self._batch_texts.clear()
        self._batch_labels.clear()


@dataclass
class AdaptationDecision:
    should_retrain: bool
    reason: str = ""
    info: Dict[str, Any] = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class AdaptationStrategy:

    def __init__(self, buffer: Optional[TrainBuffer] = None):
        self.buffer = buffer

    def on_pretrain_end(self, texts: List[str], labels: List[int]) -> None:
        if self.buffer is not None:
            self.buffer.add(texts, labels)

    def observe_batch(self, batch: dict, batch_metrics: Dict[str, Any]) -> AdaptationDecision:
        return AdaptationDecision(False, reason="base")

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        if self.buffer is None:
            return [], []
        return self.buffer.get_data()


class NoAdaptationStrategy(AdaptationStrategy):
    def __init__(self):
        super().__init__(buffer=None)

    def on_pretrain_end(self, texts: List[str], labels: List[int]) -> None:
        return

    def observe_batch(self, batch: dict, batch_metrics: Dict[str, Any]) -> AdaptationDecision:
        return AdaptationDecision(False, reason="none")



class PeriodicRetrainStrategy(AdaptationStrategy):
    def __init__(self, period_batches: int, buffer: TrainBuffer):
        super().__init__(buffer=buffer)

        if not isinstance(period_batches, int) or period_batches <= 0:
            raise ValueError(f"period_batches must be positive int, got {period_batches}.")
        self.period_batches = int(period_batches)
        self._stream_batches_seen = 0

    def observe_batch(self, batch: dict, batch_metrics: Dict[str, Any]) -> AdaptationDecision:

        self.buffer.add(batch["texts"], batch["labels"])  

        self._stream_batches_seen += 1

        if self._stream_batches_seen % self.period_batches == 0:
            return AdaptationDecision(
                True,
                reason="periodic",
                info={
                    "period_batches": self.period_batches,
                    "stream_batches_seen": self._stream_batches_seen,
                    "buffer_size": self.buffer.size(),  
                    "global_batch_id": int(batch.get("batch_id", -1)),
                },
            )

        return AdaptationDecision(False, reason="periodic_wait")




from typing import Any, Dict, List, Tuple, Optional

class TriggerRetrainByMetric(AdaptationStrategy):
    def __init__(self, buffer: TrainBuffer, params: Dict[str, Any]):
        super().__init__(buffer=buffer)
        self.threshold = float(params.get("threshold", 0.05))
        self.quality_metric = str(params.get("quality_metric", "f1"))

        self.baseline_k = int(params.get("baseline_k", 3))
        self.confirm_k = int(params.get("confirm_k", 2))
        if self.baseline_k < 1:
            self.baseline_k = 1
        if self.confirm_k < 1:
            self.confirm_k = 1

        self.baseline_metric: Optional[float] = None
        self._baseline_sum: float = 0.0
        self._baseline_n: int = 0
        self._bad_streak: int = 0

    def on_retrain(self) -> None:
        self.baseline_metric = None
        self._baseline_sum = 0.0
        self._baseline_n = 0
        self._bad_streak = 0

    def observe_batch(self, batch: dict, batch_metrics: Dict[str, Any]) -> AdaptationDecision:
        self.buffer.add(batch["texts"], batch["labels"])

        current_metric = float(batch_metrics.get(self.quality_metric, 0.0))
        batch_metrics[f"prev_{self.quality_metric}"] = current_metric  # как было

        if self.baseline_metric is None:
            self._baseline_sum += current_metric
            self._baseline_n += 1
            if self._baseline_n >= self.baseline_k:
                self.baseline_metric = self._baseline_sum / self._baseline_n
                self._bad_streak = 0
                return AdaptationDecision(False, reason="baseline_set")
            return AdaptationDecision(False, reason="building_baseline")

        drop = self.baseline_metric - current_metric
        if drop >= self.threshold:
            self._bad_streak += 1
        else:
            self._bad_streak = 0

        if self._bad_streak >= self.confirm_k:
            return AdaptationDecision(
                True,
                reason="quality_drop",
                info={"threshold": self.threshold, "drop": round(drop, 3), "current_metric": current_metric, "baseline_metric": round(self.baseline_metric, 3)},
            )

        return AdaptationDecision(False, reason="trigger_wait")

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        return self.buffer.get_data()




class TriggerRetrainByDrift(AdaptationStrategy):
    def __init__(self, buffer: TrainBuffer, params: Dict[str, Any]):
        super().__init__(buffer=buffer)
        self.drift_detector = None


        drift_type = str(params.get("drift_detector", "ADWIN"))
        delta = float(params.get("delta", 0.05))  
        print("delta ", delta)
        print("metric ")       

        if drift_type == "ADWIN":
            self.drift_detector = ADWIN(delta=delta)
        elif drift_type == "DDM":
            self.drift_detector = DDM()
        elif drift_type == "PHT":
            self.drift_detector = PageHinkley()
        else:
            raise ValueError(f"Unknown drift detector type: {drift_type}")
        
        print(f"[DEBUG] Drift detector initialized: {self.drift_detector}")

        self.metric_to_track = str(params.get("metric_to_track", "acc"))

    def observe_batch(self, batch: dict, batch_metrics: Dict[str, Any]) -> AdaptationDecision:
        self.buffer.add(batch["texts"], batch["labels"])


        y_true = batch["labels"]
        y_pred = batch["predictions"] 

        example_metrics = [1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_true))]
        print(example_metrics[:5])
        for metric in example_metrics:
            self.drift_detector.add_element(metric)

        print(f"[DEBUG] Drift detector status: {self.drift_detector.get_info()}")

        if self.drift_detector.detected_change():
            print("[DEBUG] Drift detected!")
            return AdaptationDecision(True, reason="drift_detected", info={"drift_detector": str(self.drift_detector)})

        return AdaptationDecision(False, reason="trigger_wait")

    def get_train_data(self) -> Tuple[List[str], List[int]]:
        return self.buffer.get_data()




def build_strategy(adaptation_cfg: Dict[str, Any]) -> AdaptationStrategy:
    atype = str((adaptation_cfg or {}).get("type", "none"))
    params = (adaptation_cfg or {}).get("params", {}) or {}

    if atype == "none":
        return NoAdaptationStrategy()

    if atype == "retrain":
        period = int(params.get("period_batches", 5))
        buffer_mode = str(params.get("buffer_mode", "all_seen"))
        window_size = params.get("window_size", None)
        window_batches = params.get("window_batches", None)

        buf = TrainBuffer(mode=buffer_mode, window_size=window_size, window_batches=window_batches)
        return PeriodicRetrainStrategy(period_batches=period, buffer=buf)

    if atype == "quality_drop":
        buffer_mode = str(params.get("buffer_mode", "all_seen"))
        window_size = params.get("window_size", None)
        window_batches = params.get("window_batches", None)
        if "quality_metric" not in params:
            raise ValueError("Missing 'quality_metric' parameter for 'quality_drop' strategy")

        buf = TrainBuffer(mode=buffer_mode, window_size=window_size, window_batches=window_batches)
        return TriggerRetrainByMetric(buffer=buf, params=params) 

    if atype == "drift_detected":

        buffer_mode = str(params.get("buffer_mode", "all_seen"))
        window_size = params.get("window_size", None)
        window_batches = params.get("window_batches", None)

        if "drift_detector" not in params:
            raise ValueError("Missing 'drift_detector' parameter for 'drift_detected' strategy")

        buf = TrainBuffer(mode=buffer_mode, window_size=window_size, window_batches=window_batches)
        return TriggerRetrainByDrift(buffer=buf, params=params) 

    raise ValueError(f"Unknown adaptation type: {atype}")

def mode_strategy(adaptation_cfg: Dict[str, Any]) -> AdaptationStrategy:
    atype = str((adaptation_cfg or {}).get("type", "none"))
    params = (adaptation_cfg or {}).get("params", {}) or {}
    buffer_mode = str(params.get("mode", "fit"))
    return buffer_mode
