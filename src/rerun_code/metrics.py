from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .report_labeler import LABELS_13


def _install_radgraph_legacy_tokenizer_compatibility() -> bool:
    """Restore legacy BERT tokenizer helpers expected by RadGraph.

    Current Transformers releases retain ``_encode_plus`` but may no longer
    expose ``BertTokenizer.encode_plus`` or
    ``BertTokenizer.build_inputs_with_special_tokens``. RadGraph's bundled
    AllenNLP code calls both removed public methods while constructing its
    tokenizer. The shims only apply when a method is absent and reproduce the
    standard BERT sequence layout: ``[CLS] tokens [SEP]`` (and a second
    ``[SEP]`` for a paired sequence).
    """
    try:
        from transformers import BertTokenizer
    except ImportError:
        return False
    installed = False
    if not hasattr(BertTokenizer, "encode_plus") and not hasattr(BertTokenizer, "_encode_plus"):
        raise AttributeError(
            "BertTokenizer provides neither encode_plus nor _encode_plus; "
            "RadGraph is incompatible with this Transformers installation"
        )

    if not hasattr(BertTokenizer, "encode_plus"):
        def _legacy_encode_plus(self, text, text_pair=None, *args, **kwargs):
            return self._encode_plus(text, text_pair=text_pair, *args, **kwargs)

        BertTokenizer.encode_plus = _legacy_encode_plus
        installed = True

    if not hasattr(BertTokenizer, "build_inputs_with_special_tokens"):
        def _legacy_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            cls_id = getattr(self, "cls_token_id", None)
            sep_id = getattr(self, "sep_token_id", None)
            if cls_id is None or sep_id is None:
                raise AttributeError(
                    "BertTokenizer lacks cls_token_id or sep_token_id required "
                    "for RadGraph legacy compatibility"
                )
            first = list(token_ids_0)
            if token_ids_1 is None:
                return [cls_id, *first, sep_id]
            return [cls_id, *first, sep_id, *list(token_ids_1), sep_id]

        BertTokenizer.build_inputs_with_special_tokens = _legacy_build_inputs_with_special_tokens
        installed = True

    return installed


def labels_to_vector(labels: Iterable[str]) -> list[int]:
    present = {str(value).strip().lower().replace("_", " ") for value in (labels or [])}
    return [int(label in present) for label in LABELS_13]


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def aggregate_label_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise ValueError("No records")
    reference = np.asarray(frame["reference_vector"].tolist(), dtype=int)
    prediction = np.asarray(frame["prediction_vector"].tolist(), dtype=int)
    if reference.shape != prediction.shape or reference.shape[1] != len(LABELS_13):
        raise ValueError(f"Expected matching Nx{len(LABELS_13)} vectors")
    tp = ((reference == 1) & (prediction == 1)).sum(axis=0)
    fp = ((reference == 0) & (prediction == 1)).sum(axis=0)
    fn = ((reference == 1) & (prediction == 0)).sum(axis=0)
    tn = ((reference == 0) & (prediction == 0)).sum(axis=0)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.full(len(tp), np.nan), where=(2*tp+fp+fn)!=0)
    abnormal = reference.sum(axis=1) > 0
    fp_abnormal = ((reference[abnormal] == 0) & (prediction[abnormal] == 1)).sum()
    predicted_abnormal_events = (prediction[abnormal] == 1).sum()
    micro_tp, micro_fp, micro_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    reference_normal = ~abnormal
    predicted_normal = prediction.sum(axis=1) == 0
    result = {
        "fer": _divide(float(fp.sum()), float(prediction.sum())),
        "fer_abnormal": _divide(float(fp_abnormal), float(predicted_abnormal_events)),
        "omission": _divide(float(fn.sum()), float(reference.sum())),
        "macro_f1": float(np.nanmean(f1)),
        "micro_f1": _divide(2*micro_tp, 2*micro_tp+micro_fp+micro_fn),
        "hamming_accuracy": float((reference == prediction).mean()),
        "normal_abnormal_accuracy": float((reference_normal == predicted_normal).mean()),
        "n_records": float(len(frame)),
        "false_positive_events": float(fp.sum()),
        "predicted_positive_events": float(prediction.sum()),
        "false_negative_events": float(fn.sum()),
        "reference_positive_events": float(reference.sum()),
    }
    for index, label in enumerate(LABELS_13):
        result[f"f1_{label.replace(' ', '_')}"] = float(f1[index])
    for optional in ("radgraph_f1", "cider", "bertscore_f1", "rouge_l", "runtime_seconds"):
        if optional in frame and frame[optional].notna().any():
            result[optional] = float(frame[optional].mean())
    return result


def compute_text_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    metric_errors: dict[str, str] = {}
    hypotheses = output["final_report"].fillna("").astype(str).tolist()
    references = output["reference_report"].fillna("").astype(str).tolist()
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        output["rouge_l"] = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    except ImportError:
        output["rouge_l"] = np.nan
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(hypotheses, references, lang="en", verbose=False)
        output["bertscore_f1"] = f1.cpu().numpy()
    except ImportError:
        output["bertscore_f1"] = np.nan
    try:
        from radgraph import F1RadGraph
        _install_radgraph_legacy_tokenizer_compatibility()
        evaluator = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        _, rewards, _, _ = evaluator(hyps=hypotheses, refs=references)
        # With reward_level="all", F1RadGraph returns a 3-tuple of *lists*,
        # ordered (simple entity, partial entity/relation, complete
        # entity/relation), rather than one tuple per report. RG_ER (partial)
        # is the established F1-RadGraph outcome reported for RRG studies.
        if not isinstance(rewards, (list, tuple)) or len(rewards) != 3:
            raise ValueError(
                "Unexpected F1RadGraph reward structure; expected three "
                "per-report reward lists for simple, partial, and complete"
            )
        values = np.asarray(rewards[1], dtype=float).reshape(-1)
        if len(values) != len(output):
            raise ValueError(
                f"F1RadGraph partial-reward length {len(values)} does not "
                f"match the number of reports {len(output)}"
            )
        output["radgraph_f1"] = values
    except Exception as exc:
        output["radgraph_f1"] = np.nan
        metric_errors["radgraph_f1"] = f"{type(exc).__name__}: {exc}"
    # CIDEr requires a COCO-caption-compatible implementation. Missing is explicit, never zero-filled.
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [reference] for i, reference in enumerate(references)}
        res = {i: [hypothesis] for i, hypothesis in enumerate(hypotheses)}
        _, scores = Cider().compute_score(gts, res)
        output["cider"] = np.asarray(scores, dtype=float)
    except ImportError:
        output["cider"] = np.nan
    output.attrs["text_metric_errors"] = metric_errors
    return output


def majority_vote(vectors: Sequence[Sequence[int]]) -> list[int]:
    values = np.asarray(vectors, dtype=int)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[0] % 2 == 0:
        raise ValueError("Label majority vote requires an odd number of at least three models")
    return (values.sum(axis=0) > values.shape[0] / 2).astype(int).tolist()
