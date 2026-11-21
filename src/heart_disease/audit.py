"""Descriptive audit of the frozen source snapshot."""

from collections import Counter
from statistics import mean, quantiles, stdev

from .dataset import CATEGORIES, COLUMNS, NUMERIC, SOURCE, Record, binary_target, predictor_groups


def numeric_summary(values: list[float]) -> dict[str, float]:
    q1, median, q3 = quantiles(values, n=4, method="inclusive")
    return {
        "min": min(values), "q1": q1, "median": median, "q3": q3,
        "max": max(values), "mean": mean(values), "sample_std": stdev(values),
    }


def audit_records(records: list[Record], checksum: str) -> dict:
    columns = dict(zip(COLUMNS, zip(*records, strict=True), strict=True))
    groups = predictor_groups(records)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    raw_conflicts = [
        ids for ids in duplicate_groups if len({records[i - 1][-1] for i in ids}) > 1
    ]
    binary_conflicts = [
        ids for ids in raw_conflicts if len({binary_target(records[i - 1][-1]) for i in ids}) > 1
    ]
    return {
        "source": SOURCE,
        "sha256": checksum,
        "rows": len(records),
        "features": list(COLUMNS[:-1]),
        "target_definition": {"0": "absence", "1": "presence (original values 1–4)"},
        "missing": {name: values.count(None) for name, values in columns.items()},
        "invalid_entries": 0,  # The parser rejects invalid records before this audit.
        "exact_duplicate_excess_rows": len(records) - len(set(records)),
        "duplicate_predictor_groups": duplicate_groups,
        "conflicting_raw_target_groups": raw_conflicts,
        "conflicting_binary_target_groups": binary_conflicts,
        "original_target": dict(sorted(Counter(int(value) for value in columns["num"]).items())),
        "binary_target": dict(sorted(Counter(map(binary_target, columns["num"])).items())),
        "categorical_counts": {
            name: {code: columns[name].count(code) for code in codes}
            for name, codes in CATEGORIES.items()
        },
        "numeric_summary": {name: numeric_summary(columns[name]) for name in NUMERIC},
    }
