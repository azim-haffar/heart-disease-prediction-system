import json
from pathlib import Path
from unittest import TestCase

from heart_disease.audit import audit_records
from heart_disease.dataset import binary_target, load_snapshot, parse_records
from heart_disease.split import split_rows

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "processed.cleveland.data"
EXAMPLE = b"63,1,1,145,233,1,2,150,0,2.3,3,0,6,0"


class DatasetTests(TestCase):
    def test_target_mapping(self):
        self.assertEqual([binary_target(i) for i in range(5)], [0, 1, 1, 1, 1])
        for invalid in (None, -1, 5, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                binary_target(invalid)

    def test_invalid_fields(self):
        for row in (
            b"1,2", EXAMPLE.replace(b",6,0", b",4,0"),
            EXAMPLE.replace(b"145", b"NaN"), EXAMPLE.replace(b"145", b"?"),
            EXAMPLE.replace(b"145", b"145.5"), EXAMPLE[:-1] + b"5",
        ):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Row 1"):
                parse_records(row)

    def test_audit_counts_and_nullable_fields(self):
        records, checksum = load_snapshot(SNAPSHOT)
        result = audit_records(records, checksum)
        self.assertEqual(result["rows"], 303)
        self.assertEqual({k: v for k, v in result["missing"].items() if v}, {"ca": 4, "thal": 2})
        self.assertEqual(result["binary_target"], {0: 164, 1: 139})
        self.assertEqual(result["invalid_entries"], 0)
        self.assertEqual(result["exact_duplicate_excess_rows"], 0)
        self.assertEqual(result["duplicate_predictor_groups"], [])

    def test_split_is_disjoint_reproducible_and_stratified(self):
        records, _ = load_snapshot(SNAPSHOT)
        train, test = split_rows(records)
        self.assertEqual((train, test), split_rows(records))
        self.assertEqual((len(train), len(test)), (242, 61))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(1, 304)))
        self.assertEqual(sum(binary_target(records[i - 1][-1]) for i in test), 28)

    def test_split_manifest_matches_frozen_snapshot(self):
        records, checksum = load_snapshot(SNAPSHOT)
        manifest = json.loads(SNAPSHOT.with_name("split.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"], checksum)
        self.assertEqual(split_rows(records), (manifest["train_row_ids"], manifest["test_row_ids"]))

    def test_same_binary_label_duplicates_stay_together(self):
        records, _ = load_snapshot(SNAPSHOT)
        first_positive = next(record for record in records if binary_target(record[-1]))
        positive_id = records.index(first_positive) + 1
        records.extend([first_positive, (*first_positive[:-1], 4 if first_positive[-1] != 4 else 1)])
        report = audit_records(records, "")
        self.assertEqual(report["exact_duplicate_excess_rows"], 1)
        self.assertEqual(report["conflicting_raw_target_groups"], [[positive_id, 304, 305]])
        self.assertEqual(report["conflicting_binary_target_groups"], [])
        train, test = split_rows(records)
        grouped = {positive_id, 304, 305}
        self.assertTrue(grouped <= set(train) or grouped <= set(test))

    def test_conflicting_binary_labels_fail_the_split(self):
        records, _ = load_snapshot(SNAPSHOT)
        records.append((*records[0][:-1], 1 if records[0][-1] == 0 else 0))
        self.assertEqual(audit_records(records, "")["conflicting_binary_target_groups"], [[1, 304]])
        with self.assertRaisesRegex(ValueError, "Conflicting binary labels"):
            split_rows(records)
