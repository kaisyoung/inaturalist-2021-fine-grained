"""Fail-closed tests for the immutable Stage 3 / rewritten Stage 2 bridge."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.stage5_contract import (
    FROZEN_BY_PATH,
    LEGACY_STAGE2_CONTINUITY_BINDINGS,
    NESTED_BINDING_MANIFESTS,
    PROJECT_ROOT,
    STAGE3_CONTINUITY_RELATIVE_PATH,
    STAGE3_FREEZE_RELATIVE_PATH,
    STAGE3_FREEZE_SHA256,
    STAGE3_TEST_ACCESS_RELATIVE_PATH,
    STAGE3_TEST_ACCESS_SHA256,
    _expected_stage3_continuity_document,
    _is_exact_allowlisted_stage3_legacy_binding,
    _validate_stage3_continuity_document,
    _verify_stage3_continuity_overlay,
    load_json,
    verify_nested_artifact_bindings,
)


class Stage3TraditionalRewriteContinuityTests(unittest.TestCase):
    def test_canonical_overlay_reconciles_all_five_legacy_bindings(self) -> None:
        self.assertIn(STAGE3_CONTINUITY_RELATIVE_PATH, NESTED_BINDING_MANIFESTS)
        self.assertIn(STAGE3_CONTINUITY_RELATIVE_PATH, FROZEN_BY_PATH)
        self.assertEqual(len(LEGACY_STAGE2_CONTINUITY_BINDINGS), 5)

        rows = verify_nested_artifact_bindings(
            (STAGE3_FREEZE_RELATIVE_PATH, STAGE3_CONTINUITY_RELATIVE_PATH)
        )
        by_path = {row["path"]: row for row in rows}
        self.assertEqual(by_path[STAGE3_FREEZE_RELATIVE_PATH]["sha256"], STAGE3_FREEZE_SHA256)
        self.assertEqual(
            by_path[STAGE3_TEST_ACCESS_RELATIVE_PATH]["sha256"],
            STAGE3_TEST_ACCESS_SHA256,
        )
        for binding in LEGACY_STAGE2_CONTINUITY_BINDINGS:
            row = by_path[binding.artifact_path]
            self.assertEqual(row["sha256"], binding.current_sha256)
            self.assertTrue(
                any(
                    origin.startswith(f"{STAGE3_CONTINUITY_RELATIVE_PATH}:")
                    for origin in row["origins"]
                )
            )
            self.assertFalse(
                any(
                    origin.startswith(f"{STAGE3_FREEZE_RELATIVE_PATH}:")
                    for origin in row["origins"]
                )
            )

    def test_stage3_freeze_without_overlay_is_rejected_before_exemption(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires.*continuity overlay"):
            verify_nested_artifact_bindings((STAGE3_FREEZE_RELATIVE_PATH,))

    def test_semantic_path_hash_and_assertion_tampering_are_rejected(self) -> None:
        canonical = load_json(PROJECT_ROOT / STAGE3_CONTINUITY_RELATIVE_PATH)
        mutations = []

        wrong_path = copy.deepcopy(canonical)
        wrong_path["legacy_stage2_bindings"][0]["current_replacement"]["path"] = (
            "results/tables/unrelated.csv"
        )
        mutations.append(wrong_path)

        wrong_hash = copy.deepcopy(canonical)
        wrong_hash["legacy_stage2_bindings"][2]["legacy"][
            "sha256_at_stage3_freeze"
        ] = "0" * 64
        mutations.append(wrong_hash)

        false_assertion = copy.deepcopy(canonical)
        false_assertion["assertions"][
            "stage3_checkpoints_and_model_evidence_unchanged"
        ] = False
        mutations.append(false_assertion)

        for document in mutations:
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "continuity overlay content mismatch"):
                    _validate_stage3_continuity_document(document)

    def test_semantically_equal_but_byte_tampered_overlay_fails_hash_pin(self) -> None:
        document = _expected_stage3_continuity_document()
        with tempfile.TemporaryDirectory(prefix="stage3-continuity-") as name:
            path = Path(name) / "overlay.json"
            path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            self.assertNotEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                FROZEN_BY_PATH[STAGE3_CONTINUITY_RELATIVE_PATH].sha256,
            )
            with self.assertRaisesRegex(ValueError, "continuity overlay hash mismatch"):
                _verify_stage3_continuity_overlay(path)

    def test_allowlist_requires_exact_origin_location_path_hash_and_size(self) -> None:
        item = LEGACY_STAGE2_CONTINUITY_BINDINGS[2]
        exact = {
            "origin": STAGE3_FREEZE_RELATIVE_PATH,
            "location": item.location,
            "path": item.artifact_path,
            "sha256": item.legacy_sha256,
            "bytes": item.legacy_bytes,
        }
        self.assertTrue(_is_exact_allowlisted_stage3_legacy_binding(exact))

        wrong_origin = dict(exact, origin="results/metrics/copied_stage3_freeze.json")
        self.assertFalse(_is_exact_allowlisted_stage3_legacy_binding(wrong_origin))
        wrong_location = dict(exact, location="$.stage2_artifacts.unlisted")
        self.assertFalse(_is_exact_allowlisted_stage3_legacy_binding(wrong_location))
        for field, value in (
            ("path", "results/metrics/other.json"),
            ("sha256", "0" * 64),
            ("bytes", item.legacy_bytes + 1),
        ):
            with self.subTest(field=field):
                altered = dict(exact, **{field: value})
                with self.assertRaisesRegex(ValueError, "exact continuity allowlist"):
                    _is_exact_allowlisted_stage3_legacy_binding(altered)

    def test_exact_legacy_values_in_non_allowlisted_manifest_still_conflict(self) -> None:
        item = LEGACY_STAGE2_CONTINUITY_BINDINGS[2]
        with tempfile.TemporaryDirectory(
            prefix=".stage3-continuity-conflict-", dir=PROJECT_ROOT / "results"
        ) as name:
            root = Path(name)
            artifact = root / "replacement.json"
            artifact.write_text("{}\n", encoding="utf-8")
            relative = artifact.relative_to(PROJECT_ROOT).as_posix()
            current_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            legacy_manifest = root / "legacy-copy.json"
            legacy_manifest.write_text(
                json.dumps(
                    {
                        "binding": {
                            "path": relative,
                            "sha256": item.legacy_sha256,
                            "bytes": artifact.stat().st_size,
                        }
                    }
                ),
                encoding="utf-8",
            )
            current_manifest = root / "current.json"
            current_manifest.write_text(
                json.dumps(
                    {
                        "binding": {
                            "path": relative,
                            "sha256": current_sha,
                            "bytes": artifact.stat().st_size,
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifests = tuple(
                path.relative_to(PROJECT_ROOT).as_posix()
                for path in (legacy_manifest, current_manifest)
            )
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate hashes"):
                verify_nested_artifact_bindings(manifests)


if __name__ == "__main__":
    unittest.main()
