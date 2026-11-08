"""Fail-closed and transactional tests for Stage 2 downstream rehashing."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rehash_stage2_evidence import (
    ARTIFACTS,
    DERIVED_STAGE3_COMPARISON,
    EXPECTED_SOURCE_PINS,
    RANDOM_FOREST_MODEL,
    SOURCES,
    apply_plan,
    bootstrap_plan,
    plan,
)


OLD_A = "0" * 64
OLD_B = "1" * 64


class RehashStage2EvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
        artifacts = ("artifact-a.bin", "artifact-b.bin")
        sources = ("consumer-one.py", "consumer-two.py")
        (root / artifacts[0]).write_bytes(b"rebuilt-a")
        (root / artifacts[1]).write_bytes(b"rebuilt-b")
        (root / sources[0]).write_text(
            f'PINS = {{"artifact-a.bin": "{OLD_A}"}}\n', encoding="utf-8"
        )
        (root / sources[1]).write_text(
            "PINS = {\n"
            f'    "artifact-a.bin": ("{OLD_A}"),\n'
            f'    "artifact-b.bin": "{OLD_B}",\n'
            "}\n",
            encoding="utf-8",
        )
        expected = {
            sources[0]: (artifacts[0],),
            sources[1]: artifacts,
        }
        return artifacts, sources, expected

    def test_active_stage3_collector_is_an_explicit_consumer(self) -> None:
        source = "src/analysis/collect_stage3_results.py"
        self.assertIn(source, SOURCES)
        self.assertEqual(
            EXPECTED_SOURCE_PINS[source],
            ("results/tables/stage2_traditional_results.csv",),
        )

    def test_random_forest_model_is_an_explicit_precheck_pin(self) -> None:
        self.assertIn(RANDOM_FOREST_MODEL, ARTIFACTS)
        self.assertIn(
            RANDOM_FOREST_MODEL,
            EXPECTED_SOURCE_PINS["scripts/precheck_stage4.py"],
        )

    def test_bootstrap_is_explicit_and_requires_the_derived_table_to_be_absent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts, sources, expected = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "allowed only while.*absent"):
                bootstrap_plan(
                    root=root,
                    artifacts=artifacts,
                    sources=sources,
                    expected_source_pins=expected,
                    derived_artifact=artifacts[1],
                )

            (root / artifacts[1]).unlink()
            edits, bootstrap_token = bootstrap_plan(
                root=root,
                artifacts=artifacts,
                sources=sources,
                expected_source_pins=expected,
                derived_artifact=artifacts[1],
            )
            self.assertEqual(len(bootstrap_token), 64)
            apply_plan(edits)
            digest_a = hashlib.sha256((root / artifacts[0]).read_bytes()).hexdigest()
            self.assertIn(digest_a, (root / sources[0]).read_text(encoding="utf-8"))

            (root / artifacts[1]).write_bytes(b"derived-after-bootstrap")
            final_edits, final_token = plan(
                root=root,
                artifacts=artifacts,
                sources=sources,
                expected_source_pins=expected,
            )
            self.assertNotEqual(bootstrap_token, final_token)
            apply_plan(final_edits)
            digest_b = hashlib.sha256((root / artifacts[1]).read_bytes()).hexdigest()
            self.assertIn(digest_b, (root / sources[1]).read_text(encoding="utf-8"))

    def test_default_artifact_set_keeps_final_comparison_mandatory(self) -> None:
        self.assertIn(DERIVED_STAGE3_COMPARISON, ARTIFACTS)

    def test_bootstrap_still_requires_the_absent_artifacts_placeholder_pin(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts, sources, expected = self._fixture(root)
            (root / artifacts[1]).unlink()
            (root / sources[1]).write_text(
                f'PINS = {{"artifact-a.bin": "{OLD_A}"}}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "must contain exactly 1"):
                bootstrap_plan(
                    root=root,
                    artifacts=artifacts,
                    sources=sources,
                    expected_source_pins=expected,
                    derived_artifact=artifacts[1],
                )

    def test_plan_requires_every_artifact_and_exactly_one_expected_pin(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts, sources, expected = self._fixture(root)
            (root / artifacts[1]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing required rehash inputs"):
                plan(
                    root=root,
                    artifacts=artifacts,
                    sources=sources,
                    expected_source_pins=expected,
                )

            (root / artifacts[1]).write_bytes(b"rebuilt-b")
            with (root / sources[0]).open("a", encoding="utf-8") as stream:
                stream.write(f'EXTRA = {{"artifact-a.bin": "{OLD_A}"}}\n')
            with self.assertRaisesRegex(ValueError, "exactly 1 hash pin"):
                plan(
                    root=root,
                    artifacts=artifacts,
                    sources=sources,
                    expected_source_pins=expected,
                )

    def test_apply_rewrites_the_complete_pin_group(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts, sources, expected = self._fixture(root)
            edits, token = plan(
                root=root,
                artifacts=artifacts,
                sources=sources,
                expected_source_pins=expected,
            )
            self.assertEqual(len(token), 64)
            self.assertEqual(len(edits), 2)
            apply_plan(edits)
            digest_a = hashlib.sha256((root / artifacts[0]).read_bytes()).hexdigest()
            digest_b = hashlib.sha256((root / artifacts[1]).read_bytes()).hexdigest()
            self.assertIn(digest_a, (root / sources[0]).read_text(encoding="utf-8"))
            second = (root / sources[1]).read_text(encoding="utf-8")
            self.assertIn(digest_a, second)
            self.assertIn(digest_b, second)

    def test_apply_rolls_back_all_installed_sources_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts, sources, expected = self._fixture(root)
            edits, _token = plan(
                root=root,
                artifacts=artifacts,
                sources=sources,
                expected_source_pins=expected,
            )
            originals = {edit.path: edit.original for edit in edits}
            real_replace = os.replace
            failed = False

            def fail_second_once(source: str | Path, destination: str | Path) -> None:
                nonlocal failed
                if Path(destination) == edits[1].path and not failed:
                    failed = True
                    raise OSError("injected second-install failure")
                real_replace(source, destination)

            with patch(
                "scripts.rehash_stage2_evidence.os.replace",
                side_effect=fail_second_once,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    apply_plan(edits)
            self.assertTrue(failed)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
