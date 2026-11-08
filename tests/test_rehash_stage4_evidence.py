"""Focused fail-closed tests for transactional Stage 4 evidence rehashing."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import rehash_stage4_evidence
from scripts.rehash_stage2_evidence import apply_plan
from scripts.rehash_stage4_evidence import (
    ARTIFACTS,
    EXPECTED_SOURCE_PINS,
    SOURCES,
    plan_stage4,
)


OLD_HASH = "0" * 64


class RehashStage4EvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        for index, relative in enumerate(ARTIFACTS):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"rebuilt-stage4-{index}".encode("utf-8"))
        source = root / SOURCES[0]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "FROZEN_ARTIFACTS = (\n"
            + "".join(
                f'    FrozenArtifact("stage4", "item", "{relative}", "{OLD_HASH}"),\n'
                for relative in ARTIFACTS
            )
            + ")\n",
            encoding="utf-8",
        )

    def test_allowlist_is_exactly_the_six_stage4_contract_artifacts(self) -> None:
        self.assertEqual(SOURCES, ("src/analysis/stage5_contract.py",))
        self.assertEqual(EXPECTED_SOURCE_PINS[SOURCES[0]], ARTIFACTS)
        self.assertEqual(len(ARTIFACTS), 6)
        self.assertEqual(len(set(ARTIFACTS)), 6)

    def test_plan_and_apply_rewrite_the_complete_pin_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            edits, token = plan_stage4(root=root)
            self.assertEqual(len(token), 64)
            self.assertEqual(len(edits), 1)
            self.assertEqual(len(edits[0].changes), len(ARTIFACTS))
            apply_plan(edits)
            source = (root / SOURCES[0]).read_text(encoding="utf-8")
            for relative in ARTIFACTS:
                digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
                self.assertIn(f'"{relative}", "{digest}"', source)

    def test_missing_artifact_and_unexpected_pin_count_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            (root / ARTIFACTS[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing required rehash inputs"):
                plan_stage4(root=root)

            (root / ARTIFACTS[-1]).write_bytes(b"restored")
            source = root / SOURCES[0]
            with source.open("a", encoding="utf-8") as stream:
                stream.write(f'EXTRA = {{"{ARTIFACTS[0]}": "{OLD_HASH}"}}\n')
            with self.assertRaisesRegex(ValueError, "exactly 1 hash pin"):
                plan_stage4(root=root)

    def test_cli_requires_the_current_confirmation_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            source = root / SOURCES[0]
            original = source.read_bytes()
            errors = io.StringIO()
            with patch.object(rehash_stage4_evidence, "PROJECT_ROOT", root), redirect_stdout(
                io.StringIO()
            ), redirect_stderr(errors):
                self.assertEqual(
                    rehash_stage4_evidence.main(
                        ["--apply", "--confirm", "f" * 64]
                    ),
                    1,
                )
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("requires --confirm", errors.getvalue())

            _edits, token = plan_stage4(root=root)
            with patch.object(rehash_stage4_evidence, "PROJECT_ROOT", root), redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(
                    rehash_stage4_evidence.main(["--apply", "--confirm", token]),
                    0,
                )
            self.assertNotEqual(source.read_bytes(), original)

    def test_repository_stage4_pins_are_already_current(self) -> None:
        edits, token = plan_stage4()
        self.assertEqual(edits, [])
        self.assertEqual(len(token), 64)


if __name__ == "__main__":
    unittest.main()
