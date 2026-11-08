"""Focused acceptance tests for Stage 5 report building and evidence inventory."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_report as report_builder
from scripts import run_final_reproducibility_audit as reproducibility_audit
from scripts.build_final_evidence_inventory import (
    DEFAULT_OUTPUT,
    HISTORICAL_REPORTS,
    _artifact_type,
    _discover,
    _stage,
    build_inventory,
)
from src.analysis.stage5_contract import FROZEN_ARTIFACTS, PROJECT_ROOT, READ_ONLY_REPORTS


class ReportBuildSafetyTests(unittest.TestCase):
    def _args(self, root: Path) -> argparse.Namespace:
        tex_dir = root / "cvpr"
        tex_dir.mkdir()
        (tex_dir / "main.tex").write_text("% synthetic\n", encoding="utf-8")
        return argparse.Namespace(
            tex_dir=tex_dir,
            output=root / "canonical.pdf",
            report=root / "LATEX_BUILD_REPORT.md",
            force=True,
        )

    @staticmethod
    def _successful_compile(tex_dir: Path, build_dir: Path):
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "main.pdf").write_bytes(b"%PDF-1.7\nsynthetic\n%%EOF\n")
        (build_dir / "main.aux").write_text(
            r"\newlabel{LastPageMainText}{{}{11}}" + "\n", encoding="utf-8"
        )
        (build_dir / "main.log").write_text("clean log\n", encoding="utf-8")
        return [
            report_builder.CommandResult(
                ("synthetic-latexmk",), tex_dir, 0, "synthetic success"
            )
        ]

    def test_failed_page_limit_cannot_replace_existing_canonical_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            args.output.write_bytes(b"known-good sentinel")
            with patch.object(report_builder, "_compile", self._successful_compile), patch.object(
                report_builder, "_pdf_pages", return_value=(12, "Pages: 12\n")
            ):
                self.assertEqual(report_builder.build_report(args), 1)
            self.assertEqual(args.output.read_bytes(), b"known-good sentinel")
            self.assertIn("**Status: FAIL**", args.report.read_text(encoding="utf-8"))

    def test_overfull_layout_warning_blocks_pdf_installation(self) -> None:
        def compile_with_overflow(tex_dir: Path, build_dir: Path):
            results = self._successful_compile(tex_dir, build_dir)
            (build_dir / "main.aux").write_text(
                r"\newlabel{LastPageMainText}{{}{1}}" + "\n", encoding="utf-8"
            )
            (build_dir / "main.log").write_text(
                r"Overfull \hbox (2.0pt too wide)" + "\n", encoding="utf-8"
            )
            return results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            with patch.object(report_builder, "_compile", compile_with_overflow):
                self.assertEqual(report_builder.build_report(args), 1)
            self.assertFalse(args.output.exists())
            audit = args.report.read_text(encoding="utf-8")
            self.assertIn("Overfull hbox warnings", audit)
            self.assertIn("**Status: FAIL**", audit)

    def test_run_prepends_selected_executable_directory_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "managed-texlive" / "bin" / "latexmk"
            command = [str(executable), "-pdf", "main.tex"]
            completed = report_builder.subprocess.CompletedProcess(
                command, 0, stdout="synthetic success"
            )
            with patch.dict(
                report_builder.os.environ,
                {"PATH": "/system/bin", "LATEX_BIN_DIR": ""},
                clear=False,
            ), patch.object(
                report_builder.subprocess, "run", return_value=completed
            ) as run:
                result = report_builder._run(command, root)

            self.assertEqual(result.returncode, 0)
            environment = run.call_args.kwargs["env"]
            self.assertEqual(
                environment["PATH"],
                f"{executable.parent}{report_builder.os.pathsep}/system/bin",
            )
            self.assertEqual(environment["LC_ALL"], "C")

    def test_find_command_preserves_texlive_launcher_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "texmf-dist" / "scripts" / "latexmk.pl"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env perl\n", encoding="utf-8")
            launcher = root / "bin" / "universal-darwin" / "latexmk"
            launcher.parent.mkdir(parents=True)
            launcher.symlink_to(script)

            with patch.object(
                report_builder.shutil, "which", return_value=str(launcher)
            ):
                selected = report_builder._find_command("latexmk")

            self.assertEqual(selected, launcher.absolute())
            self.assertEqual(selected.parent, launcher.parent)

    def test_path_discovery_prefers_latexmk_and_requires_fallback_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tex_dir = root / "cvpr"
            build_dir = tex_dir / "build"
            tex_dir.mkdir()
            portable_bin = root / "portable-bin"
            locations = {
                command: str(portable_bin / command)
                for command in ("pdflatex", "bibtex", "latexmk")
            }

            def locate(command: str):
                return locations[command]

            def successful_run(command: list[str], cwd: Path):
                return report_builder.CommandResult(tuple(command), cwd, 0, "ok")

            with patch.dict(
                report_builder.os.environ,
                {"PATH": "/portable/path", "LATEX_BIN_DIR": ""},
                clear=False,
            ), patch.object(
                report_builder.shutil, "which", side_effect=locate
            ) as which, patch.object(
                report_builder, "_run", side_effect=successful_run
            ) as run:
                results = report_builder._compile(tex_dir, build_dir)

            self.assertEqual(
                [item.args for item in which.call_args_list],
                [("pdflatex",), ("bibtex",), ("latexmk",)],
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(run.call_count, 1)
            command, cwd = run.call_args.args
            self.assertEqual(command[0], str(portable_bin / "latexmk"))
            self.assertEqual(cwd, tex_dir)

    def test_latex_bin_dir_takes_precedence_and_supports_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tex_dir = root / "extracted project" / "cvpr"
            build_dir = tex_dir / "build"
            tex_dir.mkdir(parents=True)
            latex_bin = root / "TeX tools with spaces" / "bin"
            latex_bin.mkdir(parents=True)
            expected_path = (
                f"{latex_bin.absolute()}{report_builder.os.pathsep}/system/bin"
            )

            def locate(command: str, *, path: str):
                self.assertEqual(path, expected_path)
                return str(latex_bin / command)

            def successful_run(command: list[str], cwd: Path):
                return report_builder.CommandResult(tuple(command), cwd, 0, "ok")

            with patch.dict(
                report_builder.os.environ,
                {"PATH": "/system/bin", "LATEX_BIN_DIR": str(latex_bin)},
                clear=False,
            ), patch.object(
                report_builder.shutil, "which", side_effect=locate
            ) as which, patch.object(
                report_builder, "_run", side_effect=successful_run
            ):
                results = report_builder._compile(tex_dir, build_dir)

            self.assertEqual(which.call_count, 3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].command[0], str(latex_bin / "latexmk"))

    def test_latex_bin_dir_is_propagated_to_subprocess_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latex_bin = root / "TeX tools with spaces"
            latex_bin.mkdir()
            executable = root / "launcher-bin" / "latexmk"
            command = [str(executable), "-pdf", "main.tex"]
            completed = report_builder.subprocess.CompletedProcess(
                command, 0, stdout="synthetic success"
            )
            with patch.dict(
                report_builder.os.environ,
                {"PATH": "/system/bin", "LATEX_BIN_DIR": str(latex_bin)},
                clear=False,
            ), patch.object(
                report_builder.subprocess, "run", return_value=completed
            ) as run:
                report_builder._run(command, root)

            self.assertEqual(
                run.call_args.kwargs["env"]["PATH"],
                report_builder.os.pathsep.join(
                    (str(latex_bin), str(executable.parent), "/system/bin")
                ),
            )

    def test_invalid_latex_bin_dir_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing TeX bin"
            with patch.dict(
                report_builder.os.environ,
                {"LATEX_BIN_DIR": str(missing)},
                clear=False,
            ), patch.object(report_builder.shutil, "which") as which:
                with self.assertRaises(NotADirectoryError) as raised:
                    report_builder._compile(root, root / "build")

            which.assert_not_called()
            message = str(raised.exception)
            self.assertIn("LATEX_BIN_DIR is not a directory", message)
            self.assertIn("unset it to use PATH", message)

    def test_missing_required_tex_tool_has_install_guidance(self) -> None:
        locations = {
            "pdflatex": None,
            "bibtex": "/portable/bin/bibtex",
            "latexmk": "/portable/bin/latexmk",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            report_builder.os.environ, {"LATEX_BIN_DIR": ""}, clear=False
        ), patch.object(
            report_builder.shutil,
            "which",
            side_effect=lambda command: locations[command],
        ):
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as raised:
                report_builder._compile(root, root / "build")

        message = str(raised.exception)
        self.assertIn("Required TeX command(s) not found: pdflatex", message)
        self.assertIn("MacTeX", message)
        self.assertIn("TeX Live", message)
        self.assertIn("LATEX_BIN_DIR", message)

    def test_pdflatex_fallback_runs_bibtex_from_tex_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tex_dir = Path(directory) / "cvpr"
            build_dir = tex_dir / "build"
            tex_dir.mkdir()
            locations = {
                "pdflatex": Path("/portable/bin/pdflatex"),
                "bibtex": Path("/portable/bin/bibtex"),
                "latexmk": None,
            }

            def find(command: str, *, search_path: str | None = None):
                return locations[command]

            def successful_run(command: list[str], cwd: Path):
                return report_builder.CommandResult(tuple(command), cwd, 0, "ok")

            with patch.dict(
                report_builder.os.environ, {"LATEX_BIN_DIR": ""}, clear=False
            ), patch.object(
                report_builder, "_find_command", side_effect=find
            ), patch.object(
                report_builder, "_run", side_effect=successful_run
            ) as run:
                results = report_builder._compile(tex_dir, build_dir)

            self.assertEqual(len(results), 4)
            invocations = [item.args for item in run.call_args_list]
            self.assertEqual(invocations[0][1], tex_dir)
            self.assertEqual(
                invocations[1], (["/portable/bin/bibtex", "build/main"], tex_dir)
            )
            self.assertEqual(invocations[2], invocations[0])
            self.assertEqual(invocations[3], invocations[0])

    def test_missing_pdfinfo_reports_poppler_guidance_and_blocks_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            with patch.object(
                report_builder, "_compile", self._successful_compile
            ), patch.object(report_builder, "_find_command", return_value=None):
                self.assertEqual(report_builder.build_report(args), 1)

            self.assertFalse(args.output.exists())
            audit = args.report.read_text(encoding="utf-8")
            self.assertIn("pdfinfo is unavailable", audit)
            self.assertIn("Install Poppler", audit)
            self.assertIn("poppler-utils", audit)
            self.assertIn("**Status: FAIL**", audit)

    def test_pdfinfo_discovery_uses_path_not_latex_bin_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latex_bin = root / "tex-bin"
            latex_bin.mkdir()
            pdf = root / "synthetic.pdf"
            with patch.dict(
                report_builder.os.environ,
                {"LATEX_BIN_DIR": str(latex_bin)},
                clear=False,
            ), patch.object(
                report_builder.shutil, "which", return_value=None
            ) as which:
                pages, message = report_builder._pdf_pages(pdf)

            which.assert_called_once_with("pdfinfo")
            self.assertIsNone(pages)
            self.assertIn("Install Poppler", message)

    def test_intermediate_undefined_warning_does_not_fail_clean_final_log(self) -> None:
        def compile_with_clean_final_log(tex_dir: Path, build_dir: Path):
            results = self._successful_compile(tex_dir, build_dir)
            (build_dir / "main.aux").write_text(
                r"\newlabel{LastPageMainText}{{}{1}}" + "\n", encoding="utf-8"
            )
            return [
                report_builder.CommandResult(
                    results[0].command,
                    tex_dir,
                    0,
                    "LaTeX Warning: There were undefined references.\n"
                    "Latexmk: final rerun completed successfully\n",
                )
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            with patch.object(
                report_builder, "_compile", compile_with_clean_final_log
            ), patch.object(report_builder, "_pdf_pages", return_value=(1, "Pages: 1\n")):
                self.assertEqual(report_builder.build_report(args), 0)

            self.assertTrue(args.output.is_file())
            self.assertIn("**Status: PASS**", args.report.read_text(encoding="utf-8"))


class FinalEvidenceInventoryTests(unittest.TestCase):
    def test_superseded_reports_are_snapshot_bound_not_canonical(self) -> None:
        historical = {
            "STAGE2_TRADITIONAL_REPORT.md",
            "STAGE2_FINAL_ACCEPTANCE_REPORT.md",
            "STAGE4_ADVANCED_ANALYSIS_REPORT.md",
        }
        frozen_paths = {artifact.relative_path for artifact in FROZEN_ARTIFACTS}
        self.assertTrue(historical.isdisjoint(frozen_paths))
        self.assertTrue(historical.issubset(READ_ONLY_REPORTS))

    def test_stage5_baseline_requires_rewrite_and_stage4_rerun_audits(self) -> None:
        self.assertIn("TRADITIONAL_REWRITE_INTEGRATION_AUDIT.md", READ_ONLY_REPORTS)
        self.assertIn("STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md", READ_ONLY_REPORTS)
        frozen_paths = {artifact.relative_path for artifact in FROZEN_ARTIFACTS}
        self.assertIn("STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md", frozen_paths)
        self.assertNotIn("STAGE4_ADVANCED_ANALYSIS_REPORT.md", frozen_paths)

    def test_inventory_uses_the_closed_frozen_evidence_set(self) -> None:
        paths = set(_discover(PROJECT_ROOT, DEFAULT_OUTPUT))
        self.assertIn("DATASET_SMOKE_TEST_REPORT.md", paths)
        self.assertIn("DEEP_LEARNING_SMOKE_TEST_REPORT.md", paths)
        self.assertIn("STAGE5_RESUME_AUDIT.md", paths)
        self.assertIn("STAGE5_RESUME_REPORT.md", paths)
        self.assertIn("TRADITIONAL_REWRITE_INTEGRATION_AUDIT.md", paths)
        self.assertIn("STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md", paths)
        self.assertIn("FINAL_POST_TRADITIONAL_REWRITE_ACCEPTANCE.md", paths)
        self.assertIn("presentation/COMP9517_Final_Presentation_10min.pptx", paths)
        self.assertTrue(HISTORICAL_REPORTS.issubset(paths))
        self.assertIn(
            "results/models/deep_learning/resnet18_pretrained_basic_best.pt", paths
        )
        self.assertTrue(
            any(path.startswith("results/predictions/robustness/") for path in paths)
        )
        self.assertTrue(
            any(path.endswith("_test_confusion.npz") for path in paths)
        )
        self.assertNotIn("results/final_evidence_inventory.csv", paths)

    def test_rewrite_and_presentation_items_are_classified_explicitly(self) -> None:
        self.assertEqual(
            _artifact_type("presentation/COMP9517_Final_Presentation_10min.pptx"),
            "presentation_deck",
        )
        self.assertEqual(
            _artifact_type("FINAL_POST_TRADITIONAL_REWRITE_ACCEPTANCE.md"),
            "report_or_audit",
        )
        self.assertEqual(_stage("TRADITIONAL_REWRITE_INTEGRATION_AUDIT.md"), "stage5")
        self.assertEqual(
            _stage("STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md"), "stage4"
        )
        self.assertEqual(
            _stage("FINAL_POST_TRADITIONAL_REWRITE_ACCEPTANCE.md"), "stage5"
        )

    def test_historical_reports_are_marked_as_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in HISTORICAL_REPORTS:
                (root / relative).write_text("historical\n", encoding="utf-8")
            output = root / "results/final_evidence_inventory.csv"
            with patch(
                "scripts.build_final_evidence_inventory._code_zip_paths",
                return_value=set(),
            ):
                rows = build_inventory(root, output)
            indexed = {str(row["file_path"]): row for row in rows}
            for relative in HISTORICAL_REPORTS:
                self.assertIn("superseded", str(indexed[relative]["reason"]))
                self.assertEqual(indexed[relative]["include_in_code_zip"], "no")


class ReproducibilityAuditSafetyTests(unittest.TestCase):
    def test_virtual_environment_launcher_is_not_resolved_to_base_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "venv" / "bin" / "python"
            launcher.parent.mkdir(parents=True)
            launcher.symlink_to(Path(sys.executable))
            report = root / "audit.md"
            args = argparse.Namespace(python=launcher, report=report, force=True)
            observed: list[tuple[str, ...]] = []

            def successful_run(label: str, command: list[str]):
                observed.append(tuple(command))
                return reproducibility_audit.AuditRun(label, tuple(command), 0, "ok", 0.0)

            with patch.object(reproducibility_audit, "_run", side_effect=successful_run):
                self.assertEqual(reproducibility_audit.run_audit(args), 0)

            self.assertEqual(len(observed), 6)
            self.assertTrue(all(command[0] == str(launcher.absolute()) for command in observed))
            self.assertEqual(
                observed[3][1:],
                ("scripts/audit_final_report.py", "--no-write-reports"),
            )
            self.assertEqual(
                observed[4][1:],
                ("scripts/make_code_zip.py", "--list-only"),
            )
            self.assertIn("--hash-files", observed[5])
            self.assertIn("--split-dir", observed[5])
            self.assertIn("--train-json", observed[5])
            self.assertIn("--val-json", observed[5])
            self.assertIn("--image-root", observed[5])
            self.assertIn(str(launcher.absolute()), report.read_text(encoding="utf-8"))

    def test_existing_different_audit_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "audit.md"
            report.write_text("known-good sentinel", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "use --force"):
                reproducibility_audit._atomic_text(report, "replacement")
            self.assertEqual(report.read_text(encoding="utf-8"), "known-good sentinel")

            reproducibility_audit._atomic_text(report, "known-good sentinel")
            reproducibility_audit._atomic_text(report, "replacement", force=True)
            self.assertEqual(report.read_text(encoding="utf-8"), "replacement")


if __name__ == "__main__":
    unittest.main()
