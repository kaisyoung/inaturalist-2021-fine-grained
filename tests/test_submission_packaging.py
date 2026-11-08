from __future__ import annotations

import tempfile
import unittest
import zipfile
import hashlib
import shutil
import stat
from pathlib import Path

from scripts.check_submission_size import (
    REPORT_PDF_LIMIT_BYTES,
    atomic_write_audit,
    check_submission,
    render_audit_report,
    validate_clean_extraction,
    validate_report_pdf,
)
from scripts.make_code_zip import (
    ARCHIVE_ROOT,
    DEFAULT_OUTPUT,
    MANIFEST_NAME,
    PackagingError,
    REQUIRED_REWRITE_EVIDENCE_FILES,
    build_code_zip,
    collect_allowed_files,
    validate_code_zip,
)


class SubmissionPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        files = {
            ".gitignore": "*.pyc\n",
            "README.md": "# Synthetic project\n",
            "requirements.txt": "numpy==2.2.6\n",
            "src/example.py": "VALUE = 1\n",
            "scripts/tool.py": "print('tool')\n",
            "configs/model.yaml": "seed: 42\n",
            "tests/test_example.py": "# unit test\n",
            "data/splits/selected_classes_seed42.txt": "1\n",
            "data/splits/class_mapping.json": "{}\n",
            "data/splits/train.csv": "new_label\n0\n",
            "data/splits/val.csv": "new_label\n0\n",
            "data/splits/test.csv": "new_label\n0\n",
            "data/splits/split_summary.json": "{}\n",
            "data/splits/selected_classes_readable.csv": "new_label,name\n0,toy\n",
            "results/metrics/final.json": "{\"status\": \"complete\"}\n",
            "results/metrics/stage2_selection_manifest.json": "{}\n",
            "results/metrics/stage2_freeze.json": "{}\n",
            "results/metrics/deep_learning/stage3_traditional_rewrite_continuity.json": "{}\n",
            "results/metrics/stage4_robustness_protocol.json": "{}\n",
            "results/metrics/confusion.npz": "forbidden binary array",
            "results/tables/final.csv": "metric,value\ntop1,0.5\n",
            "results/tables/analysis.md": "# Analysis\n",
            "results/tables/traditional_validation_grid.csv": "experiment,val_top1\nselected,0.5\n",
            "results/tables/stage2_traditional_results.csv": "model_name,test_top1\nlinear_svm,0.5\n",
            "results/tables/stage2_feature_ablation_results.csv": "experiment,val_top1\nablation,0.4\n",
            "results/tables/final_model_comparison.csv": "model_name,test_top1\nlinear_svm,0.5\n",
            "results/tables/robustness_results.csv": "model_name,top1\ntraditional_random_forest,0.4\n",
            "results/tables/robustness_drop_summary.csv": "model_name,drop\ntraditional_random_forest,0.1\n",
            "results/tables/robustness_model_ranking.csv": "model_name,rank\ntraditional_random_forest,1\n",
            "results/tables/stage4_summary.csv": "model_name,score\ntraditional_random_forest,0.4\n",
            "results/tables/final_model_comparison_with_robustness.csv": "model_name,test_top1\nlinear_svm,0.5\n",
            "results/tables/final_report_main_results.csv": "method_name,test_top1\nlinear_svm,0.5\n",
            "results/tables/final_report_ablation_results.csv": "experiment_name,val_top1\nablation,0.4\n",
            "results/tables/final_report_robustness_summary.csv": "model_name,rank\nlinear_svm,1\n",
            "results/tables/final_report_tables_manifest.json": "temporary\n",
            "report/cvpr_latex/main.tex": "\\documentclass{article}\n",
            "report/cvpr_latex/refs.bib": "@misc{toy}\n",
            "report/cvpr_latex/cvpr.sty": "% external template\n",
            "report/cvpr_latex/notes.md": "report source notes\n",
            "report/cvpr_latex/figure.png": "forbidden image",
            "results/figures/result.png": "forbidden result image",
            "results/models/model.pt": "forbidden checkpoint",
            "results/predictions/test.csv": "forbidden predictions",
            "data/raw/train.tar.gz": "forbidden raw data",
            "docs/spec.pdf": "forbidden course PDF",
            "pet-runs/pet.png": "unrelated generated asset",
            "presentation/generate_slides.py": "print('portable wrapper')\n",
            "presentation/generate_slides.mjs": "export const generator = true;\n",
            "STAGE_INTERNAL_REPORT.md": "not part of source submission\n",
        }
        for relative, content in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        table_outputs = {
            relative: hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            for relative in (
                "results/tables/final_report_main_results.csv",
                "results/tables/final_report_ablation_results.csv",
                "results/tables/final_report_robustness_summary.csv",
            )
        }
        table_source = "results/tables/stage4_summary.csv"
        table_manifest = {
            "schema_version": 1,
            "sources": {
                table_source: hashlib.sha256(
                    (self.root / table_source).read_bytes()
                ).hexdigest()
            },
            "outputs": table_outputs,
        }
        import json

        (self.root / "results/tables/final_report_tables_manifest.json").write_text(
            json.dumps(table_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_public_allowlist_is_sorted_and_excludes_every_binary_artifact(self) -> None:
        allowed = collect_allowed_files(self.root)
        values = [path.as_posix() for path in allowed]
        self.assertEqual(values, sorted(values))
        self.assertIn("results/metrics/final.json", values)
        self.assertIn("results/tables/analysis.md", values)
        self.assertIn("results/tables/final_report_tables_manifest.json", values)
        self.assertIn("report/cvpr_latex/main.tex", values)
        self.assertIn("presentation/generate_slides.py", values)
        self.assertIn("presentation/generate_slides.mjs", values)
        self.assertTrue(
            {path.as_posix() for path in REQUIRED_REWRITE_EVIDENCE_FILES}.issubset(
                values
            )
        )
        for forbidden in (
            "results/metrics/confusion.npz",
            "report/cvpr_latex/figure.png",
            "results/figures/result.png",
            "results/models/model.pt",
            "results/predictions/test.csv",
            "data/raw/train.tar.gz",
            "docs/spec.pdf",
            "pet-runs/pet.png",
            "STAGE_INTERNAL_REPORT.md",
        ):
            self.assertNotIn(forbidden, values)

    def test_two_builds_are_byte_identical_and_manifest_verified(self) -> None:
        first_path = self.root / "submission/first.zip"
        second_path = self.root / "submission/second.zip"
        first = build_code_zip(first_path, project_root=self.root)
        second = build_code_zip(second_path, project_root=self.root)
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        self.assertEqual(first["file_count"], second["file_count"])
        with zipfile.ZipFile(first_path) as archive:
            names = archive.namelist()
        self.assertIn(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}", names)
        self.assertTrue(all(name.startswith(f"{ARCHIVE_ROOT}/") for name in names))
        validate_code_zip(
            first_path,
            expected_files=collect_allowed_files(self.root),
            project_root=self.root,
        )

    def test_existing_different_zip_requires_force_and_is_atomically_replaced(self) -> None:
        output = self.root / "submission/code.zip"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"pre-existing sentinel")
        with self.assertRaises(FileExistsError):
            build_code_zip(output, project_root=self.root)
        self.assertEqual(output.read_bytes(), b"pre-existing sentinel")
        result = build_code_zip(output, project_root=self.root, force=True)
        self.assertGreater(int(result["file_count"]), 10)
        self.assertTrue(zipfile.is_zipfile(output))

    def test_tampered_or_traversing_archives_are_rejected(self) -> None:
        output = self.root / "submission/code.zip"
        build_code_zip(output, project_root=self.root)
        with zipfile.ZipFile(output, "a") as archive:
            archive.writestr(f"{ARCHIVE_ROOT}/forbidden.png", b"pixels")
        with self.assertRaises(PackagingError):
            validate_code_zip(output)

        traversal = self.root / "submission/traversal.zip"
        with zipfile.ZipFile(traversal, "w") as archive:
            archive.writestr(f"{ARCHIVE_ROOT}/../escape.py", b"bad")
        with self.assertRaises(PackagingError):
            validate_code_zip(traversal)

        noncanonical = self.root / "submission/noncanonical.zip"
        with zipfile.ZipFile(noncanonical, "w") as archive:
            archive.writestr(f"{ARCHIVE_ROOT}/src/./escape.py", b"bad")
        with self.assertRaises(PackagingError):
            validate_code_zip(noncanonical)

    def test_special_files_and_zip_bombs_are_rejected_before_extraction(self) -> None:
        regular = self.root / "submission/regular.zip"
        build_code_zip(regular, project_root=self.root)
        fifo = self.root / "submission/fifo.zip"
        with zipfile.ZipFile(regular, "r") as source, zipfile.ZipFile(fifo, "w") as target:
            for source_info in source.infolist():
                copied = zipfile.ZipInfo(source_info.filename, source_info.date_time)
                copied.compress_type = source_info.compress_type
                copied.create_system = 3
                copied.external_attr = source_info.external_attr
                if source_info.filename.endswith("/src/example.py"):
                    copied.external_attr = (stat.S_IFIFO | 0o644) << 16
                target.writestr(copied, source.read(source_info))
        with self.assertRaisesRegex(PackagingError, "not a regular file"):
            validate_code_zip(fifo)

        bomb = self.root / "submission/bomb.zip"
        relative = "src/bomb.py"
        payload = b"0" * 1_000_000
        manifest = f"{hashlib.sha256(payload).hexdigest()}  {relative}\n".encode()
        with zipfile.ZipFile(
            bomb, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr(f"{ARCHIVE_ROOT}/{relative}", payload)
            archive.writestr(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}", manifest)
        with self.assertRaisesRegex(PackagingError, "compression ratio"):
            validate_code_zip(bomb)

    def test_size_limit_is_decimal_and_enforced(self) -> None:
        output = self.root / "submission/code.zip"
        build_code_zip(output, project_root=self.root)
        with self.assertRaises(PackagingError):
            validate_code_zip(output, max_bytes=1)

    def test_report_pdf_header_trailer_and_limit(self) -> None:
        report = self.root / "report/COMP9517_Group_Project_Report.pdf"
        report.parent.mkdir(exist_ok=True)
        report.write_bytes(b"%PDF-1.7\nsynthetic\n%%EOF\n")
        result = validate_report_pdf(report, project_root=self.root)
        self.assertEqual(result["report_bytes"], report.stat().st_size)
        with self.assertRaises(PackagingError):
            validate_report_pdf(report, project_root=self.root, max_bytes=8)
        report.write_bytes(b"not a pdf\n%%EOF\n")
        with self.assertRaises(PackagingError):
            validate_report_pdf(report, project_root=self.root)
        self.assertEqual(REPORT_PDF_LIMIT_BYTES, 10_000_000)

    def test_checker_and_atomic_audit_report(self) -> None:
        output = self.root / "submission/code.zip"
        build_code_zip(output, project_root=self.root)
        result = check_submission(
            project_root=self.root,
            code_zip=output,
            skip_report=True,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["report_pdf"]["status"], "SKIP")
        self.assertEqual(result["clean_extraction"]["status"], "PASS")
        self.assertEqual(
            result["clean_extraction"]["top_level_root"], ARCHIVE_ROOT
        )
        self.assertEqual(
            result["clean_extraction"]["compileall"]["status"], "PASS"
        )
        audit = render_audit_report(result, command=["python", "checker.py"])
        self.assertIn(f"`{ARCHIVE_ROOT}/{MANIFEST_NAME}`", audit)
        self.assertIn(f"`{ARCHIVE_ROOT}/README.md`", audit)
        audit_path = atomic_write_audit(
            "SUBMISSION_PACKAGE_AUDIT.md", audit, project_root=self.root
        )
        self.assertIn("**Status: PASS**", audit_path.read_text(encoding="utf-8"))
        # Exact reruns are harmless, while different content needs force.
        atomic_write_audit(audit_path, audit, project_root=self.root)
        with self.assertRaises(FileExistsError):
            atomic_write_audit(audit_path, audit + "changed", project_root=self.root)
        atomic_write_audit(
            audit_path, audit + "changed", project_root=self.root, force=True
        )

    def test_missing_core_split_is_fatal(self) -> None:
        self.assertEqual(
            DEFAULT_OUTPUT, Path("submission/COMP9517_Group_Project_Code.zip")
        )
        (self.root / "data/splits/test.csv").unlink()
        with self.assertRaises(PackagingError):
            collect_allowed_files(self.root)

        # The seventh human-readable class manifest is part of the submission
        # contract, not merely opportunistically included when present.
        (self.root / "data/splits/test.csv").write_text("new_label\n0\n", encoding="utf-8")
        (self.root / "data/splits/selected_classes_readable.csv").unlink()
        with self.assertRaises(PackagingError):
            collect_allowed_files(self.root)

    def test_every_rewrite_evidence_file_is_mandatory(self) -> None:
        for relative in REQUIRED_REWRITE_EVIDENCE_FILES:
            with self.subTest(relative=relative.as_posix()):
                path = self.root / relative
                original = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(
                    PackagingError, "traditional-rewrite evidence|submission file"
                ):
                    collect_allowed_files(self.root)
                path.write_bytes(original)

    def test_retired_linearsvc_attempt_is_fatal(self) -> None:
        legacy = self.root / "results/metrics/linear_svm_linearsvc_attempt.json"
        legacy.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(PackagingError, "Retired pre-rewrite evidence"):
            collect_allowed_files(self.root)

    def test_report_table_manifest_hash_mismatch_is_fatal(self) -> None:
        output = self.root / "results/tables/final_report_main_results.csv"
        output.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(PackagingError, "manifest hash mismatch"):
            collect_allowed_files(self.root)

    def test_cvpr_report_source_directory_is_mandatory(self) -> None:
        shutil.rmtree(self.root / "report/cvpr_latex")
        with self.assertRaisesRegex(
            PackagingError, "Required submission directory is missing"
        ):
            collect_allowed_files(self.root)

    def test_clean_extraction_rejects_uncompilable_submitted_source(self) -> None:
        (self.root / "src/example.py").write_text("def broken(:\n", encoding="utf-8")
        output = self.root / "submission/code.zip"
        build_code_zip(output, project_root=self.root)
        with self.assertRaisesRegex(PackagingError, "compilation failed"):
            validate_clean_extraction(output)


if __name__ == "__main__":
    unittest.main()
