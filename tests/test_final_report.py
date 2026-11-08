"""Focused tests for Stage 5 report figures and source auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.analysis.generate_final_report_figures import (
    DEFAULT_OUTPUT_DIR,
    extract_dense_panel,
)
from src.analysis.report_audit import (
    DEFAULT_TEX_DIR,
    OBSERVED_CLAIMS,
    REFERENCE_COVERAGE,
    _claim_checks,
    _reference_coverage_check,
    _traditional_rewrite_contract_check,
    bibliography_keys,
    citation_keys,
    label_and_reference_keys,
    placeholder_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReportParserTests(unittest.TestCase):
    COMPLETED_IDENTITIES = (
        "Ada Lovelace (z1234567)",
        "Grace Hopper (z2345678)",
        "Katherine Johnson (z3456789)",
        "Alan Turing (z4567890)",
        "Edsger Dijkstra (z5678901)",
    )

    @staticmethod
    def _main_with_identities(identities: tuple[str, ...]) -> str:
        return "\\author{" + r" \quad ".join(identities) + r"}\begin{document}"

    def test_citation_and_bibliography_parsers(self) -> None:
        tex = r"A \cite{one,two} and \citep{three}."
        bib = "@article{one,}\n@misc{two,}\n@software{three,}"
        self.assertEqual(citation_keys(tex), {"one", "two", "three"})
        self.assertEqual(bibliography_keys(bib), ["one", "two", "three"])

    def test_label_parser_exposes_duplicate_or_missing_inputs(self) -> None:
        labels, refs = label_and_reference_keys(r"\label{fig:a}\ref{fig:a}\eqref{eq:b}")
        self.assertEqual(labels, ["fig:a"])
        self.assertEqual(refs, ["fig:a", "eq:b"])

    def test_author_placeholders_block_by_default(self) -> None:
        authors = tuple(
            f"TODO Author {index} Name (TODO zID {index})" for index in range(1, 6)
        )
        main = self._main_with_identities(authors)
        self.assertEqual(placeholder_check(main, allow_placeholders=False).status, "FAIL")
        self.assertEqual(placeholder_check(main, allow_placeholders=True).status, "WARN")

    def test_completed_author_identities_pass(self) -> None:
        main = self._main_with_identities(self.COMPLETED_IDENTITIES)
        self.assertEqual(placeholder_check(main, allow_placeholders=False).status, "PASS")
        self.assertEqual(placeholder_check(main, allow_placeholders=True).status, "PASS")

    def test_duplicate_completed_zids_fail(self) -> None:
        identities = (*self.COMPLETED_IDENTITIES[:-1], "Edsger Dijkstra (z1234567)")
        main = self._main_with_identities(identities)
        self.assertEqual(placeholder_check(main, allow_placeholders=True).status, "FAIL")

    def test_mixed_placeholder_and_completed_identities_fail(self) -> None:
        identities = (
            "TODO Author 1 Name (TODO zID 1)",
            *self.COMPLETED_IDENTITIES[1:],
        )
        main = self._main_with_identities(identities)
        self.assertEqual(placeholder_check(main, allow_placeholders=True).status, "FAIL")

    def test_traditional_rewrite_contract_accepts_current_and_rejects_stale_prose(self) -> None:
        current = (
            "The primary Stage 2 traditional comparison is the 3,704-D LinearSVC. "
            "The 2,424-D, 300-tree handcrafted Random Forest is the canonical "
            "traditional robustness comparator. The 1,280-D BoVW block uses "
            "top-3 soft assignment. We recorded 19 validation configurations."
        )
        self.assertEqual(_traditional_rewrite_contract_check(current).status, "PASS")
        stale = current + " Retired: 1,830-D hinge-SGD SGDClassifier with 200 trees."
        check = _traditional_rewrite_contract_check(stale)
        self.assertEqual(check.status, "FAIL")
        self.assertIn("stale 1,830-D descriptor", check.detail)
        self.assertIn("retired hinge-SGD model", check.detail)
        self.assertIn("retired SGDClassifier backend", check.detail)
        self.assertIn("stale 200-tree Random Forest", check.detail)

    def test_all_twenty_observed_macros_are_bound_to_source_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_dir = root / "results/tables"
            table_dir.mkdir(parents=True)
            (table_dir / "final_model_comparison.csv").write_text(
                "model_name,test_top1,test_top5,test_macro_f1\n"
                "linear_svm,0.11,0.21,0.31\n"
                "random_forest,0.12,0.22,0.32\n"
                "sift_bovw_svm,0.13,0.23,0.33\n"
                "efficientnet_b0_pretrained_strong_aug,0.14,0.24,0.34\n",
                encoding="utf-8",
            )
            (table_dir / "robustness_model_ranking.csv").write_text(
                "model_name,mean_corrupted_top1,mean_corrupted_macro_f1,"
                "worst_condition_top1,worst_condition_macro_f1\n"
                "efficientnet_b0_pretrained_strong_aug,0.41,0.42,0.43,0.44\n"
                "traditional_random_forest,0.51,0.52,0.53,0.54\n",
                encoding="utf-8",
            )
            values = {
                "test_top1": {
                    "linear_svm": 0.11,
                    "random_forest": 0.12,
                    "sift_bovw_svm": 0.13,
                    "efficientnet_b0_pretrained_strong_aug": 0.14,
                },
                "test_top5": {
                    "linear_svm": 0.21,
                    "random_forest": 0.22,
                    "sift_bovw_svm": 0.23,
                    "efficientnet_b0_pretrained_strong_aug": 0.24,
                },
                "test_macro_f1": {
                    "linear_svm": 0.31,
                    "random_forest": 0.32,
                    "sift_bovw_svm": 0.33,
                    "efficientnet_b0_pretrained_strong_aug": 0.34,
                },
                "mean_corrupted_top1": {
                    "efficientnet_b0_pretrained_strong_aug": 0.41,
                    "traditional_random_forest": 0.51,
                },
                "mean_corrupted_macro_f1": {
                    "efficientnet_b0_pretrained_strong_aug": 0.42,
                    "traditional_random_forest": 0.52,
                },
                "worst_condition_top1": {
                    "efficientnet_b0_pretrained_strong_aug": 0.43,
                    "traditional_random_forest": 0.53,
                },
                "worst_condition_macro_f1": {
                    "efficientnet_b0_pretrained_strong_aug": 0.44,
                    "traditional_random_forest": 0.54,
                },
            }
            report_lines = []
            for claim in OBSERVED_CLAIMS:
                observed = (
                    f"{values[claim.column][claim.model_name] * 100:.{claim.decimals}f}"
                )
                report_lines.extend(
                    (
                        f"\\newcommand{{\\{claim.macro_name}}}{{{observed}}}",
                        f"\\{claim.macro_name}",
                    )
                )
            report_text = "\n".join(report_lines)
            self.assertEqual(len(OBSERVED_CLAIMS), 20)
            self.assertEqual(_claim_checks(report_text, root)[0].status, "PASS")

            missing_use = report_text.replace(
                f"\n\\{OBSERVED_CLAIMS[0].macro_name}\n", "\n", 1
            )
            self.assertEqual(_claim_checks(missing_use, root)[0].status, "FAIL")


class ReportFigureTests(unittest.TestCase):
    def test_dense_panel_extraction_removes_sparse_title(self) -> None:
        image = Image.new("RGB", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 10), "sparse title", fill="black")
        draw.rectangle((20, 90, 180, 270), fill=(30, 90, 150))
        panel = extract_dense_panel(image, 0, panel_count=4)
        pixels = np.asarray(panel)
        self.assertLess(panel.width, 190)
        self.assertGreater(panel.height, 170)
        self.assertGreater(float((pixels[:, :, 2] > 100).mean()), 0.80)

    def test_report_generator_has_no_inference_imports(self) -> None:
        source = (PROJECT_ROOT / "src/analysis/generate_final_report_figures.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("src.deep_learning", source)
        self.assertIn("no model inference", source)

    @unittest.skipUnless(
        (DEFAULT_OUTPUT_DIR / "figure_manifest.json").is_file(),
        "canonical persisted report figures are intentionally excluded from the code-only ZIP",
    )
    def test_manifest_binds_lossless_vector_and_category_outputs(self) -> None:
        manifest = json.loads((DEFAULT_OUTPUT_DIR / "figure_manifest.json").read_text(encoding="utf-8"))
        outputs = {row["file"]: row for row in manifest["outputs"]}
        required = {
            "method_overview.pdf",
            "training_convergence.png",
            "gradcam_correct_examples.jpg",
            "gradcam_incorrect_examples.jpg",
            "gradcam_confused_pairs.jpg",
            "robustness_top1_curves.png",
            "robustness_macro_f1_curves.png",
        }
        self.assertTrue(required.issubset(outputs))
        self.assertNotIn("training_convergence.jpg", outputs)
        self.assertTrue((DEFAULT_OUTPUT_DIR / "method_overview.pdf").read_bytes().startswith(b"%PDF"))


class ReportSourceTests(unittest.TestCase):
    def test_repository_has_exact_five_author_identities_in_presentation_order(self) -> None:
        main = (DEFAULT_TEX_DIR / "main.tex").read_text(encoding="utf-8")
        identities = (
            "Yuan Cao",
            "Kaishi Yang",
            "Aditya Winata",
            "Jialin Guo",
            "Yuzhe Qiu",
        )
        positions = [main.index(identity) for identity in identities]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(all(main.count(identity) == 1 for identity in identities))

    def test_static_traditional_rewrite_prose_uses_local_evidence_not_snapshot_values(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        report_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [
                DEFAULT_TEX_DIR / "main.tex",
                *sorted((DEFAULT_TEX_DIR / "sections").glob("*.tex")),
            ]
        )
        for value in ("0.1252", "0.2678", "0.111589"):
            self.assertIn(value, readme)
        for stale in ("0.1302", "0.2682", "0.1167", "1,830-D", "hinge-SGD", "4.10%", r"4.10\%"):
            self.assertNotIn(stale, readme)
            self.assertNotIn(stale, report_text)
        for required in ("3,704-D", "2,424-D", "1,280-D", "19"):
            self.assertIn(required, report_text)
        self.assertIn(r"\PrimaryTraditionalTestTopOnePercent", report_text)
        self.assertIn(r"\PrimaryTraditionalTestTopFivePercent", report_text)
        self.assertIn(r"\PrimaryTraditionalTestMacroFOnePercent", report_text)

    def test_direct_audit_wrapper_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/audit_final_report.py", "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("--allow-placeholders", completed.stdout)

    def test_exact_cvpr_mode_and_main_text_boundary(self) -> None:
        main = (DEFAULT_TEX_DIR / "main.tex").read_text(encoding="utf-8")
        preamble = (DEFAULT_TEX_DIR / "preamble.tex").read_text(encoding="utf-8")
        self.assertEqual(main.count(r"\usepackage[pagenumbers]{cvpr}"), 1)
        self.assertNotIn(r"\usepackage[review]{cvpr}", main)
        self.assertIn(r"\usepackage{placeins}", preamble)
        self.assertLess(main.index(r"\FloatBarrier"), main.index(r"\label{LastPageMainText}"))
        self.assertLess(main.index(r"\label{LastPageMainText}"), main.index(r"\clearpage"))

    def test_all_citations_exist_and_no_bibliography_entry_is_unused(self) -> None:
        tex = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [DEFAULT_TEX_DIR / "main.tex", *sorted((DEFAULT_TEX_DIR / "sections").glob("*.tex"))]
        )
        cited = citation_keys(tex)
        bib_keys = bibliography_keys((DEFAULT_TEX_DIR / "references.bib").read_text(encoding="utf-8"))
        self.assertEqual(len(bib_keys), len(set(bib_keys)))
        self.assertEqual(cited, set(bib_keys))
        reference_check = _reference_coverage_check(
            cited,
            (DEFAULT_TEX_DIR / "references.bib").read_text(encoding="utf-8"),
        )
        self.assertEqual(reference_check.status, "PASS")
        curated_count = len(set().union(*REFERENCE_COVERAGE.values()))
        self.assertIn(f"All {curated_count} curated", reference_check.detail)

    def test_repository_author_identity_is_valid_handoff_or_completed(self) -> None:
        main = (DEFAULT_TEX_DIR / "main.tex").read_text(encoding="utf-8")
        permissive = placeholder_check(main, allow_placeholders=True).status
        strict = placeholder_check(main, allow_placeholders=False).status
        self.assertIn((permissive, strict), {("WARN", "FAIL"), ("PASS", "PASS")})


if __name__ == "__main__":
    unittest.main()
