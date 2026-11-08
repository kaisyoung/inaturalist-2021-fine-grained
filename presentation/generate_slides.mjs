import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.dirname(HERE);

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = value;
      index += 1;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const OUTPUT = path.resolve(
  args.output || path.join(HERE, "COMP9517_Final_Presentation_10min.pptx"),
);
const WORKSPACE = path.resolve(args.workspace || path.join(HERE, ".tmp", "build"));
const RENDER_DIR = path.resolve(
  args["render-dir"] || path.join(HERE, ".tmp", "artifact-render"),
);
const ASSET_DIR = path.join(HERE, "assets");

const EVIDENCE = {
  stage2Freeze: "results/metrics/stage2_freeze.json",
  stage2Selection: "results/metrics/stage2_selection_manifest.json",
  stage4Protocol: "results/metrics/stage4_robustness_protocol.json",
  mainResults: "results/tables/final_report_main_results.csv",
  ablationResults: "results/tables/final_report_ablation_results.csv",
  robustnessSummary: "results/tables/final_report_robustness_summary.csv",
  robustnessRanking: "results/tables/robustness_model_ranking.csv",
  robustnessResults: "results/tables/robustness_results.csv",
  robustnessDropSummary: "results/tables/robustness_drop_summary.csv",
  stage2FeatureAblation: "results/tables/stage2_feature_ablation_results.csv",
  stage2TraditionalResults: "results/tables/stage2_traditional_results.csv",
  stage3Ablation: "results/tables/stage3_ablation_results.csv",
  stage4Summary: "results/tables/stage4_summary.csv",
  finalModelComparison: "results/tables/final_model_comparison.csv",
  reportTablesManifest: "results/tables/final_report_tables_manifest.json",
  figureManifest: "results/final_report_figures/figure_manifest.json",
  randomForestMetadata: "results/models/traditional/random_forest.metadata.json",
  traditionalFeatureConfig: "results/features/traditional_feature_config.json",
  bovwFeatureConfig: "results/features/bovw/bovw_feature_config.json",
  gradcamSelection: "results/metrics/explainability/gradcam_selection_manifest.json",
  gradcamManifest: "results/metrics/explainability/gradcam_manifest.json",
  gradcamSelectedExamples: "results/tables/gradcam_selected_examples.csv",
};

const MAIN_METHODS = [
  "Handcrafted + BoVW (LinearSVC)",
  "Random Forest",
  "SIFT-BoVW (LinearSVC)",
  "ResNet18 (scratch)",
  "ResNet18 (ImageNet pretrained)",
  "EfficientNet-B0 (ImageNet pretrained)",
];

const ABLATION_EXPERIMENTS = [
  "full",
  "ablation_only_color",
  "ablation_only_texture",
  "ablation_only_hog",
  "resnet18_scratch_basic",
  "resnet18_pretrained_basic",
  "resnet18_pretrained_strong_aug",
  "resnet18_pretrained_frozen_head",
];

const ROBUSTNESS_MODELS = [
  "traditional_random_forest",
  "resnet18_scratch_basic",
  "resnet18_pretrained_strong_aug",
  "efficientnet_b0_pretrained_strong_aug",
];

const ROBUSTNESS_TO_MAIN_METHOD = {
  traditional_random_forest: "Random Forest",
  resnet18_scratch_basic: "ResNet18 (scratch)",
  resnet18_pretrained_strong_aug: "ResNet18 (ImageNet pretrained)",
  efficientnet_b0_pretrained_strong_aug: "EfficientNet-B0 (ImageNet pretrained)",
};
const ROBUSTNESS_REPORT_NAMES = ROBUSTNESS_MODELS.map(
  (modelName) => ROBUSTNESS_TO_MAIN_METHOD[modelName],
);

const CORRUPTION_SCHEDULE = {
  gaussian_blur: { parameter: "kernel_size", severities: [3, 5, 9, 15] },
  gaussian_noise: { parameter: "sigma", severities: [0.02, 0.05, 0.1, 0.2] },
  jpeg_compression: { parameter: "quality", severities: [80, 50, 30, 10] },
  motion_blur: { parameter: "kernel_length", severities: [5, 9, 15, 25] },
};

function projectPath(relativePath) {
  const resolved = path.resolve(PROJECT_ROOT, relativePath);
  const relative = path.relative(PROJECT_ROOT, resolved);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Evidence path escapes the project root: ${relativePath}`);
  }
  return resolved;
}

async function requiredBytes(relativePath) {
  try {
    return await fs.readFile(projectPath(relativePath));
  } catch (error) {
    if (error && error.code === "ENOENT") {
      throw new Error(
        `Presentation evidence is not ready: missing ${relativePath}. ` +
          "Build and bind downstream Stage 2 and current-protocol Stage 4 evidence before building the deck.",
      );
    }
    throw error;
  }
}

async function requiredText(relativePath) {
  return (await requiredBytes(relativePath)).toString("utf8");
}

async function requiredPresentationAssetBytes(fileName) {
  try {
    return await fs.readFile(path.join(ASSET_DIR, fileName));
  } catch (error) {
    if (error && error.code === "ENOENT") {
      throw new Error(`Presentation evidence is not ready: missing presentation/assets/${fileName}`);
    }
    throw error;
  }
}

async function requiredJson(relativePath) {
  const textValue = await requiredText(relativePath);
  try {
    return JSON.parse(textValue);
  } catch (error) {
    throw new Error(`Presentation evidence is not valid JSON: ${relativePath}`, {
      cause: error,
    });
  }
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

async function sha256Path(relativePath) {
  return sha256(await requiredBytes(relativePath));
}

function parseCsv(textValue, label) {
  const matrix = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < textValue.length; index += 1) {
    const character = textValue[index];
    if (quoted) {
      if (character === '"' && textValue[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      matrix.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (quoted) throw new Error(`Unterminated CSV quote in ${label}`);
  if (field !== "" || row.length > 0) {
    row.push(field.replace(/\r$/, ""));
    matrix.push(row);
  }
  if (matrix.length < 2) throw new Error(`CSV has no data rows: ${label}`);
  const headers = matrix[0];
  if (new Set(headers).size !== headers.length || headers.some((header) => !header)) {
    throw new Error(`CSV has invalid headers: ${label}`);
  }
  return matrix.slice(1).map((values, rowIndex) => {
    if (values.length !== headers.length) {
      throw new Error(`CSV row ${rowIndex + 2} has the wrong width: ${label}`);
    }
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function exactIndex(rows, key, expectedValues, label) {
  const index = new Map();
  for (const row of rows) {
    const value = String(row[key] || "");
    if (!value || index.has(value)) throw new Error(`Duplicate or empty ${key} in ${label}`);
    index.set(value, row);
  }
  const observed = [...index.keys()].sort();
  const expected = [...expectedValues].sort();
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(
      `Unexpected ${key} set in ${label}: expected ${expected.join(", ")}; ` +
        `observed ${observed.join(", ")}`,
    );
  }
  return index;
}

function finiteMetric(row, key, label) {
  if (String(row[key] ?? "").trim() === "") {
    throw new Error(`Missing unit-interval metric ${label}.${key}`);
  }
  const value = Number(row[key]);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`Invalid unit-interval metric ${label}.${key}: ${row[key]}`);
  }
  return value;
}

function finiteInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`Invalid integer ${label}: ${value}`);
  return parsed;
}

async function requireHashBinding(container, section, relativePath) {
  const expected = container?.[section]?.[relativePath];
  if (!/^[0-9a-f]{64}$/.test(String(expected || ""))) {
    throw new Error(`Missing SHA-256 binding for ${relativePath} in ${section}`);
  }
  const actual = await sha256Path(relativePath);
  if (actual !== expected) {
    throw new Error(
      `Presentation evidence hash mismatch for ${relativePath}: ` +
        `expected ${expected}, observed ${actual}`,
    );
  }
}

async function requireArtifactReference(reference, label) {
  if (!reference || typeof reference !== "object") {
    throw new Error(`Missing artifact reference: ${label}`);
  }
  const relativePath = String(reference.path || "");
  const expectedHash = String(reference.sha256 || "");
  if (!relativePath || !/^[0-9a-f]{64}$/.test(expectedHash)) {
    throw new Error(`Invalid artifact reference: ${label}`);
  }
  const bytes = await requiredBytes(relativePath);
  const actualHash = sha256(bytes);
  if (actualHash !== expectedHash) {
    throw new Error(`${label} does not match the frozen protocol binding: ${relativePath}`);
  }
  if (reference.bytes !== undefined && Number(reference.bytes) !== bytes.byteLength) {
    throw new Error(`${label} byte count does not match the frozen protocol binding`);
  }
}

function requireStage2Contract(stage2Freeze) {
  if (stage2Freeze.status !== "frozen") throw new Error("Stage 2 freeze is not frozen");
  if (
    JSON.stringify(stage2Freeze.allowed_test_models) !==
    JSON.stringify(["linear_svm", "random_forest", "sift_bovw_svm"])
  ) {
    throw new Error("Stage 2 freeze does not contain the exact approved test-model roster");
  }
  if (
    stage2Freeze.validation_grid?.rows !== 19 ||
    stage2Freeze.validation_grid?.unique_configuration_fingerprints !== 19
  ) {
    throw new Error("Stage 2 freeze does not bind exactly 19 unique validation configurations");
  }
  const models = stage2Freeze.models || {};
  const primary = models.linear_svm || {};
  const forest = models.random_forest || {};
  const bovw = models.sift_bovw_svm || {};
  if (
    primary.classifier !== "LinearSVC" ||
    primary.feature_dim !== 3704 ||
    primary.feature_set !== "combined" ||
    primary.normalisation?.alpha !== 0.3 ||
    primary.normalisation?.blockwise !== true
  ) {
    throw new Error("Stage 2 primary model is not the frozen 3,704-D block-wise LinearSVC");
  }
  if (
    forest.classifier !== "RandomForestClassifier" ||
    forest.feature_dim !== 2424 ||
    forest.feature_set !== "handcrafted" ||
    forest.estimator_parameters?.["classifier__n_estimators"] !== 300 ||
    (forest.feature_components || []).includes("bovw")
  ) {
    throw new Error("Stage 2 Random Forest is not the 2,424-D, 300-tree handcrafted comparator");
  }
  if (
    bovw.classifier !== "LinearSVC" ||
    bovw.feature_dim !== 1280 ||
    bovw.feature_set !== "bovw"
  ) {
    throw new Error("Stage 2 BoVW baseline is not the frozen 1,280-D LinearSVC");
  }
}

function requireRandomForestMetadata(metadata) {
  if (
    metadata.model_type !== "random_forest" ||
    metadata.classifier !== "RandomForestClassifier" ||
    metadata.feature_dim !== 2424 ||
    metadata.feature_set !== "handcrafted" ||
    metadata.estimator_parameters?.["classifier__n_estimators"] !== 300 ||
    JSON.stringify(metadata.feature_components) !== JSON.stringify(["color", "texture", "hog"])
  ) {
    throw new Error("Stage 4 comparator metadata is not the exact 2,424-D, 300-tree contract");
  }
}

function requireBranchPreprocessingContract(traditionalConfig, bovwConfig) {
  if (
    traditionalConfig.feature_dim !== 2424 ||
    JSON.stringify(traditionalConfig.image_size) !== JSON.stringify([128, 128]) ||
    traditionalConfig.resize?.mode !== "direct_square" ||
    traditionalConfig.resize?.colour_mode !== "RGB" ||
    traditionalConfig.resize?.exif_transpose !== true
  ) {
    throw new Error("Handcrafted preprocessing is not the direct 128 x 128 RGB contract");
  }
  if (
    bovwConfig.feature_dim !== 1280 ||
    bovwConfig.configuration?.max_side !== 256 ||
    bovwConfig.components?.bovw?.parameters?.max_side !== 256 ||
    bovwConfig.resize_policy !== "preserve aspect ratio; longest side equals max_side" ||
    bovwConfig.configuration?.soft_assign_k !== 3 ||
    bovwConfig.configuration?.vocabulary_size !== 256
  ) {
    throw new Error("SIFT-BoVW preprocessing is not the aspect-preserving max-side-256 contract");
  }
}

async function requireRefreshedFigures(figureManifest) {
  if (!String(figureManifest.operation || "").includes("no model inference")) {
    throw new Error("Final figure manifest does not declare persisted-artifact-only composition");
  }
  const sourceIndex = new Map(
    (figureManifest.sources || []).map((entry) => [String(entry.path || ""), entry]),
  );
  const outputIndex = new Map(
    (figureManifest.outputs || []).map((entry) => [String(entry.file || ""), entry]),
  );
  if (outputIndex.size !== (figureManifest.outputs || []).length || outputIndex.has("")) {
    throw new Error("Final figure manifest contains duplicate or empty output names");
  }
  if (sourceIndex.size !== (figureManifest.sources || []).length || sourceIndex.has("")) {
    throw new Error("Final figure manifest contains duplicate or empty source paths");
  }
  for (const [relativePath, entry] of sourceIndex) {
    if (
      !/^[0-9a-f]{64}$/.test(String(entry.sha256 || "")) ||
      entry.sha256 !== await sha256Path(relativePath)
    ) {
      throw new Error(`Final figure manifest is stale for source ${relativePath}`);
    }
  }
  for (const fileName of [
    "dataset_examples.jpg",
    "method_overview.png",
    "final_model_comparison.png",
    "robustness_top1_curves.png",
    "robustness_curves.png",
    "corruption_examples.jpg",
    "gradcam_correct_examples.jpg",
    "gradcam_incorrect_examples.jpg",
  ]) {
    const entry = outputIndex.get(fileName);
    if (!entry || !/^[0-9a-f]{64}$/.test(String(entry.sha256 || ""))) {
      throw new Error(`Final figure manifest does not bind ${fileName}`);
    }
    const sourceRelative = `results/final_report_figures/${fileName}`;
    const sourceHash = await sha256Path(sourceRelative);
    if (sourceHash !== entry.sha256) {
      throw new Error(`Final report figure does not match its manifest: ${fileName}`);
    }
    const presentationHash = sha256(await requiredPresentationAssetBytes(fileName));
    if (presentationHash !== sourceHash) {
      throw new Error(`Presentation asset has not been refreshed from ${sourceRelative}`);
    }
  }
  for (const [fileName, sourceRelative] of Object.entries({
    "gradcam_correct_2713957.png":
      "results/figures/gradcam/correct_high_confidence/2713957__row-4863__target-predicted-label-486_gradcam.png",
    "gradcam_incorrect_2759395.png":
      "results/figures/gradcam/incorrect_high_confidence/2759395__row-4855__target-predicted-label-486_gradcam.png",
  })) {
    const sourceEntry = sourceIndex.get(sourceRelative);
    const sourceHash = await sha256Path(sourceRelative);
    if (!sourceEntry || sourceEntry.sha256 !== sourceHash) {
      throw new Error(`Final figure manifest does not bind the selected Grad-CAM source ${sourceRelative}`);
    }
    if (sha256(await requiredPresentationAssetBytes(fileName)) !== sourceHash) {
      throw new Error(`Presentation asset has not been refreshed from ${sourceRelative}`);
    }
  }
}

async function loadPresentationEvidence() {
  const [
    stage2Freeze,
    stage2Selection,
    stage4Protocol,
    reportTablesManifest,
    figureManifest,
    randomForestMetadata,
    traditionalFeatureConfig,
    bovwFeatureConfig,
    gradcamSelection,
    gradcamManifest,
    mainText,
    ablationText,
    robustnessText,
    rankingText,
    stage4SummaryText,
  ] = await Promise.all([
    requiredJson(EVIDENCE.stage2Freeze),
    requiredJson(EVIDENCE.stage2Selection),
    requiredJson(EVIDENCE.stage4Protocol),
    requiredJson(EVIDENCE.reportTablesManifest),
    requiredJson(EVIDENCE.figureManifest),
    requiredJson(EVIDENCE.randomForestMetadata),
    requiredJson(EVIDENCE.traditionalFeatureConfig),
    requiredJson(EVIDENCE.bovwFeatureConfig),
    requiredJson(EVIDENCE.gradcamSelection),
    requiredJson(EVIDENCE.gradcamManifest),
    requiredText(EVIDENCE.mainResults),
    requiredText(EVIDENCE.ablationResults),
    requiredText(EVIDENCE.robustnessSummary),
    requiredText(EVIDENCE.robustnessRanking),
    requiredText(EVIDENCE.stage4Summary),
  ]);

  requireStage2Contract(stage2Freeze);
  requireStage2Contract(stage2Selection);
  requireBranchPreprocessingContract(traditionalFeatureConfig, bovwFeatureConfig);
  const selectionReference = stage2Freeze.selection_manifest;
  const actualSelectionHash = await sha256Path(EVIDENCE.stage2Selection);
  if (
    selectionReference?.path !== EVIDENCE.stage2Selection ||
    selectionReference.sha256 !== actualSelectionHash
  ) {
    throw new Error("Stage 2 freeze does not bind the current selection manifest");
  }
  await Promise.all([
    ...Object.entries(stage2Freeze.feature_configs || {}).map(([modelName, reference]) =>
      requireArtifactReference(reference, `Stage 2 ${modelName} feature config`),
    ),
    requireArtifactReference(stage2Freeze.validation_grid, "Stage 2 validation grid"),
  ]);

  if (
    reportTablesManifest.policy?.frozen_evidence_only !== true ||
    reportTablesManifest.policy?.training_or_inference_performed !== false ||
    reportTablesManifest.row_counts?.main_results !== 6 ||
    reportTablesManifest.row_counts?.ablation_results !== 8 ||
    reportTablesManifest.row_counts?.robustness_summary !== 4
  ) {
    throw new Error("Final report tables manifest is incomplete or not frozen-evidence-only");
  }
  await Promise.all([
    requireHashBinding(reportTablesManifest, "outputs", EVIDENCE.mainResults),
    requireHashBinding(reportTablesManifest, "outputs", EVIDENCE.ablationResults),
    requireHashBinding(reportTablesManifest, "outputs", EVIDENCE.robustnessSummary),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.finalModelComparison),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.stage2FeatureAblation),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.stage3Ablation),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.stage4Summary),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.robustnessRanking),
    requireHashBinding(reportTablesManifest, "sources", EVIDENCE.robustnessDropSummary),
  ]);

  requireRandomForestMetadata(randomForestMetadata);
  for (const [corruptionName, expected] of Object.entries(CORRUPTION_SCHEDULE)) {
    const observed = stage4Protocol.corruptions?.[corruptionName];
    if (
      observed?.parameter !== expected.parameter ||
      JSON.stringify(observed?.severities) !== JSON.stringify(expected.severities)
    ) {
      throw new Error(`Stage 4 protocol has the wrong severity schedule for ${corruptionName}`);
    }
  }
  if (
    stage4Protocol.status !== "frozen" ||
    stage4Protocol.stage !== "stage4_robustness" ||
    stage4Protocol.split !== "test" ||
    stage4Protocol.seed !== 42 ||
    stage4Protocol.expected_num_images_per_run !== 5000 ||
    stage4Protocol.expected_run_count !== 64 ||
    JSON.stringify(stage4Protocol.model_order) !== JSON.stringify(ROBUSTNESS_MODELS) ||
    JSON.stringify(Object.keys(stage4Protocol.corruptions || {}).sort()) !==
      JSON.stringify(Object.keys(CORRUPTION_SCHEDULE).sort()) ||
    stage4Protocol.policy?.no_retraining_or_weight_updates !== true ||
    stage4Protocol.policy?.no_test_tuning !== true
  ) {
    throw new Error("Stage 4 protocol is not the frozen 4-model x 4-corruption x 4-severity run");
  }
  const stage4Forest = stage4Protocol.models?.traditional_random_forest;
  const protocolModelChecks = [];
  for (const modelName of ROBUSTNESS_MODELS) {
    const model = stage4Protocol.models?.[modelName];
    protocolModelChecks.push(
      requireArtifactReference(model?.artifact, `Stage 4 ${modelName} artifact`),
      requireArtifactReference(model?.clean_metrics, `Stage 4 ${modelName} clean metrics`),
      requireArtifactReference(model?.clean_predictions, `Stage 4 ${modelName} clean predictions`),
    );
    if (model?.resolved_config) {
      protocolModelChecks.push(
        requireArtifactReference(model.resolved_config, `Stage 4 ${modelName} resolved config`),
      );
    }
    if (model?.source_config) {
      protocolModelChecks.push(
        requireArtifactReference(model.source_config, `Stage 4 ${modelName} source config`),
      );
    }
  }
  protocolModelChecks.push(
    requireArtifactReference(stage4Forest?.feature_config, "Stage 4 Random Forest feature config"),
    requireArtifactReference(stage4Forest?.model_metadata, "Stage 4 Random Forest metadata"),
  );
  await Promise.all(protocolModelChecks);
  const stage2ProtocolReference = stage4Protocol.stage2_evidence?.[EVIDENCE.stage2Freeze];
  if (
    !stage2ProtocolReference ||
    stage2ProtocolReference.sha256 !== await sha256Path(EVIDENCE.stage2Freeze)
  ) {
    throw new Error("Stage 4 protocol does not bind the current rewritten Stage 2 freeze");
  }
  await Promise.all([
    requireArtifactReference(
      stage4Protocol.stage2_evidence?.[EVIDENCE.stage2FeatureAblation],
      "Stage 4 rewritten Stage 2 ablation table",
    ),
    requireArtifactReference(
      stage4Protocol.stage2_evidence?.[EVIDENCE.stage2TraditionalResults],
      "Stage 4 rewritten Stage 2 result table",
    ),
    ...Object.entries(stage4Protocol.stage3_evidence || {}).map(([relativePath, reference]) =>
      requireArtifactReference(reference, `Stage 4 Stage 3 evidence ${relativePath}`),
    ),
  ]);

  const mainRows = parseCsv(mainText, EVIDENCE.mainResults);
  const ablationRows = parseCsv(ablationText, EVIDENCE.ablationResults);
  const robustnessRows = parseCsv(robustnessText, EVIDENCE.robustnessSummary);
  const rankingRows = parseCsv(rankingText, EVIDENCE.robustnessRanking);
  const stage4SummaryRows = parseCsv(stage4SummaryText, EVIDENCE.stage4Summary);
  const main = exactIndex(mainRows, "method_name", MAIN_METHODS, EVIDENCE.mainResults);
  const ablations = exactIndex(
    ablationRows,
    "experiment_name",
    ABLATION_EXPERIMENTS,
    EVIDENCE.ablationResults,
  );
  const ranking = exactIndex(
    rankingRows,
    "model_name",
    ROBUSTNESS_MODELS,
    EVIDENCE.robustnessRanking,
  );
  const finalRobustness = exactIndex(
    robustnessRows,
    "model_name",
    ROBUSTNESS_REPORT_NAMES,
    EVIDENCE.robustnessSummary,
  );
  const stage4Summary = exactIndex(
    stage4SummaryRows,
    "model_name",
    ROBUSTNESS_MODELS,
    EVIDENCE.stage4Summary,
  );

  for (const [name, row] of main) {
    for (const metric of ["test_top1", "test_top5", "test_macro_precision", "test_macro_recall", "test_macro_f1"]) {
      finiteMetric(row, metric, name);
    }
  }
  for (const [name, row] of ablations) {
    for (const metric of ["val_top1", "val_top5", "val_macro_f1"]) {
      finiteMetric(row, metric, name);
    }
  }
  const observedRobustnessRanks = [];
  const protocolHash = await sha256Path(EVIDENCE.stage4Protocol);
  for (const [name, row] of ranking) {
    finiteMetric(row, "mean_corrupted_top1", name);
    finiteMetric(row, "worst_condition_top1", name);
    const rank = finiteInteger(row.robustness_rank, `${name}.robustness_rank`);
    if (rank < 1 || rank > 4) throw new Error(`Invalid robustness rank for ${name}`);
    const worstSeverity = finiteInteger(row.worst_severity, `${name}.worst_severity`);
    if (
      !Object.hasOwn(CORRUPTION_SCHEDULE, row.worst_corruption_type) ||
      worstSeverity < 1 ||
      worstSeverity > 4 ||
      finiteInteger(row.num_corrupted_conditions, `${name}.num_corrupted_conditions`) !== 16
    ) {
      throw new Error(`Invalid robustness coverage or worst-condition label for ${name}`);
    }
    observedRobustnessRanks.push(rank);
    const summaryRow = stage4Summary.get(name);
    const cleanMainRow = main.get(ROBUSTNESS_TO_MAIN_METHOD[name]);
    const finalSummaryRow = finalRobustness.get(ROBUSTNESS_TO_MAIN_METHOD[name]);
    if (
      summaryRow.protocol_sha256 !== protocolHash ||
      finiteInteger(summaryRow.robustness_runs, `${name}.robustness_runs`) !== 16 ||
      finiteInteger(summaryRow.robustness_rank, `${name}.summary_robustness_rank`) !== rank ||
      summaryRow.worst_corruption_type !== row.worst_corruption_type ||
      finiteInteger(summaryRow.worst_severity, `${name}.summary_worst_severity`) !==
        worstSeverity ||
      Math.abs(
        finiteMetric(summaryRow, "average_corrupted_top1", name) -
          finiteMetric(row, "mean_corrupted_top1", name),
      ) > 1e-12 ||
      Math.abs(
        finiteMetric(summaryRow, "worst_corruption_top1", name) -
          finiteMetric(row, "worst_condition_top1", name),
      ) > 1e-12 ||
      Math.abs(
        finiteMetric(summaryRow, "clean_test_top1", name) -
          finiteMetric(cleanMainRow, "test_top1", ROBUSTNESS_TO_MAIN_METHOD[name]),
      ) > 1e-12 ||
      finiteInteger(
        finalSummaryRow.robustness_rank,
        `${ROBUSTNESS_TO_MAIN_METHOD[name]}.final_robustness_rank`,
      ) !== rank
    ) {
      throw new Error(`Stage 4 summary is not bound to the current protocol/ranking for ${name}`);
    }
  }
  if (JSON.stringify(observedRobustnessRanks.sort()) !== JSON.stringify([1, 2, 3, 4])) {
    throw new Error("Stage 4 robustness ranking is not a unique 1-through-4 ordering");
  }
  const primary = main.get("Handcrafted + BoVW (LinearSVC)");
  const forest = main.get("Random Forest");
  const bovw = main.get("SIFT-BoVW (LinearSVC)");
  if (
    primary.architecture_or_classifier !== "LinearSVC" ||
    forest.architecture_or_classifier !== "RandomForestClassifier" ||
    bovw.architecture_or_classifier !== "LinearSVC"
  ) {
    throw new Error("Final main-results table contains a legacy traditional estimator identity");
  }

  const selectedExamplesHash = await sha256Path(EVIDENCE.gradcamSelectedExamples);
  const selectionHash = await sha256Path(EVIDENCE.gradcamSelection);
  if (
    gradcamSelection.status !== "selected" ||
    gradcamSelection.analysis_experiment !== "efficientnet_b0_pretrained_strong_aug" ||
    gradcamSelection.total_selected !== 33 ||
    gradcamSelection.counts?.correct_high_confidence !== 12 ||
    gradcamSelection.counts?.incorrect_high_confidence !== 12 ||
    gradcamSelection.counts?.confused_pair !== 9 ||
    gradcamSelection.outputs?.selected_examples?.path !== EVIDENCE.gradcamSelectedExamples ||
    gradcamSelection.outputs.selected_examples.sha256 !== selectedExamplesHash
  ) {
    throw new Error("Grad-CAM selection manifest is not the frozen 33-image contract");
  }
  if (
    gradcamManifest.status !== "complete" ||
    gradcamManifest.analysis_experiment !== gradcamSelection.analysis_experiment ||
    gradcamManifest.parameter_updates !== false ||
    gradcamManifest.model_eval_mode !== true ||
    gradcamManifest.source_selection_total !== gradcamSelection.total_selected ||
    gradcamManifest.processed_examples !== gradcamSelection.total_selected ||
    gradcamManifest.selection_manifest_sha256 !== selectionHash ||
    gradcamManifest.cam_counts?.predicted !== 33 ||
    gradcamManifest.cam_counts?.true !== 21 ||
    gradcamManifest.cam_counts?.total !== 54 ||
    gradcamManifest.figures?.length !== gradcamManifest.cam_counts.total ||
    gradcamManifest.method?.target_layer !== "features.8"
  ) {
    throw new Error("Grad-CAM manifest is not the frozen 54-map contract");
  }

  const demoCaptureText = (
    await requiredPresentationAssetBytes("demo_validation_output.txt")
  ).toString("utf8");
  const demoRuntimeMatch = demoCaptureText.match(/Ran 2 tests in ([0-9]+(?:\.[0-9]+)?)s\s+\n\s*OK\s*$/m);
  if (
    !demoRuntimeMatch ||
    !demoCaptureText.includes("test_every_contracted_hash_matches") ||
    !demoCaptureText.includes("test_all_nested_manifest_bindings_match_and_deduplicate")
  ) {
    throw new Error("Demo validation capture is missing the exact two-test PASS output");
  }

  await requireRefreshedFigures(figureManifest);
  return {
    mainRows,
    ablationRows,
    rankingRows,
    main,
    ablations,
    ranking,
    demoRuntimeSeconds: demoRuntimeMatch[1],
    gradcamSelection,
    gradcamManifest,
  };
}

const presentationEvidence = await loadPresentationEvidence();

async function locateArtifactTool() {
  const resolver = createRequire(path.join(WORKSPACE, "__artifact_resolver__.cjs"));
  try {
    return resolver.resolve("@oai/artifact-tool");
  } catch {
    const explicit = process.env.CODEX_ARTIFACT_TOOL_ENTRY;
    if (explicit) {
      const candidate = path.resolve(explicit);
      try {
        const details = await fs.stat(candidate);
        if (details.isFile()) return candidate;
      } catch {
        // Report the invalid explicit override below.
      }
      throw new Error(
        `CODEX_ARTIFACT_TOOL_ENTRY is set but does not point to a readable ` +
          `artifact-tool entry file: ${candidate}. Correct the override or unset ` +
          `it to use @oai/artifact-tool from the selected workspace.`,
      );
    }
    throw new Error(
      `Could not resolve @oai/artifact-tool from the selected workspace ` +
        `(${WORKSPACE}). Install or prepare the dependency there, or set ` +
        `CODEX_ARTIFACT_TOOL_ENTRY to its artifact_tool.mjs entry file.`,
    );
  }
}

const artifactEntry = await locateArtifactTool();
const { Presentation, PresentationFile } = await import(
  pathToFileURL(artifactEntry).href
);

const W = 1280;
const H = 720;
const FONT = "Arial";
const MONO = "Menlo";
const C = {
  white: "#FFFFFF",
  ink: "#0B1F33",
  muted: "#536473",
  pale: "#EDF2F5",
  panel: "#F5F7F8",
  rule: "#C8D2D9",
  blue: "#2F80ED",
  cyan: "#5BC0EB",
  green: "#2E9B75",
  amber: "#E6A23C",
  orange: "#D9772B",
  red: "#C94C5A",
  purple: "#7357C7",
  traditional: "#E5A646",
  deep: "#2F80ED",
};

function percent(value, digits = 2) {
  return `${(100 * Number(value)).toFixed(digits)}%`;
}

function spokenPercent(value, digits = 2) {
  return `${(100 * Number(value)).toFixed(digits)} percent`;
}

function decimal(value, digits = 4) {
  return Number(value).toFixed(digits);
}

function percentagePoints(left, right, digits = 2) {
  return (100 * (Number(left) - Number(right))).toFixed(digits);
}

function wordsLabel(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

const primaryResult = presentationEvidence.main.get("Handcrafted + BoVW (LinearSVC)");
const forestResult = presentationEvidence.main.get("Random Forest");
const bovwResult = presentationEvidence.main.get("SIFT-BoVW (LinearSVC)");
const scratchResult = presentationEvidence.main.get("ResNet18 (scratch)");
const pretrainedResult = presentationEvidence.main.get("ResNet18 (ImageNet pretrained)");
const efficientResult = presentationEvidence.main.get("EfficientNet-B0 (ImageNet pretrained)");

const fullAblation = presentationEvidence.ablations.get("full");
const colourAblation = presentationEvidence.ablations.get("ablation_only_color");
const textureAblation = presentationEvidence.ablations.get("ablation_only_texture");
const hogAblation = presentationEvidence.ablations.get("ablation_only_hog");
const scratchAblation = presentationEvidence.ablations.get("resnet18_scratch_basic");
const basicAblation = presentationEvidence.ablations.get("resnet18_pretrained_basic");
const strongAblation = presentationEvidence.ablations.get("resnet18_pretrained_strong_aug");
const frozenAblation = presentationEvidence.ablations.get("resnet18_pretrained_frozen_head");

const forestRobustness = presentationEvidence.ranking.get("traditional_random_forest");
const scratchRobustness = presentationEvidence.ranking.get("resnet18_scratch_basic");
const pretrainedRobustness = presentationEvidence.ranking.get(
  "resnet18_pretrained_strong_aug",
);
const efficientRobustness = presentationEvidence.ranking.get(
  "efficientnet_b0_pretrained_strong_aug",
);
const robustnessRowsByRank = [...presentationEvidence.rankingRows].sort(
  (left, right) => Number(left.robustness_rank) - Number(right.robustness_rank),
);
const strongestRobustness = robustnessRowsByRank[0];
const robustnessDisplayNames = {
  traditional_random_forest: "Random Forest",
  resnet18_scratch_basic: "scratch ResNet18",
  resnet18_pretrained_strong_aug: "pretrained ResNet18",
  efficientnet_b0_pretrained_strong_aug: "EfficientNet-B0",
};
const cleanResultsByRobustnessModel = {
  traditional_random_forest: forestResult,
  resnet18_scratch_basic: scratchResult,
  resnet18_pretrained_strong_aug: pretrainedResult,
  efficientnet_b0_pretrained_strong_aug: efficientResult,
};
const strongestRobustnessClean = cleanResultsByRobustnessModel[strongestRobustness.model_name];
const strongestHeldOut = [...presentationEvidence.mainRows].sort(
  (left, right) => Number(right.test_top1) - Number(left.test_top1),
)[0];

const scripts = {
  p1: `Hello, I'm Yuan Cao, and I'll introduce our project, Fine-Grained Species Recognition at Scale. We compare classical features with deep transfer learning on a controlled 500-species task.

Species recognition is a fine-grained problem. Different species may differ by a small wing marking, leaf shape, or plumage cue, while the same species changes with pose, life stage, background, scale, lighting, and occlusion. With 500 labels, a model must choose among many visually similar alternatives. This matters for biodiversity monitoring and community-science collections, where manual identification does not scale.

The literature moves from fixed colour and texture summaries, through local SIFT descriptors and bag-of-visual-words, to margin classifiers and random forests. CNNs then learn features end to end; ResNet makes deep optimisation practical, ImageNet supports transfer, and EfficientNet scales capacity efficiently. Robustness benchmarks and Grad-CAM extend evaluation beyond a clean accuracy number.

We use iNaturalist 2021 Mini. Seed 42 selects 500 classes deterministically. For every class, 40 train and 10 internal-validation images come from train_mini, while 10 official-validation images form the held-out test set: 20,000, 5,000, and 5,000 images. The manifests are disjoint. Stage 3 freeze-before-test is verified; historical Stage 2 pre-test locking is not independently established. Our questions are: how far classical vision goes; how much a scratch CNN improves; what ImageNet transfer adds descriptively; and how robust and interpretable the final models are. P2 will now explain the classical baselines.`,
  p2: `Hello, I'm Kaishi Yang, and I'll cover the traditional pipeline. Every image receives deterministic EXIF correction and RGB conversion. The handcrafted branch resizes directly to 128 by 128, while the SIFT-BoVW branch uses grayscale, aspect-preserving resizing with maximum side 256. We extract spatial colour, multi-scale texture, HOG shape, and train-fitted local SIFT-BoVW evidence. Validation controls model and ablation choices; the current Stage 2 manifest binds that selection and the resulting held-out evidence.

The handcrafted descriptor has 2,424 dimensions: 520 from spatial joint H-S and V histograms, 140 from two uniform-LBP scales, and 1,764 from HOG. A 256-word SIFT vocabulary uses top-three soft assignment over a whole-image plus two-by-two spatial pyramid, adding 1,280 dimensions. The primary 3,704-dimensional pipeline applies block-wise power normalisation with validation-selected alpha 0.3, then fits LinearSVC. The robustness comparator is a 300-tree Random Forest over handcrafted features only.

The vocabulary is fitted from training images only: four deterministic images per class and at most 100 descriptors per image. The current Stage 2 manifest binds validation selection across exactly 19 configurations; historical pre-test locking is not independently established. Test descriptors never fit a vocabulary, normaliser, or configuration.

Validation-only ablations isolate colour, texture, and HOG; the complete four-block descriptor is the selected reference. We report top-1, top-5, and macro precision, recall, and F1. These baselines test fixed cues, local invariance, and linear versus nonlinear classifiers under strict leakage controls. P3 will show what learned representations change.`,
  p3: `Hello, I'm Aditya Winata, and I'll cover deep learning. The deep pipeline starts with augmentation, passes a 224-pixel crop through a CNN, and replaces the original classifier with a 500-class head. Validation macro F1 selects the best checkpoint; top-1, top-5, loss, and earlier epoch break ties. Once the selected model set is frozen, the held-out test is evaluated without further tuning.

We compare ResNet18 trained from random weights with ImageNet-pretrained ResNet18 and pretrained EfficientNet-B0. A frozen-backbone ResNet18 ablation tests how useful pretrained features are before adaptation, while the main transfer runs fine-tune all layers.

Basic augmentation uses random crop and horizontal flip. Strong augmentation widens the crop range and adds moderate colour jitter, affine rotation, and random erasing, while avoiding vertical flips and extreme crops. In the matched ResNet18 comparison, strong augmentation changes validation top-1 from ${spokenPercent(basicAblation.val_top1)} to ${spokenPercent(strongAblation.val_top1)}.

Transfer is especially valuable here because each class offers only 40 training images. ImageNet supplies reusable edges, textures, and parts before task-specific fine-tuning. However, scratch and pretrained schedules differ, so their gap is descriptive rather than a controlled one-factor estimate. Exact learning rates, epoch limits, optimisers, and parameter counts are in the appendix. This makes the selection logic explicit without treating validation as a second test set. The test set was accessed only after checkpoint and configuration hashes were frozen. P4 will now present the held-out evidence and its limits.`,
  p4: `Hello, I'm Jialin Guo, and I'll summarise the results. The rewritten primary traditional comparison is the 3,704-dimensional LinearSVC, which reaches ${spokenPercent(primaryResult.test_top1)} top-1, ${spokenPercent(primaryResult.test_top5)} top-5, and ${spokenPercent(primaryResult.test_macro_f1)} macro F1. Scratch ResNet18 reaches ${spokenPercent(scratchResult.test_top1)} top-1, selected pretrained ResNet18 reaches ${spokenPercent(pretrainedResult.test_top1)}, and ${strongestHeldOut.method_name} is strongest at ${spokenPercent(strongestHeldOut.test_top1)} top-1. The observed ${percentagePoints(efficientResult.test_top1, pretrainedResult.test_top1)}-point EfficientNet-versus-pretrained-ResNet18 difference is descriptive; with one seed and no confidence intervals, we do not claim statistical significance.

Robustness uses four frozen models across Gaussian noise, Gaussian blur, JPEG compression, and motion blur, each at four severities. ${robustnessDisplayNames[strongestRobustness.model_name]} ranks first on the regenerated robustness evidence at ${spokenPercent(strongestRobustness.mean_corrupted_top1)} mean corrupted top-1. Its worst condition is ${wordsLabel(strongestRobustness.worst_corruption_type)} severity ${strongestRobustness.worst_severity}, where top-1 falls to ${spokenPercent(strongestRobustness.worst_condition_top1)}. Clean accuracy is therefore not a reliability guarantee.

Grad-CAM adds qualitative inspection. Correct cases often focus on the organism, while the matrimony-vine error attends to visually similar red fruit and predicts Jerusalem cherry with high confidence. These maps are class-conditional and useful for forming hypotheses, but they are coarse, non-causal, and not segmentations.

The main conclusion is that the frozen tables support a transparent comparison across traditional, scratch, and transfer-learning systems, while the regenerated Stage 4 tables expose reliability under corruption. In this evidence snapshot, ${strongestHeldOut.method_name} leads clean held-out top-1 and ${robustnessDisplayNames[strongestRobustness.model_name]} leads mean corrupted top-1. Limitations are one principal seed, only ten test images per class, no repeated runs or confidence intervals, four synthetic corruption families, and qualitative Grad-CAM. P5 will close with reproducibility and the software demonstration.`,
  p5: `Hello, I'm Yuzhe Qiu. I'll finish with reproducibility and the demo. Seed 42 deterministically builds immutable train, validation, and test manifests. The Stage 2 selection and Stage 3 freeze files bind the approved configurations and checkpoints; final table and figure manifests bind metric sources and repository outputs. Before recording, the generator verifies declared hashes, exact row sets, estimator identities, Stage 4 protocol linkage, and copied presentation-figure bytes. The CVPR report and deterministic source ZIP exclude data, checkpoints, caches, and predictions.

Here is the repository, organised into source code, tests, reports, and persisted results. I'll run one read-only validation: these tests recompute artifact hashes and nested manifest bindings. The two checks must pass against the refreshed evidence contract before recording. The table and comparison figure then show the six frozen held-out methods, led by ${strongestHeldOut.method_name} at ${spokenPercent(strongestHeldOut.test_top1)} top-1. Our robustness artifact covers four frozen models, four corruption types, and four severities; its ranking and worst condition come directly from the regenerated Stage 4 tables. Finally, this correct Grad-CAM attends to the organism, whereas the confident error focuses on visually similar fruit. These maps are qualitative, not causal. Nothing here retrains a model or reruns inference.

Together, the project provides a transparent traditional reference, a held-out comparison led by ${strongestHeldOut.method_name}, and regenerated robustness evidence alongside qualitative Grad-CAM diagnostics. Thank you; we welcome your questions.`,
};

function addShape(slide, {
  left,
  top,
  width,
  height,
  fill = "none",
  lineFill = "none",
  lineWidth = 0,
  geometry = "rect",
  name,
}) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addText(slide, textValue, {
  left,
  top,
  width,
  height,
  fontSize = 24,
  color = C.ink,
  bold = false,
  alignment = "left",
  verticalAlignment = "top",
  typeface = FONT,
  fill = "none",
  lineFill = "none",
  lineWidth = 0,
  name,
}) {
  const box = addShape(slide, {
    left,
    top,
    width,
    height,
    fill,
    lineFill,
    lineWidth,
    geometry: "textbox",
    name,
  });
  box.text = String(textValue);
  box.text.style = {
    fontSize,
    color,
    bold,
    alignment,
    verticalAlignment,
    typeface,
  };
  return box;
}

function addRule(slide, left, top, width, color = C.rule, thickness = 1) {
  addShape(slide, {
    left,
    top,
    width,
    height: thickness,
    fill: color,
    lineFill: color,
    lineWidth: 0,
  });
}

function addBulletList(slide, items, {
  left,
  top,
  width,
  fontSize = 22,
  lineHeight = 44,
  color = C.ink,
  bulletColor = C.blue,
}) {
  items.forEach((item, index) => {
    const y = top + index * lineHeight;
    addShape(slide, {
      left,
      top: y + 9,
      width: 9,
      height: 9,
      geometry: "ellipse",
      fill: bulletColor,
    });
    addText(slide, item, {
      left: left + 20,
      top: y,
      width: width - 20,
      height: lineHeight,
      fontSize,
      color,
    });
  });
}

function addMetric(slide, value, label, {
  left,
  top,
  width,
  accent = C.blue,
  valueSize = 44,
}) {
  addText(slide, value, {
    left,
    top,
    width,
    height: 55,
    fontSize: valueSize,
    color: accent,
    bold: true,
  });
  addText(slide, label, {
    left,
    top: top + 55,
    width,
    height: 42,
    fontSize: 17,
    color: C.muted,
  });
}

function addHeader(slide, title, presenter, section, slideNumber) {
  slide.background.fill = C.white;
  addText(slide, title, {
    left: 44,
    top: 50,
    width: 1140,
    height: 64,
    fontSize: 40,
    color: C.ink,
    bold: true,
    name: `slide-${slideNumber}-title`,
  });
  addRule(slide, 44, 118, 1192, C.rule, 1);
  addText(slide, String(slideNumber), {
    left: 1190,
    top: 682,
    width: 46,
    height: 18,
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
  // Keep presenter and section in one text object. LibreOffice intermittently
  // drops a short standalone object at this coordinate during PPTX import, while
  // the combined eyebrow renders consistently in both supported render paths.
  addText(slide, `${section.toUpperCase()}   ·   ${presenter}`, {
    left: 96,
    top: 24,
    width: 600,
    height: 22,
    fontSize: 13,
    color: C.blue,
    bold: true,
  });
}

function addFooter(slide, sourceText, presenter = "") {
  addText(slide, sourceText, {
    left: 44,
    top: 681,
    width: 1140,
    height: 17,
    fontSize: sourceText.length > 110 ? 8.5 : sourceText.length > 85 ? 9.5 : 10,
    color: C.muted,
    alignment: "left",
  });
}

function setNotes(slide, narration, sources) {
  const sourceBlock = sources.map((source) => `- ${source}`).join("\n");
  slide.speakerNotes.textFrame.setText(
    `${narration}\n\n[Sources]\n${sourceBlock}`,
  );
  slide.speakerNotes.setVisible(true);
}

async function readAsset(name) {
  const file = path.join(ASSET_DIR, name);
  const bytes = await fs.readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function addImage(slide, name, {
  left,
  top,
  width,
  height,
  fit = "contain",
  crop,
  alt,
  geometry = "rect",
  borderRadius,
}) {
  const ext = path.extname(name).toLowerCase();
  const contentType = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
  return slide.images.add({
    blob: await readAsset(name),
    contentType,
    alt,
    fit,
    crop,
    geometry,
    borderRadius,
    position: { left, top, width, height },
  });
}

function addManualTable(slide, {
  left,
  top,
  widths,
  rowHeight,
  rows,
  headerFill = C.ink,
  headerColor = C.white,
  fontSize = 17,
  highlightRows = {},
  alignments = [],
}) {
  const tableWidth = widths.reduce((sum, value) => sum + value, 0);
  rows.forEach((row, rowIndex) => {
    const y = top + rowIndex * rowHeight;
    const fill = rowIndex === 0
      ? headerFill
      : (highlightRows[rowIndex] || (rowIndex % 2 === 0 ? C.panel : C.white));
    addShape(slide, {
      left,
      top: y,
      width: tableWidth,
      height: rowHeight,
      fill,
      lineFill: C.rule,
      lineWidth: 1,
    });
    let x = left;
    row.forEach((cell, columnIndex) => {
      addText(slide, cell, {
        left: x + 10,
        top: y + 8,
        width: widths[columnIndex] - 20,
        height: rowHeight - 16,
        fontSize,
        color: rowIndex === 0 ? headerColor : C.ink,
        bold: rowIndex === 0 || Boolean(highlightRows[rowIndex]),
        alignment: alignments[columnIndex] || (columnIndex === 0 ? "left" : "right"),
        verticalAlignment: "center",
      });
      x += widths[columnIndex];
      if (columnIndex < row.length - 1) {
        addRule(slide, x, y, 1, C.rule, rowHeight);
      }
    });
  });
}

function addFlow(slide, labels, {
  left,
  top,
  totalWidth,
  nodeHeight = 82,
  nodeFill = C.panel,
  accentIndex = -1,
  fontSize = 17,
}) {
  const gap = 24;
  const nodeWidth = (totalWidth - gap * (labels.length - 1)) / labels.length;
  // Connectors are created first so they remain behind every entity node.
  for (let index = 0; index < labels.length - 1; index += 1) {
    const arrowLeft = left + nodeWidth * (index + 1) + gap * index + 3;
    addShape(slide, {
      left: arrowLeft,
      top: top + nodeHeight / 2 - 9,
      width: gap - 6,
      height: 18,
      geometry: "rightArrow",
      fill: C.rule,
    });
  }
  labels.forEach((label, index) => {
    const x = left + index * (nodeWidth + gap);
    addShape(slide, {
      left: x,
      top,
      width: nodeWidth,
      height: nodeHeight,
      fill: index === accentIndex ? C.blue : nodeFill,
      lineFill: index === accentIndex ? C.blue : C.rule,
      lineWidth: 1,
    });
    addText(slide, label, {
      left: x + 8,
      top: top + 10,
      width: nodeWidth - 16,
      height: nodeHeight - 20,
      fontSize,
      color: index === accentIndex ? C.white : C.ink,
      bold: true,
      alignment: "center",
      verticalAlignment: "center",
    });
  });
}

function appendixHeader(slide, title, number) {
  slide.background.fill = C.white;
  addText(slide, `\u00A0\u00A0A${number - 12}  /  ${title}`, {
    left: 96,
    top: 50,
    width: 1088,
    height: 64,
    fontSize: 32,
    color: C.ink,
    bold: true,
  });
  addRule(slide, 44, 118, 1192, C.rule, 1);
  addText(slide, `${number} / A${number - 12}`, {
    left: 1160,
    top: 682,
    width: 76,
    height: 18,
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
}

const deck = Presentation.create({ slideSize: { width: W, height: H } });

// Slide 1 - title
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addShape(slide, { left: 0, top: 0, width: 18, height: H, fill: C.blue });
  addText(slide, "COMP9517 COMPUTER VISION", {
    left: 62,
    top: 54,
    width: 510,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  addText(slide, "Fine-Grained\nSpecies Recognition\nat Scale", {
    left: 62,
    top: 126,
    width: 570,
    height: 260,
    fontSize: 54,
    color: C.ink,
    bold: true,
  });
  addText(
    slide,
    "Traditional baselines  /  transfer learning  /  robustness  /  Grad-CAM",
    {
      left: 62,
      top: 420,
      width: 575,
      height: 72,
      fontSize: 22,
      color: C.muted,
    },
  );
  addText(slide, "Yuan Cao   /   Kaishi Yang   /   Aditya Winata\nJialin Guo   /   Yuzhe Qiu", {
    left: 62,
    top: 558,
    width: 560,
    height: 54,
    fontSize: 13,
    color: C.ink,
    bold: true,
  });
  addText(slide, "2026 Term 2", {
    left: 62,
    top: 616,
    width: 300,
    height: 28,
    fontSize: 16,
    color: C.muted,
  });
  addShape(slide, {
    left: 676,
    top: 44,
    width: 526,
    height: 572,
    fill: C.pale,
    lineFill: C.rule,
    lineWidth: 1,
  });
  await addImage(slide, "dataset_examples.jpg", {
    left: 690,
    top: 58,
    width: 498,
    height: 500,
    fit: "contain",
    alt: "Twelve labelled examples from the frozen train, validation, and test partitions.",
  });
  addText(slide, "Frozen 500-species subset  /  deterministic seed 42", {
    left: 690,
    top: 572,
    width: 498,
    height: 28,
    fontSize: 14,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, "1", {
    left: 1180,
    top: 682,
    width: 40,
    height: 18,
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
  setNotes(slide, scripts.p1.split("\n\n")[0], [
    "results/final_report_figures/dataset_examples.jpg",
    "data/splits/split_summary.json",
    "report/cvpr_latex/main.tex (canonical project title and verified roster)",
  ]);
}

// Slide 2 - challenge and literature
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Species recognition is fine-grained by design",
    "P1",
    "Problem and literature",
    2,
  );
  const challengeRows = [
    ["BETWEEN SPECIES", "A small marking, edge, or local structure can change the label."],
    ["WITHIN SPECIES", "Pose, life stage, scale, lighting, occlusion, and background all vary."],
    ["DECISION SPACE", "500 visually similar classes; only 40 training images per class."],
  ];
  challengeRows.forEach((row, index) => {
    const y = 156 + index * 126;
    addText(slide, row[0], {
      left: 52,
      top: y,
      width: 220,
      height: 26,
      fontSize: 15,
      color: index === 2 ? C.red : C.blue,
      bold: true,
    });
    addText(slide, row[1], {
      left: 52,
      top: y + 32,
      width: 450,
      height: 74,
      fontSize: 23,
      color: C.ink,
      bold: true,
    });
    if (index < challengeRows.length - 1) addRule(slide, 52, y + 112, 450);
  });
  addText(slide, "LITERATURE PROGRESSION", {
    left: 568,
    top: 156,
    width: 620,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  const progression = [
    ["1", "Colour + texture", "fixed summaries"],
    ["2", "SIFT + BoVW", "local invariance"],
    ["3", "Classical classifiers", "margin + ensembles"],
    ["4", "CNNs + transfer", "learned hierarchy"],
    ["5", "Robustness + Grad-CAM", "failure evidence"],
  ];
  progression.forEach((item, index) => {
    const y = 202 + index * 80;
    addText(slide, item[0], {
      left: 568,
      top: y,
      width: 38,
      height: 38,
      fontSize: 19,
      color: C.white,
      bold: true,
      alignment: "center",
      verticalAlignment: "center",
      fill: index === 4 ? C.blue : C.ink,
    });
    addText(slide, item[1], {
      left: 624,
      top: y - 2,
      width: 300,
      height: 30,
      fontSize: 22,
      color: C.ink,
      bold: true,
    });
    addText(slide, item[2], {
      left: 930,
      top: y,
      width: 250,
      height: 28,
      fontSize: 18,
      color: C.muted,
      alignment: "right",
    });
    if (index < progression.length - 1) addRule(slide, 587, y + 40, 1, C.rule, 40);
  });
  addFooter(
    slide,
    "Citations: Van Horn et al. (CVPR 2018, 2021); Lowe (2004); He et al. (2016); Tan & Le (2019); Selvaraju et al. (2017)",
    "P1",
  );
  setNotes(slide, scripts.p1.split("\n\n").slice(1, 3).join("\n\n"), [
    "report/cvpr_latex/sections/01_introduction.tex",
    "report/cvpr_latex/sections/02_literature_review.tex",
    "report/cvpr_latex/references.bib",
  ]);
}

// Slide 3 - dataset and research questions
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "A leakage-safe 40 / 10 / 10 protocol frames four questions",
    "P1",
    "Dataset and questions",
    3,
  );
  const splits = [
    { label: "TRAIN", count: "20,000", per: "40 / class", source: "train_mini", fill: C.ink },
    { label: "INTERNAL VAL", count: "5,000", per: "10 / class", source: "train_mini", fill: C.blue },
    { label: "HELD-OUT TEST", count: "5,000", per: "10 / class", source: "official val", fill: C.red },
  ];
  splits.forEach((item, index) => {
    const x = 52 + index * 184;
    addShape(slide, {
      left: x,
      top: 166,
      width: 164,
      height: 264,
      fill: index === 0 ? C.panel : C.white,
      lineFill: item.fill,
      lineWidth: 2,
    });
    addText(slide, item.label, {
      left: x + 12,
      top: 184,
      width: 140,
      height: 34,
      fontSize: 14,
      color: item.fill,
      bold: true,
      alignment: "center",
    });
    addText(slide, item.count, {
      left: x + 8,
      top: 236,
      width: 148,
      height: 62,
      fontSize: 37,
      color: C.ink,
      bold: true,
      alignment: "center",
    });
    addText(slide, item.per, {
      left: x + 8,
      top: 316,
      width: 148,
      height: 34,
      fontSize: 19,
      color: C.ink,
      bold: true,
      alignment: "center",
    });
    addText(slide, item.source, {
      left: x + 8,
      top: 372,
      width: 148,
      height: 32,
      fontSize: 16,
      color: C.muted,
      alignment: "center",
    });
  });
  addText(slide, "500 classes  /  seed 42  /  zero cross-split overlap", {
    left: 52,
    top: 454,
    width: 530,
    height: 40,
    fontSize: 20,
    color: C.green,
    bold: true,
    alignment: "center",
  });
  addText(slide, "RESEARCH QUESTIONS", {
    left: 640,
    top: 156,
    width: 530,
    height: 26,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  const questions = [
    "How far do traditional CV methods perform?",
    "How much does a scratch CNN improve?",
    "What observed gain comes with ImageNet transfer?",
    "How robust and interpretable are the final models?",
  ];
  questions.forEach((question, index) => {
    const y = 196 + index * 92;
    addText(slide, String(index + 1), {
      left: 640,
      top: y,
      width: 44,
      height: 44,
      fontSize: 22,
      color: C.white,
      bold: true,
      alignment: "center",
      verticalAlignment: "center",
      fill: index === 3 ? C.red : C.blue,
    });
    addText(slide, question, {
      left: 704,
      top: y - 2,
      width: 490,
      height: 58,
      fontSize: 23,
      color: C.ink,
      bold: true,
    });
    if (index < questions.length - 1) addRule(slide, 704, y + 68, 490);
  });
  addText(slide, "Current binding: validation selects. Stage 3 froze before test.", {
    left: 640,
    top: 582,
    width: 554,
    height: 44,
    fontSize: 22,
    color: C.red,
    bold: true,
  });
  addFooter(
    slide,
    "Sources: data/splits/split_summary.json; train.csv; val.csv; test.csv; report Dataset and Protocol",
    "P1",
  );
  setNotes(slide, scripts.p1.split("\n\n").slice(3).join("\n\n"), [
    "data/splits/split_summary.json",
    "data/splits/train.csv; data/splits/val.csv; data/splits/test.csv",
    "report/cvpr_latex/sections/03_dataset_protocol.tex",
  ]);
}

// Slide 4 - traditional pipeline
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Classical baselines expose the limits of fixed descriptors",
    "P2",
    "Traditional pipeline",
    4,
  );
  addFlow(
    slide,
    [
      "Input\nimage",
      "EXIF\ncorrection",
      "Branch\nresize",
      "Feature\nblocks",
      "Power\nnorm",
      "Classifier",
      "500-way\nscores",
      "Frozen test\nmetrics",
    ],
    {
      left: 50,
      top: 250,
      totalWidth: 1180,
      nodeHeight: 100,
      nodeFill: "#FFF8EC",
      accentIndex: 6,
      fontSize: 17,
    },
  );
  addText(slide, "TRAIN ONLY", {
    left: 355,
    top: 178,
    width: 340,
    height: 30,
    fontSize: 16,
    color: C.traditional,
    bold: true,
    alignment: "center",
  });
  addRule(slide, 380, 216, 292, C.traditional, 3);
  addText(slide, "Validation-only model selection", {
    left: 708,
    top: 178,
    width: 250,
    height: 32,
    fontSize: 16,
    color: C.blue,
    bold: true,
    alignment: "center",
  });
  addShape(slide, {
    left: 824,
    top: 214,
    width: 18,
    height: 28,
    geometry: "downArrow",
    fill: C.blue,
  });
  addText(slide, "Same frozen split, class mapping, and metric definitions as deep learning", {
    left: 190,
    top: 420,
    width: 900,
    height: 58,
    fontSize: 26,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addText(
    slide,
    "Handcrafted branch: RGB, direct 128 x 128  /  BoVW branch: grayscale SIFT, aspect-preserving max side 256",
    {
      left: 150,
      top: 474,
      width: 980,
      height: 34,
      fontSize: 18,
      color: C.traditional,
      bold: true,
      alignment: "center",
    },
  );
  addBulletList(
    slide,
    [
      "No validation or test descriptors fit the power normaliser or BoVW vocabulary.",
      "Current Stage 2 manifest binds exactly 19 validation configurations; historical pre-test locking is not independently established.",
    ],
    { left: 230, top: 526, width: 840, fontSize: 19, lineHeight: 43, bulletColor: C.traditional },
  );
  addFooter(
    slide,
    "Sources: report Methods; Stage 2 freeze, selection manifest, and validation grid",
    "P2",
  );
  setNotes(slide, scripts.p2.split("\n\n")[0], [
    "report/cvpr_latex/sections/04_methods.tex",
    "results/metrics/stage2_freeze.json",
    "results/metrics/stage2_selection_manifest.json",
    "results/tables/traditional_validation_grid.csv",
  ]);
}

// Slide 5 - features and classifiers
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Handcrafted features are meaningful - but low-capacity",
    "P2",
    "Features and classifiers",
    5,
  );
  addText(slide, "3,704-D PRIMARY COMBINED DESCRIPTOR", {
    left: 52,
    top: 150,
    width: 510,
    height: 28,
    fontSize: 15,
    color: C.traditional,
    bold: true,
  });
  const dims = [
    { name: "Colour", value: 520, fill: "#E96B65", width: 92 },
    { name: "Texture", value: 140, fill: "#7B61A8", width: 62 },
    { name: "HOG", value: 1764, fill: C.blue, width: 230 },
    { name: "BoVW", value: 1280, fill: C.traditional, width: 136 },
  ];
  let x = 52;
  const available = 520;
  dims.forEach((item, index) => {
    const width = item.width;
    addShape(slide, {
      left: x,
      top: 194,
      width,
      height: 66,
      fill: item.fill,
      lineFill: C.white,
      lineWidth: 1,
    });
    if (width >= 120) {
      addText(slide, `${item.name}  ${item.value}`, {
        left: x + 12,
        top: 213,
        width: width - 24,
        height: 28,
        fontSize: 20,
        color: C.white,
        bold: true,
        alignment: "center",
      });
    }
    x += width;
  });
  addText(slide, "Colour 520  +  texture 140  +  HOG 1,764  +  BoVW 1,280", {
    left: 52,
    top: 274,
    width: available,
    height: 30,
    fontSize: 19,
    color: C.ink,
    bold: true,
  });
  addText(slide, "VALIDATION TOP-1 ABLATION", {
    left: 52,
    top: 334,
    width: 510,
    height: 28,
    fontSize: 15,
    color: C.traditional,
    bold: true,
  });
  const ablations = [
    ["Texture", 100 * Number(textureAblation.val_top1)],
    ["Colour", 100 * Number(colourAblation.val_top1)],
    ["HOG", 100 * Number(hogAblation.val_top1)],
    ["Full", 100 * Number(fullAblation.val_top1)],
  ];
  const ablationMaximum = Math.max(...ablations.map(([, value]) => value));
  ablations.forEach((item, index) => {
    const y = 382 + index * 52;
    addText(slide, item[0], {
      left: 52,
      top: y,
      width: 95,
      height: 28,
      fontSize: 17,
      color: C.ink,
    });
    addShape(slide, {
      left: 154,
      top: y + 2,
      width: (item[1] / ablationMaximum) * 340,
      height: 24,
      fill: index === 3 ? C.blue : C.traditional,
    });
    addText(slide, `${item[1].toFixed(2)}%`, {
      left: 490,
      top: y,
      width: 82,
      height: 28,
      fontSize: 17,
      color: C.ink,
      bold: true,
      alignment: "right",
    });
  });
  addRule(slide, 620, 154, 1, C.rule, 466);
  const baselines = [
    ["Primary LinearSVC", "3,704-D combined descriptor\nBlock-wise power norm; alpha = 0.3"],
    ["Robustness Random Forest", "300 trees; 2,424-D handcrafted only\nPer-image corruption features can be recomputed"],
    ["SIFT-BoVW LinearSVC", "1,280-D; 256-word train-only vocabulary\nTop-3 soft assignment; 1 + 2 x 2 pooling"],
  ];
  baselines.forEach((item, index) => {
    const y = 158 + index * 142;
    addText(slide, item[0], {
      left: 666,
      top: y,
      width: 500,
      height: 36,
      fontSize: 24,
      color: index === 2 ? C.blue : C.ink,
      bold: true,
    });
    addText(slide, item[1], {
      left: 666,
      top: y + 42,
      width: 500,
      height: 70,
      fontSize: 19,
      color: C.muted,
    });
    if (index < baselines.length - 1) addRule(slide, 666, y + 124, 500);
  });
  addText(slide, "Metrics: top-1  /  top-5  /  macro P  /  R  /  F1", {
    left: 666,
    top: 582,
    width: 500,
    height: 34,
    fontSize: 19,
    color: C.green,
    bold: true,
  });
  addFooter(
    slide,
    "Citations: Lowe (2004); Csurka et al. (2004); Dalal & Triggs (2005); Ojala et al. (2002); Cortes & Vapnik (1995); Breiman (2001)",
    "P2",
  );
  setNotes(slide, scripts.p2.split("\n\n").slice(1).join("\n\n"), [
    "results/features/traditional_feature_config.json",
    "results/features/bovw/bovw_feature_config.json",
    "results/tables/final_report_ablation_results.csv",
    "report/cvpr_latex/sections/04_methods.tex",
  ]);
}

// Slide 6 - deep pipeline
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Deep learning learns visual features end to end",
    "P3",
    "Deep pipeline",
    6,
  );
  addFlow(
    slide,
    [
      "Input\nimage",
      "Train-time\naugmentation",
      "CNN\nbackbone",
      "500-class\nhead",
      "Best val\ncheckpoint",
      "Frozen test\nevaluation",
    ],
    {
      left: 70,
      top: 238,
      totalWidth: 1140,
      nodeHeight: 110,
      nodeFill: "#EEF6FF",
      accentIndex: 5,
      fontSize: 18,
    },
  );
  addText(slide, "TRAINING BRANCHES", {
    left: 280,
    top: 160,
    width: 720,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
    alignment: "center",
  });
  addText(slide, "Random initialisation", {
    left: 240,
    top: 390,
    width: 310,
    height: 44,
    fontSize: 24,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addText(slide, "ImageNet pretraining", {
    left: 730,
    top: 390,
    width: 310,
    height: 44,
    fontSize: 24,
    color: C.blue,
    bold: true,
    alignment: "center",
  });
  addRule(slide, 400, 364, 1, C.rule, 24);
  addRule(slide, 886, 364, 1, C.rule, 24);
  addText(slide, "Scratch ResNet18", {
    left: 250,
    top: 454,
    width: 290,
    height: 42,
    fontSize: 20,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, "ResNet18  /  EfficientNet-B0", {
    left: 706,
    top: 454,
    width: 356,
    height: 42,
    fontSize: 20,
    color: C.muted,
    alignment: "center",
  });
  addShape(slide, {
    left: 226,
    top: 526,
    width: 830,
    height: 70,
    fill: C.panel,
    lineFill: C.rule,
    lineWidth: 1,
  });
  addText(slide, "Validation macro F1 selects checkpoints; held-out test never tunes the model.", {
    left: 246,
    top: 546,
    width: 790,
    height: 36,
    fontSize: 22,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addFooter(
    slide,
    "Citations: He et al. (CVPR 2016); Deng et al. (CVPR 2009); Tan & Le (ICML 2019)",
    "P3",
  );
  setNotes(slide, scripts.p3.split("\n\n")[0], [
    "report/cvpr_latex/sections/04_methods.tex",
    "results/metrics/deep_learning/stage3_freeze.json",
    "results/metrics/deep_learning/stage3_test_access.json",
  ]);
}

// Slide 7 - models and augmentation
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Transfer matters when each class has only 40 images",
    "P3",
    "Models and selection",
    7,
  );
  addText(slide, "MODEL SET", {
    left: 52,
    top: 150,
    width: 500,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  const models = [
    ["Scratch ResNet18", "random weights", "test"],
    ["Pretrained ResNet18", "full fine-tuning", "test"],
    ["Pretrained EfficientNet-B0", "full fine-tuning", "test"],
    ["Frozen-head ResNet18", "linear head only", "val only"],
  ];
  models.forEach((item, index) => {
    const y = 190 + index * 92;
    addText(slide, item[0], {
      left: 52,
      top: y,
      width: 330,
      height: 32,
      fontSize: 22,
      color: index === 2 ? C.blue : C.ink,
      bold: true,
    });
    addText(slide, item[1], {
      left: 52,
      top: y + 36,
      width: 330,
      height: 28,
      fontSize: 17,
      color: C.muted,
    });
    addText(slide, item[2], {
      left: 410,
      top: y + 8,
      width: 104,
      height: 28,
      fontSize: 15,
      color: item[2] === "test" ? C.green : C.amber,
      bold: true,
      alignment: "right",
    });
    if (index < models.length - 1) addRule(slide, 52, y + 76, 462);
  });
  addRule(slide, 588, 150, 1, C.rule, 470);
  addText(slide, "MATCHED AUGMENTATION ABLATION", {
    left: 632,
    top: 150,
    width: 552,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  addShape(slide, {
    left: 632,
    top: 196,
    width: 246,
    height: 164,
    fill: C.panel,
    lineFill: C.rule,
    lineWidth: 1,
  });
  addText(slide, "BASIC", {
    left: 652,
    top: 214,
    width: 206,
    height: 28,
    fontSize: 17,
    color: C.muted,
    bold: true,
  });
  addText(slide, "Random crop\nHorizontal flip", {
    left: 652,
    top: 258,
    width: 206,
    height: 70,
    fontSize: 21,
    color: C.ink,
    bold: true,
  });
  addShape(slide, {
    left: 906,
    top: 196,
    width: 278,
    height: 164,
    fill: "#EEF6FF",
    lineFill: C.blue,
    lineWidth: 2,
  });
  addText(slide, "STRONG", {
    left: 926,
    top: 214,
    width: 238,
    height: 28,
    fontSize: 17,
    color: C.blue,
    bold: true,
  });
  addText(slide, "Wider crop + colour\nAffine + random erasing", {
    left: 926,
    top: 258,
    width: 238,
    height: 70,
    fontSize: 20,
    color: C.ink,
    bold: true,
  });
  addMetric(slide, percent(basicAblation.val_top1), "basic ResNet18 validation top-1", {
    left: 632,
    top: 402,
    width: 246,
    accent: C.muted,
    valueSize: 40,
  });
  addMetric(slide, percent(strongAblation.val_top1), "strong ResNet18 validation top-1", {
    left: 906,
    top: 402,
    width: 278,
    accent: C.blue,
    valueSize: 40,
  });
  addText(
    slide,
    `${Number(percentagePoints(strongAblation.val_top1, basicAblation.val_top1)) >= 0 ? "+" : ""}${percentagePoints(strongAblation.val_top1, basicAblation.val_top1)} percentage points  /  one-factor transform comparison`,
    {
    left: 632,
    top: 532,
    width: 552,
    height: 44,
    fontSize: 20,
    color: C.green,
    bold: true,
    alignment: "center",
    },
  );
  addText(slide, "Best checkpoint: validation macro F1  /  configuration hashes frozen before test", {
    left: 632,
    top: 586,
    width: 552,
    height: 38,
    fontSize: 17,
    color: C.muted,
    alignment: "center",
  });
  addFooter(
    slide,
    "Sources: final_report_ablation_results.csv; deep YAML/resolved configs; report Methods and Experimental Setup",
    "P3",
  );
  setNotes(slide, scripts.p3.split("\n\n").slice(1).join("\n\n"), [
    "results/tables/final_report_ablation_results.csv",
    "configs/deep_resnet18_scratch.yaml",
    "configs/deep_resnet18_pretrained_strong_aug.yaml",
    "configs/deep_efficientnet_b0_pretrained.yaml",
    "report/cvpr_latex/sections/05_experimental_setup.tex",
  ]);
}

// Slide 8 - main results
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Held-out comparison across frozen methods",
    "P4",
    "Main results",
    8,
  );
  addManualTable(slide, {
    left: 48,
    top: 160,
    widths: [380, 138, 138, 150],
    rowHeight: 64,
    fontSize: 18,
    alignments: ["left", "right", "right", "right"],
    highlightRows: { 6: "#DCEBFF" },
    rows: [
      ["Held-out test method", "Top-1", "Top-5", "Macro F1"],
      ["3,704-D handcrafted + BoVW LinearSVC", percent(primaryResult.test_top1), percent(primaryResult.test_top5), percent(primaryResult.test_macro_f1)],
      ["2,424-D handcrafted Random Forest", percent(forestResult.test_top1), percent(forestResult.test_top5), percent(forestResult.test_macro_f1)],
      ["1,280-D SIFT-BoVW LinearSVC", percent(bovwResult.test_top1), percent(bovwResult.test_top5), percent(bovwResult.test_macro_f1)],
      ["ResNet18 - scratch", percent(scratchResult.test_top1), percent(scratchResult.test_top5), percent(scratchResult.test_macro_f1)],
      ["ResNet18 - ImageNet + strong", percent(pretrainedResult.test_top1), percent(pretrainedResult.test_top5), percent(pretrainedResult.test_macro_f1)],
      ["EfficientNet-B0 - ImageNet + strong", percent(efficientResult.test_top1), percent(efficientResult.test_top5), percent(efficientResult.test_macro_f1)],
    ],
  });
  addRule(slide, 890, 160, 1, C.rule, 448);
  addMetric(slide, percent(strongestHeldOut.test_top1), "best held-out top-1", {
    left: 930,
    top: 166,
    width: 260,
    accent: C.blue,
  });
  addMetric(slide, percent(strongestHeldOut.test_top5), "best held-out top-5", {
    left: 930,
    top: 300,
    width: 260,
    accent: C.green,
  });
  addMetric(
    slide,
    `${Number(percentagePoints(efficientResult.test_top1, pretrainedResult.test_top1)) >= 0 ? "+" : ""}${percentagePoints(efficientResult.test_top1, pretrainedResult.test_top1)} pts`,
    "EfficientNet vs pretrained ResNet18 top-1",
    {
    left: 930,
    top: 434,
    width: 260,
    accent: C.purple,
    valueSize: 39,
    },
  );
  addText(slide, "Descriptive differences only\n1 seed  /  no confidence intervals", {
    left: 930,
    top: 566,
    width: 260,
    height: 56,
    fontSize: 17,
    color: C.red,
    bold: true,
  });
  addFooter(
    slide,
    "Source: results/tables/final_report_main_results.csv (manifest-bound frozen aggregate; exact decimals in Appendix A6)",
    "P4",
  );
  setNotes(slide, scripts.p4.split("\n\n")[0], [
    "results/tables/final_report_main_results.csv",
    "results/tables/final_report_tables_manifest.json",
    "results/metrics/traditional_linear_svm_metrics.json",
    "results/metrics/traditional_random_forest_metrics.json",
    "results/metrics/traditional_sift_bovw_svm_metrics.json",
    "results/metrics/deep_learning/resnet18_scratch_basic_test_metrics.json",
    "results/metrics/deep_learning/resnet18_pretrained_strong_aug_test_metrics.json",
    "results/metrics/deep_learning/efficientnet_b0_pretrained_strong_aug_test_metrics.json",
    "report/cvpr_latex/sections/06_results.tex",
  ]);
}

// Slide 9 - robustness and Grad-CAM
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Corruption reveals a separate reliability problem",
    "P4",
    "Robustness, explainability, limits",
    9,
  );
  const corruptedTop1Values = [
    Number(forestRobustness.mean_corrupted_top1),
    Number(scratchRobustness.mean_corrupted_top1),
    Number(pretrainedRobustness.mean_corrupted_top1),
    Number(efficientRobustness.mean_corrupted_top1),
  ];
  const corruptedAxisMaximum = Math.min(
    1,
    Math.max(0.1, Math.ceil((Math.max(...corruptedTop1Values) + 0.05) * 10) / 10),
  );
  addText(slide, "MEAN TOP-1 ACROSS 16 CORRUPTED CONDITIONS", {
    left: 50,
    top: 146,
    width: 630,
    height: 26,
    fontSize: 14,
    color: C.blue,
    bold: true,
  });
  slide.charts.add("bar", {
    position: { left: 42, top: 176, width: 650, height: 338 },
    categories: ["RF", "Scratch R18", "Pretrained R18", "EffNet-B0"],
    series: [
      {
        name: "Mean corrupted top-1 (%)",
        values: corruptedTop1Values,
        valuesFormatCode: "0.00%",
        fill: C.blue,
        points: [
          { idx: 0, fill: C.traditional },
          { idx: 1, fill: "#8FB9ED" },
          { idx: 2, fill: "#5A9DEB" },
          { idx: 3, fill: C.blue },
        ],
      },
    ],
    hasLegend: false,
    dataLabels: {
      showValue: true,
      position: "outEnd",
      textStyle: { fontSize: 15, fill: C.ink, bold: true },
    },
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
    xAxis: {
      visible: false,
      min: 0,
      max: corruptedAxisMaximum,
      majorUnit: 0.1,
      numberFormatCode: "0%",
      tickLabelPosition: "none",
      textStyle: { fontSize: 13, fill: C.muted },
      majorGridlines: null,
      line: { style: "solid", fill: C.white, width: 0 },
    },
    yAxis: {
      visible: true,
      textStyle: { fontSize: 13, fill: C.ink },
      line: { style: "solid", fill: C.rule, width: 1 },
    },
    chartFill: C.white,
    chartLine: { style: "solid", fill: C.white, width: 0 },
    plotAreaFill: "none",
    plotAreaLine: { style: "solid", fill: C.white, width: 0 },
  });
  // Artifact Tool and LibreOffice disagree on whether `tickLabelPosition: none`
  // suppresses value-axis labels. Mask the unused axis band, then label the bars
  // directly so the exact model/value mapping remains explicit in every renderer.
  addShape(slide, { left: 42, top: 478, width: 650, height: 42, fill: C.white });
  [
    ["EffNet-B0", 230, C.white, 150],
    ["Pretrained R18", 299, C.white, 170],
    ["Scratch R18", 368, C.ink, 150],
    ["RF", 437, C.ink, 52],
  ].forEach(([label, top, color, width]) => {
    addText(slide, label, {
      left: 56,
      top,
      width,
      height: 24,
      fontSize: 12,
      color,
      bold: true,
      verticalAlignment: "center",
    });
  });
  addShape(slide, {
    left: 58,
    top: 526,
    width: 612,
    height: 76,
    fill: "#FFF2F2",
    lineFill: C.red,
    lineWidth: 1,
  });
  addText(
    slide,
    `${robustnessDisplayNames[strongestRobustness.model_name]}: ${wordsLabel(strongestRobustness.worst_corruption_type)} severity ${strongestRobustness.worst_severity}\n${percent(strongestRobustnessClean.test_top1)} clean -> ${percent(strongestRobustness.worst_condition_top1)} top-1`,
    {
    left: 78,
    top: 536,
    width: 572,
    height: 56,
    fontSize: 20,
    color: C.red,
    bold: true,
    alignment: "center",
    },
  );
  addText(slide, "GRAD-CAM: CLASS-CONDITIONAL DIAGNOSTIC", {
    left: 734,
    top: 146,
    width: 490,
    height: 26,
    fontSize: 14,
    color: C.blue,
    bold: true,
  });
  await addImage(slide, "gradcam_correct_2713957.png", {
    left: 734,
    top: 182,
    width: 224,
    height: 250,
    fit: "cover",
    crop: { left: 0.75, top: 0.12, right: 0.004, bottom: 0.10 },
    alt: "Aligned Grad-CAM overlay for a high-confidence correct Jerusalem cherry prediction.",
  });
  await addImage(slide, "gradcam_incorrect_2759395.png", {
    left: 988,
    top: 182,
    width: 224,
    height: 250,
    fit: "cover",
    crop: { left: 0.744, top: 0.12, right: 0.004, bottom: 0.10 },
    alt: "Aligned Grad-CAM overlay for a matrimony vine incorrectly predicted as Jerusalem cherry.",
  });
  // These persisted composites include source titles above the panels. The
  // identities are restated accessibly below, so cover the scaled title strip
  // instead of exposing unreadable, partially cropped text.
  addShape(slide, { left: 734, top: 182, width: 224, height: 40, fill: C.white });
  addShape(slide, { left: 988, top: 182, width: 224, height: 40, fill: C.white });
  addText(slide, "Correct\nJerusalem cherry -> same\np = 0.999", {
    left: 734,
    top: 444,
    width: 224,
    height: 72,
    fontSize: 17,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addText(slide, "Confident error\nmatrimony vine -> Jerusalem cherry\np = 0.986", {
    left: 974,
    top: 444,
    width: 252,
    height: 72,
    fontSize: 16,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addText(slide, "Qualitative  /  coarse  /  non-causal  /  not segmentation", {
    left: 734,
    top: 542,
    width: 478,
    height: 36,
    fontSize: 18,
    color: C.red,
    bold: true,
    alignment: "center",
  });
  addText(slide, "Limits: one seed  |  10 test images/class  |  no confidence intervals  |  4 synthetic corruption families", {
    left: 58,
    top: 626,
    width: 1154,
    height: 28,
    fontSize: 16,
    color: C.muted,
    alignment: "center",
  });
  addFooter(
    slide,
    "Sources: final_report_robustness_summary.csv; robustness protocol; gradcam_selected_examples.csv; persisted Grad-CAM maps",
    "P4",
  );
  setNotes(slide, scripts.p4.split("\n\n").slice(1).join("\n\n"), [
    "results/tables/final_report_robustness_summary.csv",
    "results/metrics/stage4_robustness_protocol.json",
    "results/tables/gradcam_selected_examples.csv",
    "results/figures/gradcam/correct_high_confidence/2713957__row-4863__target-predicted-label-486_gradcam.png",
    "results/figures/gradcam/incorrect_high_confidence/2759395__row-4855__target-predicted-label-486_gradcam.png",
    "report/cvpr_latex/sections/07_discussion.tex",
  ]);
}

// Slide 10 - reproducibility
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Every reported result is connected to frozen evidence",
    "P5",
    "Report and reproducibility",
    10,
  );
  await addImage(slide, "method_overview.png", {
    left: 46,
    top: 160,
    width: 624,
    height: 250,
    fit: "contain",
    alt: "Leakage-safe workflow connecting frozen splits, validation selection, and frozen held-out evaluation.",
  });
  addText(slide, "TRACEABLE EVIDENCE CHAIN", {
    left: 728,
    top: 150,
    width: 470,
    height: 28,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  const chain = [
    "Seed-42 split manifests",
    "Stage 2 selection + Stage 3 freeze",
    "Final predictions + metric JSON",
    "Manifest-bound tables + figures",
    "Focused evidence-contract checks",
  ];
  // Chain connectors first.
  chain.slice(0, -1).forEach((_, index) => {
    addShape(slide, {
      left: 756,
      top: 227 + index * 76,
      width: 18,
      height: 30,
      geometry: "downArrow",
      fill: C.rule,
    });
  });
  chain.forEach((item, index) => {
    const y = 190 + index * 76;
    addShape(slide, {
      left: 728,
      top: y,
      width: 470,
      height: 52,
      fill: index === chain.length - 1 ? "#E8F7F1" : C.panel,
      lineFill: index === chain.length - 1 ? C.green : C.rule,
      lineWidth: 1,
    });
    addText(slide, item, {
      left: 752,
      top: y + 11,
      width: 420,
      height: 30,
      fontSize: 20,
      color: C.ink,
      bold: true,
      verticalAlignment: "center",
    });
  });
  addText(slide, "CVPR REPORT", {
    left: 54,
    top: 456,
    width: 180,
    height: 26,
    fontSize: 15,
    color: C.blue,
    bold: true,
  });
  addBulletList(
    slide,
    [
      "Final PDF and generated tables require the post-refresh evidence audit.",
      "Deterministic source ZIP excludes data, checkpoints, caches, and predictions.",
      "Focused integrity checks must pass against the refreshed evidence contract before recording.",
    ],
    { left: 54, top: 492, width: 600, fontSize: 18, lineHeight: 43, bulletColor: C.green },
  );
  addShape(slide, {
    left: 728,
    top: 594,
    width: 470,
    height: 54,
    fill: "#FFF7E8",
    lineFill: C.amber,
    lineWidth: 1,
  });
  addText(slide, "Pre-recording action: rehearse all five handovers, then run the strict evidence check.", {
    left: 744,
    top: 605,
    width: 438,
    height: 34,
    fontSize: 16,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addFooter(
    slide,
    "Sources: README.md; Stage 2 / Stage 3 freezes; final evidence manifests and focused contract tests",
    "P5",
  );
  setNotes(slide, scripts.p5.split("\n\n")[0], [
    "README.md",
    "results/metrics/stage2_freeze.json",
    "results/metrics/deep_learning/stage3_freeze.json",
    "results/metrics/stage4_robustness_protocol.json",
    "results/tables/stage4_summary.csv",
    "results/tables/final_report_tables_manifest.json",
    "results/final_report_figures/figure_manifest.json",
  ]);
}

// Slide 11 - demo
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Demo: verify hashes, then inspect persisted outputs",
    "P5",
    "Software demonstration",
    11,
  );
  addText(slide, `READ-ONLY CHECK  /  CAPTURED ${presentationEvidence.demoRuntimeSeconds}s PASS`, {
    left: 48,
    top: 146,
    width: 540,
    height: 26,
    fontSize: 14,
    color: C.green,
    bold: true,
  });
  addShape(slide, {
    left: 48,
    top: 182,
    width: 540,
    height: 396,
    fill: "#111820",
    lineFill: "#111820",
    lineWidth: 1,
  });
  addText(
    slide,
    `comp9517 % T=tests.test_stage5_tables.FrozenEvidenceTests\ncomp9517 % .venv/bin/python -m unittest -v \\\n  "$T.test_every_contracted_hash_matches" \\\n  "$T.test_all_nested_manifest_bindings_match_and_deduplicate"\n\n... test_every_contracted_hash_matches ... ok\n... test_all_nested_manifest_bindings ... ok\n\nRan 2 tests in ${presentationEvidence.demoRuntimeSeconds}s\n\nOK`,
    {
      left: 70,
      top: 206,
      width: 494,
      height: 340,
      fontSize: 13,
      color: "#DCE6EE",
      typeface: MONO,
    },
  );
  addText(slide, "SHOW, DON'T RECOMPUTE", {
    left: 640,
    top: 146,
    width: 574,
    height: 26,
    fontSize: 14,
    color: C.blue,
    bold: true,
  });
  const thumbnails = [
    ["final_model_comparison.png", "Frozen comparison", "6 held-out methods"],
    ["corruption_examples.jpg", "Robustness artifacts", "4 corruptions x 4 severities"],
    ["gradcam_correct_examples.jpg", "Correct Grad-CAM", "organism attention"],
    ["gradcam_incorrect_examples.jpg", "Failure Grad-CAM", "qualitative shortcut clue"],
  ];
  for (let index = 0; index < thumbnails.length; index += 1) {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const x = 640 + col * 290;
    const y = 184 + row * 206;
    await addImage(slide, thumbnails[index][0], {
      left: x,
      top: y,
      width: 266,
      height: 128,
      fit: "contain",
      alt: `${thumbnails[index][1]} preview from persisted project artifacts.`,
    });
    addText(slide, thumbnails[index][1], {
      left: x,
      top: y + 134,
      width: 266,
      height: 24,
      fontSize: 17,
      color: C.ink,
      bold: true,
      alignment: "center",
    });
    addText(slide, thumbnails[index][2], {
      left: x,
      top: y + 160,
      width: 266,
      height: 24,
      fontSize: 14,
      color: C.muted,
      alignment: "center",
    });
  }
  addShape(slide, {
    left: 48,
    top: 604,
    width: 1166,
    height: 48,
    fill: C.panel,
    lineFill: C.rule,
    lineWidth: 1,
  });
  addText(slide, "No training  /  no full evaluation  /  no robustness inference  /  no Grad-CAM regeneration  /  no network", {
    left: 64,
    top: 617,
    width: 1134,
    height: 24,
    fontSize: 17,
    color: C.red,
    bold: true,
    alignment: "center",
  });
  addFooter(
    slide,
    "Exact commands, expected output, durations, narration, and fallbacks: presentation/DEMO_RUNBOOK.md",
    "P5",
  );
  setNotes(slide, scripts.p5.split("\n\n")[1], [
    "tests/test_stage5_tables.py (two focused read-only hash-binding tests)",
    "presentation/assets/demo_validation_output.txt",
    "report/cvpr_latex/generated/main_results_table.tex",
    "results/final_report_figures/final_model_comparison.png",
    "results/final_report_figures/corruption_examples.jpg",
    "results/final_report_figures/gradcam_correct_examples.jpg",
    "results/final_report_figures/gradcam_incorrect_examples.jpg",
  ]);
}

// Slide 12 - takeaways
{
  const slide = deck.slides.add();
  addHeader(
    slide,
    "Clean and corrupted results answer different questions",
    "P5",
    "Final takeaways",
    12,
  );
  const takeaways = [
    ["01", "Traditional features", `${primaryResult.method_name}: ${percent(primaryResult.test_top1)} held-out top-1`, C.traditional],
    ["02", "Scratch deep learning", `${scratchResult.method_name}: ${percent(scratchResult.test_top1)} held-out top-1`, "#78A9E8"],
    ["03", "Held-out leader", `${strongestHeldOut.method_name}: ${percent(strongestHeldOut.test_top1)} held-out top-1`, C.blue],
    ["04", "Beyond clean accuracy", `${robustnessDisplayNames[strongestRobustness.model_name]} leads regenerated mean corrupted top-1`, C.red],
  ];
  takeaways.forEach((item, index) => {
    const y = 158 + index * 104;
    addText(slide, item[0], {
      left: 56,
      top: y,
      width: 58,
      height: 52,
      fontSize: 25,
      color: item[3],
      bold: true,
      alignment: "center",
    });
    addText(slide, item[1], {
      left: 140,
      top: y - 2,
      width: 330,
      height: 40,
      fontSize: 27,
      color: C.ink,
      bold: true,
    });
    addText(slide, item[2], {
      left: 486,
      top: y,
      width: 680,
      height: 48,
      fontSize: 23,
      color: C.muted,
    });
    if (index < takeaways.length - 1) addRule(slide, 140, y + 70, 1026);
  });
  addText(slide, "Thank you. Questions?", {
    left: 56,
    top: 596,
    width: 1120,
    height: 54,
    fontSize: 38,
    color: C.blue,
    bold: true,
  });
  addFooter(
    slide,
    "Conclusion source: frozen held-out metrics, robustness aggregates, Grad-CAM selection, and final report",
    "P5",
  );
  setNotes(slide, scripts.p5.split("\n\n").slice(2).join("\n\n"), [
    "results/tables/final_report_main_results.csv",
    "results/tables/final_report_robustness_summary.csv",
    "results/tables/gradcam_selected_examples.csv",
    "report/cvpr_latex/sections/08_conclusion.tex",
  ]);
}

// Slide 13 - appendix divider
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addText(slide, "APPENDIX", {
    left: 44,
    top: 40,
    width: 220,
    height: 30,
    fontSize: 15,
    color: C.purple,
    bold: true,
  });
  addText(slide, "Evidence for\nquestions", {
    left: 44,
    top: 186,
    width: 880,
    height: 230,
    fontSize: 76,
    color: C.ink,
    bold: true,
  });
  addText(
    slide,
    "Split protocol  /  feature details  /  hyperparameters  /  ablations  /  full metrics  /  robustness  /  Grad-CAM",
    {
      left: 44,
      top: 520,
      width: 1030,
      height: 94,
      fontSize: 26,
      color: C.muted,
    },
  );
  addText(slide, "13 / A1", {
    left: 1160,
    top: 682,
    width: 76,
    height: 18,
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
  setNotes(slide, "Appendix divider. Not included in the ten-minute timing.", [
    "All appendix evidence is derived from frozen repository artifacts listed on the following slides.",
  ]);
}

// Slide 14 - split details
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Exact dataset construction and leakage controls", 14);
  addText(slide, "DETERMINISTIC CONSTRUCTION", {
    left: 50,
    top: 150,
    width: 530,
    height: 26,
    fontSize: 14,
    color: C.purple,
    bold: true,
  });
  const steps = [
    "All 10,000 classes meet the 50 train_mini + 10 official-val threshold.",
    "PCG64 seed 42 samples 500 original category IDs; selected IDs are sorted.",
    "Per-class deterministic permutation selects 40 train + 10 internal val.",
    "Ten official-validation images per class become the held-out test set.",
    "Labels map contiguously to 0-499; manifests retain scientific names and IDs.",
  ];
  addBulletList(slide, steps, {
    left: 50,
    top: 192,
    width: 540,
    fontSize: 18,
    lineHeight: 76,
    bulletColor: C.purple,
  });
  addText(slide, "AUDITED LEAKAGE CONTROLS", {
    left: 650,
    top: 150,
    width: 540,
    height: 26,
    fontSize: 14,
    color: C.purple,
    bold: true,
  });
  addManualTable(slide, {
    left: 650,
    top: 192,
    widths: [240, 150, 150],
    rowHeight: 56,
    fontSize: 16,
    alignments: ["left", "center", "center"],
    highlightRows: { 4: "#E8F7F1" },
    rows: [
      ["Partition", "Rows", "Per class"],
      ["Train / train_mini", "20,000", "40"],
      ["Internal val / train_mini", "5,000", "10"],
      ["Held-out test / official val", "5,000", "10"],
      ["Cross-partition overlap", "0", "PASS"],
    ],
  });
  addBulletList(
    slide,
    [
      "No overlap by image ID, path, UUID filename, or independently checked byte hash.",
      "Train-only fitting for scalers, vocabulary, classifiers, and CNN optimisation.",
    ],
    { left: 650, top: 506, width: 540, fontSize: 17, lineHeight: 48, bulletColor: C.green },
  );
  addShape(slide, {
    left: 650,
    top: 602,
    width: 540,
    height: 62,
    fill: "#FFF8EC",
    lineFill: C.traditional,
    lineWidth: 1,
  });
  addText(
    slide,
    "Chronology: Stage 3 checkpoint/config hashes were verified before test access. Stage 2 is hash-bound now; historical pre-test locking was not independently established.",
    {
      left: 666,
      top: 610,
      width: 508,
      height: 46,
      fontSize: 14,
      color: C.ink,
      bold: true,
    },
  );
  addFooter(
    slide,
    "Sources: split_summary.json; split manifests; report Dataset and Protocol",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "data/splits/split_summary.json",
    "data/splits/train.csv; data/splits/val.csv; data/splits/test.csv",
    "report/cvpr_latex/sections/03_dataset_protocol.tex",
  ]);
}

// Slide 15 - traditional details
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Traditional feature dimensions and BoVW configuration", 15);
  addManualTable(slide, {
    left: 48,
    top: 154,
    widths: [210, 130, 380, 390],
    rowHeight: 60,
    fontSize: 16,
    alignments: ["left", "right", "left", "left"],
    rows: [
      ["Feature", "Dim.", "Configuration", "Normalisation / role"],
      ["Spatial colour", "520", "joint H-S 12x8 + V 8; 1 + 2x2 cells", "raw nonnegative block"],
      ["Uniform LBP", "140", "P=8/R=1 and P=16/R=2; 1 + 2x2", "raw multi-scale texture"],
      ["HOG", "1,764", "9 orientations; 16px cells; 2x2 blocks", "L2-Hys; shape"],
      ["Handcrafted", "2,424", "colour -> texture -> HOG", "Random-Forest comparator"],
      ["Combined", "3,704", "handcrafted -> BoVW", "power norm alpha=.3 + LinearSVC"],
    ],
  });
  addShape(slide, {
    left: 48,
    top: 538,
    width: 1210,
    height: 110,
    fill: "#FFF8EC",
    lineFill: C.traditional,
    lineWidth: 1,
  });
  addText(slide, "SIFT-BoVW", {
    left: 68,
    top: 558,
    width: 180,
    height: 32,
    fontSize: 22,
    color: C.traditional,
    bold: true,
  });
  addText(
    slide,
    "Aspect-preserving resize (long side 256)  /  grayscale SIFT  /  256-word MiniBatchKMeans vocabulary  /  4 train images per class  /  <=100 descriptors per vocabulary image  /  top-3 soft assignment  /  whole + 2x2 pooling  /  1,280-D",
    {
      left: 250,
      top: 552,
      width: 980,
      height: 72,
      fontSize: 16,
      color: C.ink,
    },
  );
  addText(slide, "Vocabulary fitting is train-only; the current manifest binds validation selection across exactly 19 configurations. Historical pre-test locking is not independently established.", {
    left: 68,
    top: 616,
    width: 1140,
    height: 34,
    fontSize: 13,
    color: C.red,
    bold: true,
  });
  addFooter(
    slide,
    "Sources: Stage 2 freeze; selection manifest; validation grid; traditional and BoVW feature configs; report Methods",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "results/metrics/stage2_freeze.json",
    "results/metrics/stage2_selection_manifest.json",
    "results/metrics/traditional_validation_grid.csv",
    "results/features/traditional_feature_config.json",
    "results/features/bovw/bovw_feature_config.json",
    "report/cvpr_latex/sections/04_methods.tex",
  ]);
}

// Slide 16 - deep hyperparameters
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Deep-learning hyperparameters and checkpoint selection", 16);
  addManualTable(slide, {
    left: 36,
    top: 150,
    widths: [300, 102, 120, 160, 155, 170, 150],
    rowHeight: 72,
    fontSize: 15,
    alignments: ["left", "right", "center", "center", "right", "center", "right"],
    highlightRows: { 3: "#EAF4FF", 5: "#EAF4FF" },
    rows: [
      ["Run", "LR", "Epoch cap", "Aug.", "Trainable", "Best epoch", "Test status"],
      ["ResNet18 scratch", "1e-3", "25", "basic", "11.43M", "23", "tested"],
      ["ResNet18 pretrained", "1e-4", "15", "basic", "11.43M", "15", "val only"],
      ["ResNet18 pretrained", "1e-4", "15", "strong", "11.43M", "14", "tested"],
      ["ResNet18 frozen head", "1e-3", "10", "basic", "0.257M", "10", "val only"],
      ["EfficientNet-B0 pretrained", "1e-4", "15", "strong", "4.65M", "14", "tested"],
    ],
  });
  addText(slide, "COMMON", {
    left: 52,
    top: 596,
    width: 110,
    height: 24,
    fontSize: 14,
    color: C.purple,
    bold: true,
  });
  addText(
    slide,
    "224 x 224 crop  /  batch 32  /  AdamW  /  weight decay 1e-4  /  cosine schedule  /  gradient clip 1.0  /  seed 42  /  validation macro F1 checkpoint selection",
    {
      left: 156,
      top: 590,
      width: 1070,
      height: 48,
      fontSize: 16,
      color: C.ink,
    },
  );
  addFooter(
    slide,
    "Sources: five deep YAML configs; five resolved-config JSONs; report Methods and Experimental Setup",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "configs/deep_resnet18_scratch.yaml",
    "configs/deep_resnet18_pretrained_basic.yaml",
    "configs/deep_resnet18_pretrained_strong_aug.yaml",
    "configs/deep_resnet18_pretrained_frozen_head.yaml",
    "configs/deep_efficientnet_b0_pretrained.yaml",
    "results/metrics/deep_learning/resnet18_scratch_basic_resolved_config.json",
    "results/metrics/deep_learning/resnet18_pretrained_basic_resolved_config.json",
    "results/metrics/deep_learning/resnet18_pretrained_strong_aug_resolved_config.json",
    "results/metrics/deep_learning/resnet18_pretrained_frozen_head_resolved_config.json",
    "results/metrics/deep_learning/efficientnet_b0_pretrained_strong_aug_resolved_config.json",
  ]);
}

// Slide 17 - validation ablations
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Validation-only ablations preserve test isolation", 17);
  addManualTable(slide, {
    left: 60,
    top: 150,
    widths: [470, 165, 165, 165, 210],
    rowHeight: 52,
    fontSize: 16,
    alignments: ["left", "right", "right", "right", "left"],
    highlightRows: { 4: "#FFF8EC", 7: "#EAF4FF" },
    rows: [
      ["Validation experiment", "Top-1", "Top-5", "Macro F1", "Interpretation"],
      ["Full 3,704-D combined reference", percent(fullAblation.val_top1), percent(fullAblation.val_top5), percent(fullAblation.val_macro_f1), "combined"],
      ["Colour only", percent(colourAblation.val_top1), percent(colourAblation.val_top5), percent(colourAblation.val_macro_f1), "colour"],
      ["Texture only", percent(textureAblation.val_top1), percent(textureAblation.val_top5), percent(textureAblation.val_macro_f1), "texture"],
      ["HOG only", percent(hogAblation.val_top1), percent(hogAblation.val_top5), percent(hogAblation.val_macro_f1), "single block"],
      ["ResNet18 scratch / basic", percent(scratchAblation.val_top1), percent(scratchAblation.val_top5), percent(scratchAblation.val_macro_f1), "descriptive"],
      ["ResNet18 pretrained / basic", percent(basicAblation.val_top1), percent(basicAblation.val_top5), percent(basicAblation.val_macro_f1), "reference"],
      ["ResNet18 pretrained / strong", percent(strongAblation.val_top1), percent(strongAblation.val_top5), percent(strongAblation.val_macro_f1), `${Number(percentagePoints(strongAblation.val_top1, basicAblation.val_top1)) >= 0 ? "+" : ""}${percentagePoints(strongAblation.val_top1, basicAblation.val_top1)} pts`],
      ["ResNet18 pretrained / frozen", percent(frozenAblation.val_top1), percent(frozenAblation.val_top5), percent(frozenAblation.val_macro_f1), "val only"],
    ],
  });
  addText(
    slide,
    "Only basic vs strong pretrained ResNet18 is a matched one-factor augmentation comparison. No significance claim: one principal seed, no repeated runs.",
    {
      left: 60,
      top: 628,
      width: 1150,
      height: 34,
      fontSize: 16,
      color: C.red,
      bold: true,
      alignment: "center",
    },
  );
  addFooter(slide, "Source: results/tables/final_report_ablation_results.csv");
  setNotes(slide, "Untimed appendix evidence.", [
    "results/tables/final_report_ablation_results.csv",
    "report/cvpr_latex/generated/ablation_results_table.tex",
    "results/tables/final_report_tables_manifest.json",
  ]);
}

// Slide 18 - full held-out metrics
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Complete held-out test metrics", 18);
  addManualTable(slide, {
    left: 34,
    top: 150,
    widths: [360, 120, 120, 150, 150, 150],
    rowHeight: 67,
    fontSize: 15,
    alignments: ["left", "right", "right", "right", "right", "right"],
    highlightRows: { 6: "#DCEBFF" },
    rows: [
      ["Method", "Top-1", "Top-5", "Macro P", "Macro R", "Macro F1"],
      ["3,704-D handcrafted + BoVW LinearSVC", decimal(primaryResult.test_top1), decimal(primaryResult.test_top5), decimal(primaryResult.test_macro_precision), decimal(primaryResult.test_macro_recall), decimal(primaryResult.test_macro_f1)],
      ["2,424-D handcrafted Random Forest", decimal(forestResult.test_top1), decimal(forestResult.test_top5), decimal(forestResult.test_macro_precision), decimal(forestResult.test_macro_recall), decimal(forestResult.test_macro_f1)],
      ["1,280-D SIFT-BoVW LinearSVC", decimal(bovwResult.test_top1), decimal(bovwResult.test_top5), decimal(bovwResult.test_macro_precision), decimal(bovwResult.test_macro_recall), decimal(bovwResult.test_macro_f1)],
      ["ResNet18 scratch / basic", decimal(scratchResult.test_top1), decimal(scratchResult.test_top5), decimal(scratchResult.test_macro_precision), decimal(scratchResult.test_macro_recall), decimal(scratchResult.test_macro_f1)],
      ["ResNet18 pretrained / strong", decimal(pretrainedResult.test_top1), decimal(pretrainedResult.test_top5), decimal(pretrainedResult.test_macro_precision), decimal(pretrainedResult.test_macro_recall), decimal(pretrainedResult.test_macro_f1)],
      ["EfficientNet-B0 pretrained / strong", decimal(efficientResult.test_top1), decimal(efficientResult.test_top5), decimal(efficientResult.test_macro_precision), decimal(efficientResult.test_macro_recall), decimal(efficientResult.test_macro_f1)],
    ],
  });
  addText(
    slide,
    "Balanced support means macro recall equals top-1 here. Per-class recall moves in 0.1 steps because the test split contains ten images per class.",
    {
      left: 60,
      top: 634,
      width: 1150,
      height: 28,
      fontSize: 16,
      color: C.muted,
      alignment: "center",
    },
  );
  addFooter(
    slide,
    "Source: final_report_main_results.csv; cross-checked against six metric JSONs and generated report tables",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "results/tables/final_report_main_results.csv",
    "results/metrics/traditional_linear_svm_metrics.json",
    "results/metrics/traditional_random_forest_metrics.json",
    "results/metrics/traditional_sift_bovw_svm_metrics.json",
    "results/metrics/deep_learning/resnet18_scratch_basic_test_metrics.json",
    "results/metrics/deep_learning/resnet18_pretrained_strong_aug_test_metrics.json",
    "results/metrics/deep_learning/efficientnet_b0_pretrained_strong_aug_test_metrics.json",
  ]);
}

// Slide 19 - robustness details
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Robustness: 4 corruptions x 4 severities x 4 models", 19);
  await addImage(slide, "robustness_top1_curves.png", {
    left: 40,
    top: 146,
    width: 770,
    height: 490,
    fit: "contain",
    alt: "Top-1 accuracy curves for four models under four corruptions at four severities.",
  });
  addText(slide, "SEVERITY SCHEDULE", {
    left: 848,
    top: 152,
    width: 350,
    height: 26,
    fontSize: 14,
    color: C.purple,
    bold: true,
  });
  const severity = [
    ["Gaussian blur", "kernel 3 / 5 / 9 / 15"],
    ["Gaussian noise", "sigma .02 / .05 / .10 / .20"],
    ["JPEG", "quality 80 / 50 / 30 / 10"],
    ["Motion blur", "length 5 / 9 / 15 / 25"],
  ];
  severity.forEach((item, index) => {
    const y = 194 + index * 74;
    addText(slide, item[0], {
      left: 848,
      top: y,
      width: 330,
      height: 28,
      fontSize: 19,
      color: C.ink,
      bold: true,
    });
    addText(slide, item[1], {
      left: 848,
      top: y + 32,
      width: 330,
      height: 26,
      fontSize: 16,
      color: C.muted,
    });
  });
  addShape(slide, {
    left: 840,
    top: 500,
    width: 374,
    height: 126,
    fill: "#FFF2F2",
    lineFill: C.red,
    lineWidth: 1,
  });
  addText(slide, `${robustnessDisplayNames[strongestRobustness.model_name]} worst condition`, {
    left: 860,
    top: 518,
    width: 334,
    height: 28,
    fontSize: 17,
    color: C.red,
    bold: true,
    alignment: "center",
  });
  addText(slide, `${wordsLabel(strongestRobustness.worst_corruption_type)} severity ${strongestRobustness.worst_severity}\n${decimal(strongestRobustnessClean.test_top1)} -> ${decimal(strongestRobustness.worst_condition_top1)}`, {
    left: 860,
    top: 554,
    width: 334,
    height: 58,
    fontSize: 19,
    color: C.ink,
    bold: true,
    alignment: "center",
  });
  addFooter(
    slide,
    "Sources: Stage 4 protocol; robustness_results.csv; final robustness summary",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "results/figures/robustness/robustness_top1_curves.png",
    "results/metrics/stage4_robustness_protocol.json",
    "results/tables/final_report_robustness_summary.csv",
    "results/tables/robustness_results.csv",
  ]);
}

// Slide 20 - Grad-CAM and reproducibility Q&A
{
  const slide = deck.slides.add();
  appendixHeader(slide, "Grad-CAM selection and reproducibility Q&A", 20);
  await addImage(slide, "gradcam_correct_examples.jpg", {
    left: 44,
    top: 150,
    width: 400,
    height: 312,
    fit: "contain",
    alt: "Four presentation-ready high-confidence correct Grad-CAM overlays.",
  });
  await addImage(slide, "gradcam_incorrect_examples.jpg", {
    left: 466,
    top: 150,
    width: 400,
    height: 312,
    fit: "contain",
    alt: "Four presentation-ready high-confidence incorrect Grad-CAM overlays.",
  });
  addText(slide, "DETERMINISTIC SELECTION", {
    left: 910,
    top: 154,
    width: 310,
    height: 26,
    fontSize: 14,
    color: C.purple,
    bold: true,
  });
  addMetric(slide, String(presentationEvidence.gradcamSelection.total_selected), "unique images", {
    left: 910,
    top: 184,
    width: 132,
    accent: C.blue,
    valueSize: 40,
  });
  addMetric(slide, String(presentationEvidence.gradcamManifest.cam_counts.total), "target-specific maps", {
    left: 1060,
    top: 184,
    width: 150,
    accent: C.green,
    valueSize: 40,
  });
  addBulletList(
    slide,
    [
      "12 correct + 12 incorrect + 9 confused-pair representatives",
      `EfficientNet-B0 layer ${presentationEvidence.gradcamManifest.method.target_layer}`,
      "Predicted-target map for all; true-target map for every error",
      "Class-conditional, coarse, qualitative, non-causal",
    ],
    { left: 910, top: 320, width: 304, fontSize: 16, lineHeight: 44, bulletColor: C.purple },
  );
  addRule(slide, 44, 506, 1170);
  addText(slide, "FAST READ-ONLY DEMO COMMAND", {
    left: 44,
    top: 520,
    width: 320,
    height: 24,
    fontSize: 14,
    color: C.green,
    bold: true,
  });
  addShape(slide, {
    left: 44,
    top: 550,
    width: 1170,
    height: 96,
    fill: "#111820",
    lineFill: "#111820",
    lineWidth: 1,
  });
  addText(
    slide,
    ".venv/bin/python -m unittest -v \\\n  tests.test_stage5_tables.FrozenEvidenceTests.test_every_contracted_hash_matches \\\n  tests.test_stage5_tables.FrozenEvidenceTests.test_all_nested_manifest_bindings_match_and_deduplicate",
    {
      left: 64,
      top: 567,
      width: 1130,
      height: 66,
      fontSize: 14,
      color: "#DCE6EE",
      typeface: MONO,
    },
  );
  addFooter(
    slide,
    "Sources: gradcam_selected_examples.csv; Grad-CAM manifests; hash-bound report composites; test_stage5_tables.py",
  );
  setNotes(slide, "Untimed appendix evidence.", [
    "results/tables/gradcam_selected_examples.csv",
    "results/metrics/explainability/gradcam_selection_manifest.json",
    "results/metrics/explainability/gradcam_manifest.json",
    "results/final_report_figures/figure_manifest.json",
    "tests/test_stage5_tables.py",
  ]);
}

await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
await fs.mkdir(RENDER_DIR, { recursive: true });

for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    path.join(RENDER_DIR, `${stem}.png`),
    new Uint8Array(await png.arrayBuffer()),
  );
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
}

const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(
  path.join(RENDER_DIR, "deck-montage.webp"),
  new Uint8Array(await montage.arrayBuffer()),
);

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUTPUT);

console.log(`Created ${OUTPUT}`);
console.log(`Slides: ${deck.slides.items.length} (12 main + 8 appendix)`);
console.log(`Artifact-tool renders: ${RENDER_DIR}`);
