"""Evidence for explicitly accepting finite FP16 experiments, not accuracy approval."""

EXPERIMENT_VERSION = 1
EXPERIMENT_KIND = "fp16-finite-experiment"


def finite_execution_passed(report, backend):
    """Retain shape, finite, repeat, I/O and placement checks while relaxing accuracy."""
    provider = {"cpu": "CPUExecutionProvider", "directml": "DmlExecutionProvider"}.get(backend)
    runs = report.get("runs", {})
    if provider is None or set(runs) != {"fp32", "fp16", "fp16-diagnostic", "fp16-profile"}:
        return False
    if any(not run.get("cases") for run in runs.values()):
        return False
    profile = runs["fp16-profile"].get("profile", {})
    if backend == "directml":
        if (profile.get("directml_fp16_convolution_observed") is not True
                or profile.get("cpu_neural_compute_events") != {}):
            return False
    elif (set(profile.get("provider_events", {})) != {provider}
            or not profile.get("cpu_neural_compute_events") or not profile.get("typed_operator_events")):
        return False
    for label, run in runs.items():
        if run.get("public_io_fp32") is not True or run.get("providers", [])[:1] != [provider]:
            return False
        if backend == "cpu" and run["providers"] != [provider]:
            return False
        for row in run["cases"].values():
            if row.get("finite") is not True or not row.get("repeat_checks"):
                return False
            if not all(repeat and all(check.get("passed") is True for check in repeat.values())
                       for repeat in row["repeat_checks"]):
                return False
            if label != "fp32":
                if row.get("shape_matches_baseline") is not True or not isinstance(row.get("engineering_metrics", {}).get("passed"), bool):
                    return False
            elif "original_fp32_checks" in row and not all(
                    check.get("passed") is True for check in row["original_fp32_checks"].values()):
                return False
            if label in ("fp16-diagnostic", "fp16-profile") and row.get("production_compare", {}).get("passed") is not True:
                return False
    return True


def validate_experiment(manifest, report, backend):
    """Match a published result to the backend and acoustic files being loaded."""
    if (backend not in ("cpu", "directml") or report.get("status") != "completed" or report.get("backend") != backend
            or report.get("finite_experiment", {}).get("version") != EXPERIMENT_VERSION
            or report["finite_experiment"].get("passed") is not True):
        raise ValueError(f"{backend} FP16 finite experiment has incomplete execution evidence")
    bindings = {"candidate_graph_sha256": manifest["graphs"]["decode"]["sha256"],
                "candidate_diagnostic_sha256": manifest["graphs"]["diagnostic"]["sha256"],
                "candidate_weights_sha256": manifest["weights"]["sha256"],
                "source_manifest_sha256": manifest["precision"]["source_manifest_sha256"]}
    if any(report.get(key) != value for key, value in bindings.items()):
        raise ValueError("FP16 finite experiment does not match the candidate package")


def validate_experiment_for_publication(manifest, report, backend):
    """Check the complete experiment once, before publishing its hashed result."""
    validate_experiment(manifest, report, backend)
    if (not isinstance(report.get("engineering_screen", {}).get("passed"), bool)
            or not report.get("onnxruntime") or not finite_execution_passed(report, backend)):
        raise ValueError(f"{backend} FP16 finite experiment has incomplete execution evidence")
    precision = manifest["precision"]
    for label, run in report["runs"].items():
        graph = (precision["source_graphs"]["decode"]["sha256"] if label == "fp32" else
                 manifest["graphs"]["diagnostic" if label == "fp16-diagnostic" else "decode"]["sha256"])
        weights = precision["source_weights"]["sha256"] if label == "fp32" else manifest["weights"]["sha256"]
        if run.get("graph_sha256") != graph or run.get("weights_sha256") != weights:
            raise ValueError("FP16 finite experiment run hashes do not match the candidate package")
