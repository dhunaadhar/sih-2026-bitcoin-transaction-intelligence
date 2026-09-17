"""
M6.12 — Graph Temporal Leakage Provenance Audit

Purpose
-------
Trace graph/entity-derived model features back to the M5 artifacts and
determine whether their construction is:

    SAFE_STATIC
    HISTORICALLY_CAUSAL
    POTENTIALLY_FUTURE_DEPENDENT
    DEFINITELY_GLOBAL

This is a provenance audit. It does NOT modify the benchmark dataset.

The audit is intentionally conservative:
if a feature was constructed from the complete observation window and
there is no time restriction in its construction, it is marked
DEFINITELY_GLOBAL rather than being assumed safe.

Class semantics are not inferred.
"""

from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "feature_manifest.json"
)

UNIFIED_FEATURES_PATH = (
    ROOT
    / "data"
    / "derived"
    / "unified_features.parquet"
)

ENTITY_EVIDENCE_PATH = (
    ROOT
    / "data"
    / "derived"
    / "entity_evidence.parquet"
)

TX_ENTITY_EVIDENCE_PATH = (
    ROOT
    / "data"
    / "derived"
    / "transaction_entity_evidence.parquet"
)

COSPONSOR_CLUSTERS_PATH = (
    ROOT
    / "data"
    / "derived"
    / "cosponsor_cluster_thresholds.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "graph_temporal_provenance.json"
)


# ---------------------------------------------------------------------
# KNOWN M5 PROVENANCE
# ---------------------------------------------------------------------

# Features known to originate from the M5 graph/entity pipeline.
#
# The provenance is intentionally based on the actual M5 construction
# already established for this project:
#
#   AddrAddr_edgelist
#       -> co-sponsorship graph
#       -> connected components / clusters
#       -> entity_evidence
#       -> transaction_entity_evidence
#       -> unified_features
#
# M5 artifacts were constructed over the complete available dataset.
# They were not generated separately for each historical time step.

KNOWN_GLOBAL_GRAPH_FEATURES = {
    "entity_input_address_count",
    "entity_output_address_count",

    "input_repeated_cosponsor_count",
    "input_repeated_cosponsor_ratio",
    "output_repeated_cosponsor_count",
    "output_repeated_cosponsor_ratio",

    "input_max_repeated_cosponsor_degree",
    "output_max_repeated_cosponsor_degree",

    "input_max_shared_transaction_count",
    "output_max_shared_transaction_count",

    "input_distinct_clusters_t1",
    "input_distinct_clusters_t2",
    "input_distinct_clusters_t3",

    "output_distinct_clusters_t1",
    "output_distinct_clusters_t2",
    "output_distinct_clusters_t3",

    "input_max_cluster_size_t1",
    "input_max_cluster_size_t2",
    "input_max_cluster_size_t3",

    "output_max_cluster_size_t1",
    "output_max_cluster_size_t2",
    "output_max_cluster_size_t3",
}


# Some M5-derived features may have names differing slightly from the
# explicit list above. These keywords are used only for discovery.
GRAPH_KEYWORDS = [
    "cosponsor",
    "cluster",
    "repeated",
    "shared",
    "entity",
]


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

def load_manifest():
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_feature_columns():
    df = pd.read_parquet(
        UNIFIED_FEATURES_PATH
    )

    return list(df.columns)


def load_artifact_metadata():
    metadata = {}

    paths = {
        "entity_evidence": ENTITY_EVIDENCE_PATH,
        "transaction_entity_evidence": TX_ENTITY_EVIDENCE_PATH,
        "cosponsor_cluster_thresholds": COSPONSOR_CLUSTERS_PATH,
    }

    for name, path in paths.items():

        if not path.exists():

            metadata[name] = {
                "exists": False,
                "path": str(path),
            }

            continue

        df = pd.read_parquet(path)

        metadata[name] = {
            "exists": True,
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

        if "time_step" in df.columns:

            metadata[name][
                "has_time_step_column"
            ] = True

            metadata[name][
                "time_step_min"
            ] = int(df["time_step"].min())

            metadata[name][
                "time_step_max"
            ] = int(df["time_step"].max())

        else:

            metadata[name][
                "has_time_step_column"
            ] = False

    return metadata


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------

def classify_feature(feature):
    """
    Conservative feature classification.

    Graph/entity features from the global M5 graph construction are
    classified as DEFINITELY_GLOBAL.

    Features outside the graph/entity pipeline are not automatically
    classified as safe because this script is specifically a graph
    provenance audit.
    """

    if feature in KNOWN_GLOBAL_GRAPH_FEATURES:

        return {
            "category": "DEFINITELY_GLOBAL",
            "reason": (
                "M5 graph/entity evidence was constructed over "
                "the complete available dataset rather than "
                "within historical time windows."
            ),
            "temporal_safe_for_chronological_test": False,
            "requires_rebuild": True,
        }

    lower = feature.lower()

    if any(
        keyword in lower
        for keyword in GRAPH_KEYWORDS
    ):

        return {
            "category": "POTENTIALLY_FUTURE_DEPENDENT",
            "reason": (
                "Feature name indicates graph/entity provenance, "
                "but the exact construction path requires "
                "artifact-level confirmation."
            ),
            "temporal_safe_for_chronological_test": False,
            "requires_rebuild": True,
        }

    return {
        "category": "OUTSIDE_GRAPH_AUDIT",
        "reason": (
            "Feature is not identified as an M5 graph/entity "
            "feature by the current provenance map."
        ),
        "temporal_safe_for_chronological_test": None,
        "requires_rebuild": False,
    }


# ---------------------------------------------------------------------
# DATA-LEVEL CHECK
# ---------------------------------------------------------------------

def inspect_graph_features(feature_names):
    """
    Inspect actual unified feature columns.

    Returns only features that exist in the current unified matrix.
    """

    unified_columns = set(
        load_feature_columns()
    )

    discovered = []

    for feature in feature_names:

        if feature not in unified_columns:
            continue

        classification = classify_feature(
            feature
        )

        if (
            classification["category"]
            != "OUTSIDE_GRAPH_AUDIT"
        ):

            discovered.append(
                {
                    "feature": feature,
                    **classification,
                }
            )

    return discovered


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def build_summary(features):
    counts = {}

    for item in features:

        category = item["category"]

        counts[category] = (
            counts.get(category, 0) + 1
        )

    definitely_global = [
        item["feature"]
        for item in features
        if item["category"]
        == "DEFINITELY_GLOBAL"
    ]

    potentially_future = [
        item["feature"]
        for item in features
        if item["category"]
        == "POTENTIALLY_FUTURE_DEPENDENT"
    ]

    return {
        "category_counts": counts,
        "definitely_global_features": definitely_global,
        "potentially_future_dependent_features": (
            potentially_future
        ),
        "total_graph_features": len(features),
        "requires_temporal_safe_rebuild": bool(
            definitely_global
            or potentially_future
        ),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "M6.12 — GRAPH TEMPORAL LEAKAGE "
        "PROVENANCE AUDIT"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # MANIFEST
    # -------------------------------------------------------------

    manifest = load_manifest()

    model_candidates = manifest[
        "model_candidates"
    ]

    print(
        f"Model candidate features: "
        f"{len(model_candidates)}"
    )

    # -------------------------------------------------------------
    # ARTIFACT METADATA
    # -------------------------------------------------------------

    print()
    print(
        "Inspecting M5 graph artifacts..."
    )

    artifact_metadata = (
        load_artifact_metadata()
    )

    for name, metadata in artifact_metadata.items():

        print()
        print(
            f"{name}:"
        )

        print(
            f"  Exists: "
            f"{metadata['exists']}"
        )

        if metadata["exists"]:

            print(
                f"  Rows: "
                f"{metadata['rows']:,}"
            )

            print(
                f"  Columns: "
                f"{len(metadata['columns'])}"
            )

            print(
                f"  Has time_step: "
                f"{metadata['has_time_step_column']}"
            )

            if metadata[
                "has_time_step_column"
            ]:

                print(
                    f"  Time range: "
                    f"{metadata['time_step_min']}"
                    f"–"
                    f"{metadata['time_step_max']}"
                )

    # -------------------------------------------------------------
    # FEATURE DISCOVERY
    # -------------------------------------------------------------

    print()
    print(
        "Tracing graph/entity model features..."
    )

    features = inspect_graph_features(
        model_candidates
    )

    summary = build_summary(
        features
    )

    # -------------------------------------------------------------
    # PRINT FEATURE CLASSIFICATION
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "FEATURE PROVENANCE CLASSIFICATION"
    )
    print("-" * 70)

    for item in features:

        print()
        print(
            f"{item['feature']}"
        )

        print(
            f"  Category: "
            f"{item['category']}"
        )

        print(
            f"  Temporal-safe: "
            f"{item['temporal_safe_for_chronological_test']}"
        )

        print(
            f"  Rebuild required: "
            f"{item['requires_rebuild']}"
        )

        print(
            f"  Reason: "
            f"{item['reason']}"
        )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "AUDIT SUMMARY"
    )
    print("-" * 70)

    print(
        f"Graph/entity features: "
        f"{summary['total_graph_features']}"
    )

    for category, count in sorted(
        summary["category_counts"].items()
    ):

        print(
            f"{category}: {count}"
        )

    print()

    if summary[
        "requires_temporal_safe_rebuild"
    ]:

        print(
            "STATUS: TEMPORAL-SAFE GRAPH REBUILD REQUIRED"
        )

    else:

        print(
            "STATUS: NO GRAPH REBUILD FLAGGED"
        )

    # -------------------------------------------------------------
    # IMPORTANT METHODOLOGICAL NOTE
    # -------------------------------------------------------------

    methodology_note = (
        "The current M5 graph/entity artifacts were "
        "constructed over the complete available "
        "transaction/address observation window. "
        "Consequently, graph-derived aggregates and "
        "clusters that depend on relationships observed "
        "later in the dataset cannot be considered "
        "causally valid for earlier transactions without "
        "a time-aware rebuild. This audit therefore "
        "flags those features conservatively rather than "
        "claiming that observed predictive performance "
        "is free of graph temporal leakage."
    )

    # -------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------

    report = {
        "stage": "M6.12",

        "task": (
            "graph_temporal_leakage_provenance_audit"
        ),

        "status": (
            "TEMPORAL_SAFE_GRAPH_REBUILD_REQUIRED"
            if summary[
                "requires_temporal_safe_rebuild"
            ]
            else "NO_GRAPH_REBUILD_FLAGGED"
        ),

        "model_candidate_feature_count": (
            len(model_candidates)
        ),

        "graph_feature_audit": {
            "features": features,
            "summary": summary,
        },

        "m5_artifacts": artifact_metadata,

        "methodology": {
            "classification_policy": (
                "Conservative provenance classification"
            ),
            "global_graph_features_are_safe": False,
            "future_information_claim": (
                "A global graph construction is treated "
                "as potentially future-dependent when "
                "used in chronological evaluation."
            ),
            "test_results_are_not_invalidated": (
                "The audit does not delete or alter "
                "previous benchmark results. It identifies "
                "a methodological limitation that must be "
                "resolved before production claims."
            ),
        },

        "required_next_step": (
            "Rebuild graph/entity features causally so "
            "each transaction only uses graph relationships "
            "available at or before its observation time."
        ),

        "methodology_note": methodology_note,

        "class_semantics": (
            "Numeric class IDs retained; no semantic "
            "interpretation assigned."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        f"Report: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()