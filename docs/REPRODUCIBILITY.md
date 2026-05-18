Reproducibility Notes
---------------------

The framework operates under frozen detector contracts and deterministic execution procedures.

Repository structure:

n1_toolkit/ → time-order validation tools
n2_toolkit/ → interaction validation tools
n4_toolkit/ → closure validation tools
shared_pipeline/ → shared preprocessing and utilities
audit_tools/ → audit and reproducibility checks

Scope of Included Examples
--------------------------

The repository contains representative excerpts from study inputs and locked outputs intended to demonstrate detector interfaces, execution structure, falsification behavior, and verdict formatting.

Due to the size and operational complexity of the complete evaluation pipelines, the repository does not include all raw datasets, full sweep archives, or every intermediate execution artifact generated during large-scale analysis.

The included materials are intended as lightweight reproducibility references accompanying the manuscript and supplementary documentation, while preserving the frozen-contract execution structure and audit logic used throughout the study.

General workflow:

1. Prepare time-series inputs.
2. Execute toolkit evaluations under fixed preprocessing conditions.
3. Apply mandatory falsification controls.
4. Export observable summaries and PASS/FAIL outcomes.
5. Verify deterministic rerun consistency using audit tools.

The framework preserves PASS, FAIL, and NO_CLEAR outcomes under fixed evaluation contracts without adaptive retuning.

This repository accompanies the manuscript:

“A Falsification-Based Framework for Validating Structure in Time-Series Data”
