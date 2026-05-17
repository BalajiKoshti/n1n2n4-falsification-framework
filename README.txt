ψ4 Falsification Framework for Time-Series Structure Validation
==============================================================

Author
------
Balaji Koshti
Independent Researcher, India
balaji.koshti25@gmail.com

Overview
--------
This repository contains the reference implementation, manuscript, supplementary material, representative figures, and reproducibility examples associated with the paper:

"A Falsification-Based Framework for Validating Structure in Time-Series Data"

The framework evaluates structured dependencies in time-series signals through a hierarchy of three toolkit levels:

• n1 — time-order structure
• n2 — interaction structure
• n4 — closure structure

Each toolkit operates under a frozen contract consisting of:
• predefined observables,
• mandatory falsification procedures,
• and strict decision criteria.

The framework is designed to preserve fail-closed behavior:
structure is accepted only when the defining observable degrades correctly under controlled falsification.

Repository Structure
--------------------

code/
    Core toolkit implementations and audit utilities.

    n1_toolkit/
        n1 time-order toolkit.

    n2_toolkit/
        n2 interaction toolkit.

    n4_toolkit/
        n4 closure toolkit.

    shared_pipeline/
        Shared processing and utility components.

    audit_tools/
        Reproducibility and falsification audit utilities.

data_examples/
    Minimal representative example inputs for reproducibility demonstration.

results_examples/
    Representative output summaries generated from the example inputs.

figures/
    manuscript_figures/
        Main-paper figures.

    supplementary_figures/
        Supplementary falsification, cross-domain, and audit figures.

manuscript/
    Main manuscript (.pdf and .docx).

supplement/
    Supplementary document (.pdf and .docx).

docs/
    Reproducibility notes and execution guidance.

Key Principles
--------------
The framework does not infer causality or universal structure.

Instead, it evaluates whether observed structure survives explicit falsification tests.

Important properties:
• FAIL outcomes are preserved and treated as meaningful results.
• Generic similarity alone is not sufficient for structural acceptance.
• Interaction and closure are accepted only when they collapse correctly under their defining falsifiers.
• The framework functions as a complementary validation layer rather than a replacement for domain-specific methods.

Representative Results
----------------------
Representative evaluations included in this repository demonstrate:

• n1:
    directional asymmetry may exist without satisfying irreversibility criteria.

• n2:
    alignment-dependent interaction collapses under mismatch pairing.

• n4:
    closure is selective and degrades under mandatory closure-breaking controls.

Across evaluated systems, the framework preserves conservative fail-closed behavior and rejects unsupported structural interpretation.

Reproducibility
---------------
All representative examples are intentionally lightweight and are included only to demonstrate:

• expected input structure,
• execution format,
• observable outputs,
• and verdict behavior.

They are not complete raw archives of the full study datasets.

License
-------
See LICENSE.txt