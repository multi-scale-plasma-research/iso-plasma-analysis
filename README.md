# ISO Perihelion Plasma-Beta Analysis (OMNI + ACE Cross-Check)

This directory contains the fully reproducible analysis pipeline for the
ISO perihelion electromagnetic anomalies paper (submitted to ApJ).

It implements:
- An OMNI-based computation of plasma beta, β = p_th / p_mag, from hourly
  proton density, temperature, and |B|.
- Baseline vs perihelion window statistics for 1I/'Oumuamua, 2I/Borisov,
  and 3I/ATLAS using a 30-day pre-perihelion baseline and ±7-day perihelion
  window.
- Cross-checks against legacy ACE Level 2–based results used in the
  originally submitted analysis.

## Layout

- [scripts/analyze_plasma_pipeline.py](cci:7://file:///e:/Research/Infrastructure-build/scripts/analyze_plasma_pipeline.py:0:0-0:0)  
  Main analysis driver that loads the extended OMNI CSV and computes
  perihelion vs baseline β statistics for the ISOs.

- [scripts/compare_iso_beta_ace_omni.py](cci:7://file:///e:/Research/Infrastructure-build/scripts/compare_iso_beta_ace_omni.py:0:0-0:0)  
  Compares month-level OMNI β medians against ACE month-level medians from
  [Plasma_Beta_Summary_Statistics.csv](cci:7://file:///e:/Research/Instrument-Data/Plasma_Beta_Summary_Statistics.csv:0:0-0:0).

- [scripts/compare_iso_beta_windows_ace_omni.py](cci:7://file:///e:/Research/Infrastructure-build/scripts/compare_iso_beta_windows_ace_omni.py:0:0-0:0)  
  Uses the same baseline/perihelion windows as the main pipeline to compute
  OMNI β statistics and joins them with ACE month-level medians, providing
  the robustness/dataset-choice comparison discussed in the manuscript.

- `data/omni/omni_complete_corrected.csv`  
  Extended OMNI hourly dataset (1996–2025) produced from OMNI CDFs.

- `data/external/Plasma_Beta_Summary_Statistics.csv`  
  ACE-derived month-level β summaries used in the original paper analysis.

## Environment

Install Python 3.11+ and then:

```bash
pip install -r requirements.txt