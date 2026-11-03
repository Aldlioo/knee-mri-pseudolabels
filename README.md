# Learning twelve knee MRI findings from 58 expert labels

Code for the paper *"Learning twelve knee MRI findings from 58 expert labels: report labels from large language models and
iterative out-of-fold pseudo-labelling"* (under review). It was developed for the
[RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) challenge,
in which only 58 of 4407 training studies carry expert labels and the rest come with free-text radiology reports.

The method has three parts:

1. **Report labels from open large language models**: Qwen2.5-7B-Instruct and Gemma 4 26B-A4B-it grade each of the
   twelve findings in every report, with the organisers' annotation criteria in the prompt (macro AUC 0.898 against the
   expert labels).
2. **A 2.5D reader**: a 2D backbone (ConvNeXt-Tiny or EfficientNetV2-S) on three-slice windows of five canonicalised
   series (left knees mirrored, 100 mm physical crop, 224 px), a transformer over all windows of a study, and one
   attention pooling per finding; five folds grouped by acquisition setting plus one full-data model per family.
3. **Iterative out-of-fold pseudo-labelling**: the next round's target is `0.5 * report labels + 0.5 * mean out-of-fold
   probability of the previous models`. The target rose from 0.900 to 0.932, 0.937 and 0.942; the same reader went
   from 0.907 to 0.925 out-of-fold, and the best ensemble reached 0.933 out-of-fold and 0.932 on the private leaderboard.

## Repository layout

All notebooks are written as `# %%` Python scripts and run as Kaggle notebooks (two T4 GPUs, internet off unless
stated in the header of each script). Each header lists the inputs to attach and the settings.
`tools/py2ipynb.ps1` converts a script to `.ipynb`.

| Step | Script | What it does |
|---|---|---|
| Setup | `00_wheels.py` | Builds offline wheels for DICOM decoding (pylibjpeg, gdcm) |
| Data | `01_eda.py` | Dataset statistics: series, planes, languages, label prevalence |
| Report labels | `02_report_labeler.py`, `02b_labeler_gemma.py` | Qwen2.5-7B and Gemma 4 report labellers (zero-shot, graded 0/1/2, official criteria) |
| Preprocessing | `03_preprocess.py`, `03c_preprocess_m2.py`, `03d_preprocess_zoom.py`, `03e_preprocess_hires.py` | Series selection into 5 slots, reorientation, left-knee mirroring, physical crop, 8-bit cache |
| Training | `04e_train_reader.py` | The 2.5D reader; `FAMILY` selects the model: `m4`-`m6` (round 1), `m7` (round 2), `m9`/`m11` (round 3), `m12` (round 4), `m8` (320 px) |
| | `04f`-`04k` scripts | Thin wrappers that run `04e` for one family and a set of folds (two folds in parallel, one per GPU) |
| | `04_train.py`, `04c_*`, `04d_*` | Earlier models M1 (ResNet-34), M2 (ResNet-34, 5 slots) and M3 (DINOv2 slice transformer) |
| Pseudo-labels | `09_label_lab.py` | Scores label sources and candidate targets on the 58 expert studies; writes `targets_v2`-`v4` |
| Inference | `05_submit.py`, `05c_submit_m2.py` | Test-time preprocessing (identical to the cache), dense windows, rank ensemble of model families |
| Analysis | `10_window_check.py`, `10b_hires_read_check.py`, `14_ensemble_check.py` | Test-time window layouts, high-resolution reads, out-of-fold ensemble selection with a paired bootstrap |
| Figures | `16_paper_mri_figure.py`, `paper_figures/` | Example-study figure (run on Kaggle) and the charts of the paper |

## Data

The challenge data are available to registered participants under the challenge rules and the RSNA licence. This
repository contains **no data, labels, model weights or notebook outputs**. Run the notebooks on Kaggle with the
competition data attached; outputs of earlier notebooks (caches, label files, fold models) are attached as inputs to
later ones, as listed in each script's header.

## Acknowledgements

The reader architecture follows the public 2.5D ConvNeXt multiple-instance reader by goodpjw2008 on Kaggle
(Apache 2.0); see `NOTICE`. The round-1 target also uses two public label sets shared by challenge participants:
the six-source consensus labels (Kaggle dataset `karttikjangid05/rsna-knee-public-llm-label-sets-compared`, CC0) and
Claude Opus 4.5 report labels (`NTejas-1/RSNA-Knee-Abnormality-Detection` on GitHub, MIT).

## Licence

Apache License 2.0; see `LICENSE`.
