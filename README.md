<div align="center">
  <img src="assets/integrating-sphere-banner.svg" alt="Integrating Sphere — Camera Sensor Characterization" width="100%" />

  <p><strong>Optical hardware development · Camera sensor testing · Python image analysis</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Domain-Optical%20Instrumentation-245B78?style=flat-square" alt="Optical instrumentation" />
    <img src="https://img.shields.io/badge/Analysis-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Data-FITS%20%7C%20TIFF-596579?style=flat-square" alt="FITS and TIFF" />
    <img src="https://img.shields.io/badge/Project-IIT%20Kanpur-9B1C31?style=flat-square" alt="IIT Kanpur" />
  </p>

  <p><a href="#overview">Overview</a> · <a href="#characterization-workflow">Workflow</a> · <a href="#results-gallery">Results</a> · <a href="#repository-structure">Repository</a></p>
</div>

---

## Overview

This repository documents a camera-sensor characterization project built around an integrating-sphere setup. The work combines optical test hardware with Python-based analysis of camera image datasets to examine sensor response, noise, uniformity and image-quality characteristics.

The analysis scripts and result files are organized by characterization task. The repository includes FITS image data, Python analysis scripts, plots and exported numerical summaries.

## Characterization workflow

| Area | What is examined | Available project material |
|---|---|---|
| Photon Transfer Curve (PTC) | Signal–variance behaviour and sensor response | `PTC.py`, PTC plots |
| Bias and dark frames | Baseline signal and dark-frame behaviour | `bias.py`, `dark.py`, dark-analysis outputs |
| Dark current | Signal variation with exposure | `dark_current.py`, summary and plots |
| Fixed-pattern noise (FPN) | Spatial noise structure across frames | `FPN.py`, 100-frame analysis output |
| Flat-field response | PRNU and illumination non-uniformity | `flat_field.py`, flat-field outputs |
| Hot pixels | Pixel-level outliers in dark frames | `hot_pixel.py`, pixel map and CSV |
| Linearity | Response consistency across signal levels | `linearity.py`, results CSV and plot |
| Dynamic range | Usable signal range relative to noise | `dynamic_range.py`, metrics and plots |

## Results gallery

The figures below are generated outputs already included in this repository.

### Photon Transfer Curve

<img src="Camera_characterization/PTC/ptc_analysis.png" alt="Photon Transfer Curve analysis" width="82%" />

### Fixed-pattern noise

<img src="results/FPN/fpn_analysis.png" alt="Fixed-pattern noise analysis" width="82%" />

### Dark-frame and dark-current analysis

<p align="center">
  <img src="results/dark/dark_analysis_plots.png" alt="Dark-frame analysis plots" width="82%" />
</p>
<p align="center">
  <img src="results/dark/dark_current_analysis.png" alt="Dark-current analysis" width="82%" />
</p>

### Flat-field response

<img src="results/flat_field_analysis/flat_analysis_plots.png" alt="Flat-field analysis plots" width="82%" />

### Dynamic range

<img src="results/dynamic_range/dynamic_range_analysis.png" alt="Dynamic-range analysis" width="82%" />

### Hot-pixel detection

<p align="center">
  <img src="results/hot_pixel/hot_pixel_analysis.png" alt="Hot-pixel analysis" width="82%" />
</p>
<p align="center">
  <img src="results/hot_pixel/hot_pixel_map.png" alt="Hot-pixel map" width="62%" />
</p>

### Linearity

<img src="results/linearity/linearity_analysis.png" alt="Linearity analysis" width="82%" />

## Selected recorded outputs

The following values are copied from the text/JSON summaries currently stored in `results/`. They are presented as recorded outputs, not as independently validated sensor specifications.

| Analysis | Recorded output |
|---|---:|
| FPN analysis | 100 frames; reported PRNU: 44.91% |
| Flat-field analysis | Average signal: 47,770.52 DN |
| Flat-field analysis | Reported average PRNU: 31.42% |
| Flat-field analysis | Reported maximum vignetting: 89.13% |
| Hot-pixel analysis | 14 hot pixels detected in 400 dark frames |

> **Interpretation note:** The repository contains multiple analysis outputs and metric definitions. Values should be interpreted in the context of the corresponding script, dataset, exposure and calculation method. In particular, the two PRNU values above come from different result summaries and should not be treated as interchangeable.

## Tools and data

- **Python** for analysis workflows
- **FITS / TIFF** image datasets
- **NumPy, Matplotlib** and related scientific Python tooling (as used by the analysis scripts)
- **Teledyne FLIR SpinView** for camera acquisition workflow context

## Repository structure

```text
Integrating_sphere/
├── Camera_characterization/
│   ├── FNP/                 # FPN image sequence
│   ├── PTC/                 # PTC figure
│   └── python/              # Characterization scripts
├── results/
│   ├── FPN/
│   ├── PTC/
│   ├── bias/
│   ├── dark/
│   ├── dynamic_range/
│   ├── flat_field_analysis/
│   ├── hot_pixel/
│   └── linearity/
├── assets/                  # Repository presentation assets
└── README.md
```

## About the project

**Author:** Palak Prabhat  
**Institution:** Indian Institute of Technology Kanpur  
**Project area:** Camera sensor performance characterization

---

<div align="center">
  <sub>Built to document experimental work, analysis workflows and sensor-characterization results.</sub>
</div>
