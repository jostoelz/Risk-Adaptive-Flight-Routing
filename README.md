# 📘 Risk Adaptive Flight Routing
## 🔗 Online Access

- Website of the project: https://risk-adaptive-flight-routing.streamlit.app/
- This project was featured in the following thesis: <https://drive.google.com/drive/folders/1ag495ByeFjiVlfqUUsdaL6ZCu044Agzx>
- GitHub Repository of the thesis: <https://github.com/jostoelz/Autonomous-Drone-System-for-Wolf-Detection-Deterrence-and-Sheep-Protection>

## 🔍 Abstract
This repository implements a production-grade, highly adaptive **Receding Horizon Control (RHC)** and **Markov Decision Process (MDP)** path-planning pipeline designed for autonomous UAVs to protect livestock herds against apex predators (e.g., wolves). Leveraging real-time **Gaussian Kernel Density Estimation (KDE)**, the system synthesizes dynamic biological tracking matrices and environment-specific threat vectors—such as static forest edge proximities—into a unified risk surface. 

The core software architecture dynamically alternates between three major operational profiles depending on livestock dispersion:
1. **Philosophy A ('The Bodyguard Mode'):** Generates highly continuous, tight cubic B-spline orbits directly centered over compact herds, ensuring a localized digital shield.
2. **Dynamic Split-Herd Hysteresis:** Combines Singular Value Decomposition (SVD) principal axis analysis with a strict geometric hysteresis lock ($>45\text{ m}$ separation thresholds) to decisively secure sub-clusters without control-induced jitter or mid-gap chattering. An automated high-weight wolf-override protocol ensures rapid cross-pasture interception during active threats.
3. **Phase-Shifting Area Coverage:** Automatically transitions into a macro-scale sweeping patrol using a time-dependent, phase-shifting Lissajous (Figure-8) trajectory during highly scattered herd states, wiping out horizontal blind spots and guaranteeing $100\%$ spatial coverage over time.

The entire control loop is fully constrained by physical hardware edge parameters, mapping real-world Camera Field-of-View (FoV) geometry, Raspberry Pi object detection inference rates, and minimum wolf pixel sensor requirements directly onto variable operational flight altitudes and speeds. The project features a fully containerized, real-time interactive Streamlit dashboard backed by a high-performance FastAPI/Render backend for live-telemetry visualization.

## 📖 Citation

If you find this project useful for your research, please consider citing it:

```bibtex
@thesis{Stoelzle2026Development,
  author      = {St{\"o}lzle, Johannes},
  title       = {Development of an Autonomous Drone System for Wolf Perception, Deterrence, and Livestock Protection},
  institution = {Kantonsschule Romanshorn},
  year        = {2026},
  type        = {Matura Thesis},
  url         = {https://github.com/jostoelz/Autonomous-Drone-System-for-Wolf-Detection-Deterrence-and-Sheep-Protection}
}
```

## 🧭 Visual Overview

<p align="center">
  <img src="Demonstration_Image_1.jpg" alt="Multi-object detection of sheep in an open field." width="900"/>
  <br/>
  <em>Multi-object detection of sheep in an open field.</em>
</p>

<p align="center">
  <img src="Demonstration_Image_2.jpg" alt="Wolf detection in a wildlife park enclosure." width="900"/>
  <br/>
  <em>Wolf detection in a wildlife park enclosure.</em>
</p>

## 📜 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
✨ Enjoy exploring the thesis materials. 
