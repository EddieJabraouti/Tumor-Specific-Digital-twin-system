# Glioblastoma Digital Twin System

> A modular pipeline for generating tumor-specific digital twins from multimodal MRI data using deep learning and physics-inspired simulation.

[![Research Project](https://img.shields.io/badge/Status-Research-orange)](https://github.com)
[![License](https://img.shields.io/badge/License-Personal_Research-lightgrey)](https://github.com)

## ⚠️ Disclaimer

**This is a research project and is not intended for clinical use.** This tool is for research purposes only and should not be used for clinical diagnosis or treatment decisions.

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Pipeline Architecture](#pipeline-architecture)
- [Components](#components)
  - [Segmentation Model](#segmentation-model)
  - [Generative Model](#generative-model)
  - [3D Reconstruction](#3d-reconstruction)
  - [Physics Simulation](#physics-simulation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Evaluation & Validation](#evaluation--validation)
- [Citation & References](#citation--references)

---

## Overview

The **Glioblastoma Digital Twin System** is a comprehensive research pipeline that generates anatomically realistic 3D digital representations of glioblastoma multiforme (GBM) tumors from multimodal MRI data. The system integrates deep learning models for segmentation and generation with physics-based simulation to enable spatiotemporal visualization of therapeutic agent distribution.

### Key Features

- **Multimodal MRI Processing**: Supports T1, T1ce, T2, and FLAIR sequences
- **Automated Tumor Segmentation**: U-Net/nnU-Net based 4-class segmentation
- **Synthetic MRI Generation**: Conditional diffusion models for anatomically plausible data generation
- **3D Tumor Reconstruction**: High-fidelity mesh generation from segmentation masks
- **Therapeutic Agent Simulation**: Physics-informed neural networks for drug distribution modeling
- **Interactive Web Interface**: Modern React-based frontend for pipeline orchestration and visualization

---

## Project Goals

The primary objective of this project is to create a synthetic, MRI-derived 3D digital twin of glioblastoma that enables:

1. **Anatomically Faithful Representation**: High-fidelity 3D models that preserve tumor morphology and spatial relationships
2. **Segmentation Consistency**: Robust segmentation performance across real and synthetic MRI data
3. **Simulation-Ready Structures**: Spatial meshes optimized for computational fluid dynamics and transport modeling
4. **Interactive Visualization**: Real-time exploration of therapeutic agent distribution over time
5. **Research Tool**: Extensible framework for hypothesis testing and treatment strategy exploration

This framework emphasizes **reproducibility**, **interpretability**, and **computational feasibility** while remaining clinically relevant in research contexts.

---

## Pipeline Architecture

The pipeline is modular and consists of six sequential stages:

```
┌─────────────────┐
│  MRI Input      │  T1, T1ce, T2, FLAIR (+ optional mask)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Skull stripping, co-registration, normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Segmentation   │  U-Net/nnU-Net → 4-class mask
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Diffusion      │  Conditional generation → Synthetic MRI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Re-segmentation│  Validation on synthetic data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3D Modeling    │  Mesh generation (GLB/OBJ)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Simulation     │  Therapeutic agent distribution
└─────────────────┘
```

### Pipeline Stages

1. **Preprocessing**: Standardized MRI data preparation
2. **Segmentation**: Tumor region identification and classification
3. **Diffusion Generation**: Synthetic MRI volume synthesis
4. **Re-segmentation**: Validation of synthetic data quality
5. **3D Reconstruction**: Mesh generation from segmentation masks
6. **Simulation**: Physics-based therapeutic agent modeling

---

## Components

### Segmentation Model

**Architecture**: U-Net or nnU-Net  
**Purpose**: Establish anatomical foundation for digital twin

#### Input
- Multimodal MRI volumes (T1, T1ce, T2, FLAIR)

#### Output
4-class voxel-wise segmentation mask:
- **Background**: Normal brain tissue
- **Edema**: Peritumoral edema region
- **Non-enhancing Tumor**: Necrotic core / non-enhancing tumor
- **Enhancing Tumor**: Active tumor region with contrast enhancement

#### Key Characteristics
- Voxel-wise classification with spatial consistency
- Handles multimodal input fusion
- Produces masks suitable for 3D reconstruction

### Generative Model

**Architecture**: Conditional Diffusion Model  
**Purpose**: Generate anatomically plausible synthetic MRI volumes

#### Conditioning
- Tumor segmentation masks (required)
- Original MRI volumes (optional, configurable)
- Both mask and volume conditioning (configurable)

#### Output
- Synthetic MRI volumes (T1, T1ce, T2, FLAIR)
- Anatomically realistic GBM morphology
- Preserved tissue contrast patterns

#### Key Characteristics
- Conditional generation ensures tumor geometry preservation
- Generates realistic multi-modal MRI sequences
- Maintains spatial alignment with input segmentation

### 3D Reconstruction

**Purpose**: Convert segmentation masks into simulation-ready 3D meshes

#### Input
- Segmentation mask (from primary or re-segmentation step)

#### Output
- 3D mesh model (GLB or OBJ format)
- Vertex and face statistics
- Volume measurements

#### Applications
- Visualization and inspection
- Computational fluid dynamics meshing
- Therapeutic agent transport modeling

### Physics Simulation

**Architecture**: Physics-Informed Neural Network (PINN)  
**Purpose**: Model therapeutic agent distribution over time

#### Input
- 3D tumor model
- Segmentation mask (optional, for tissue type mapping)
- Simulation parameters (diffusion coefficient, decay rate, etc.)

#### Output
- Time-series distribution maps
- Frame-by-frame agent concentration data
- Spatiotemporal visualization data

#### Key Characteristics
- Incorporates physics constraints (diffusion, advection, decay)
- Handles complex boundary conditions
- Produces interpretable temporal dynamics

---

## Dataset

### Source
**BraTS (Brain Tumor Segmentation Challenge)**  
Annual competition dataset with standardized glioblastoma annotations

### Modalities
- **T1**: T1-weighted structural MRI
- **T1ce**: T1-weighted with gadolinium contrast enhancement
- **T2**: T2-weighted structural MRI
- **FLAIR**: Fluid-attenuated inversion recovery

### Preprocessing Pipeline

All MRI data undergo standardized preprocessing to ensure consistency:

1. **Skull Stripping**: Removal of non-brain tissue
2. **Multimodal Co-registration**: Spatial alignment across sequences
3. **Intensity Normalization**: Standardization of signal intensities
4. **Resampling**: Consistent voxel resolution across all volumes

These preprocessing steps ensure:
- Spatial alignment between modalities
- Stable training behavior across models
- Consistent downstream processing

### Dataset Characteristics
- Preprocessed volumes: Spatially aligned and normalized
- Ground truth: Expert-annotated 4-class segmentation masks
- Quality: Validated through BraTS challenge framework

---

## Project Structure

```
TUMOR-SPECIFIC-DIGITAL-TWIN/
│
├── Diffusion/
│   └── diffusion.py              # Diffusion model implementation
│
├── UNet/
│   └── net.py                    # U-Net/nnU-Net architecture
│
├── Physics Based Simulation/
│   └── nn.py                     # Physics-informed NN for simulation
│
├── notebooks/
│   ├── diffusion.ipynb           # Diffusion model development
│   ├── phyNN.ipynb               # Physics simulation experiments
│   └── Unet.ipynb                # Segmentation model training
│
├── frontend/                      # Web application interface
│   ├── src/
│   │   ├── api/                  # API client and mock services
│   │   ├── components/           # React UI components
│   │   ├── state/                # State management (Zustand)
│   │   └── App.tsx               # Main application
│   ├── package.json
│   ├── vite.config.ts
│   └── README.md                 # Frontend documentation
│
└── README.md                      # This file
```

### Component Descriptions

- **Diffusion/**: Conditional diffusion model for synthetic MRI generation
- **UNet/**: Segmentation model architecture and training scripts
- **Physics Based Simulation/**: Physics-informed neural network for therapeutic agent simulation
- **notebooks/**: Jupyter notebooks for model development, training, and experimentation
- **frontend/**: React + TypeScript web application for pipeline orchestration and visualization

---

## Setup & Usage

### Prerequisites

- Python 3.8+ (for model training and inference)
- Node.js 18+ (for frontend web application)
- GPU recommended for model training (CUDA-compatible)

### Installation

#### Backend Models

```bash
# Install Python dependencies (create virtual environment recommended)
pip install -r requirements.txt

# Additional setup as needed for:
# - PyTorch/TensorFlow (depending on model implementation)
# - Medical imaging libraries (nibabel, SimpleITK, etc.)
# - Diffusion model dependencies
```

#### Frontend Web Application

```bash
cd frontend
npm install
npm run dev    # Development server at http://localhost:5173
npm run build  # Production build
```

See [`frontend/README.md`](frontend/README.md) for detailed frontend documentation.

### Usage

#### Web Interface (Recommended)

1. Launch the frontend web application
2. Upload multimodal MRI files (T1, T1ce, T2, FLAIR)
3. Select pipeline steps to execute
4. Monitor progress through the interactive dashboard
5. View results in the outputs panel

#### Programmatic Usage

```python
# Example usage (implementation-specific)
from UNet.net import UNet
from Diffusion.diffusion import ConditionalDiffusion

# Load models
segmenter = UNet.load_from_checkpoint(...)
generator = ConditionalDiffusion.load_from_checkpoint(...)

# Run pipeline
mask = segmenter.segment(mri_volumes)
synthetic_mri = generator.generate(mask, mri_volumes)
```

---

## Evaluation & Validation

Evaluation focuses on **pipeline consistency** and **anatomical plausibility** rather than clinical prediction accuracy.

### Validation Metrics

1. **Segmentation Consistency**
   - Dice coefficient between primary and re-segmentation masks
   - Class-wise overlap metrics
   - Spatial distribution analysis

2. **Synthetic Data Quality**
   - Visual inspection of anatomical realism
   - Intensity distribution comparison
   - Tumor morphology preservation

3. **3D Model Quality**
   - Mesh smoothness and topology
   - Volume preservation from segmentation to mesh
   - Geometric accuracy metrics

4. **Simulation Validation**
   - Physics constraint satisfaction
   - Temporal consistency checks
   - Boundary condition compliance

### Key Validation Steps

- **Visual Inspection**: Expert review of synthetic MRI realism
- **Re-segmentation**: Segmentation of synthetic data using trained models
- **Morphology Comparison**: Quantitative comparison of tumor shape between real and synthetic outputs
- **Class Coherence**: Verification that segmentation classes are preserved across pipeline stages

---

## Citation & References

### Datasets

- **BraTS**: [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats/)
  - Menze, B.H., et al. "The multimodal brain tumor image segmentation benchmark (BRATS)." IEEE TMI (2015)

### Key Methodologies

- **U-Net**: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI (2015)
- **nnU-Net**: Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021)
- **Diffusion Models**: Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS (2020)
- **Physics-Informed Neural Networks**: Raissi, M., et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics (2019)

---

## Future Work

- Integration of additional imaging modalities
- Enhanced 3D visualization capabilities
- Real-time simulation optimization
- Extended validation on external datasets
- Multi-patient batch processing support
- Advanced therapeutic agent models

---

## License & Acknowledgments

**This is a personal research project and is not intended for clinical use.**
