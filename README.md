Glioblastoma Digital Twin – Project README
1. Pipeline Overview

This project implements a tumor-specific digital twin pipeline for glioblastoma (GBM) using multimodal MRI data, generative modeling, and physics-inspired simulation concepts.
The system produces a synthetic, anatomically realistic 3D tumor representation and enables spatiotemporal visualization of therapeutic agent distribution within tumor and peritumoral tissue.

The pipeline is modular and consists of:

MRI preprocessing

Tumor segmentation

Conditional generative modeling (diffusion)

Synthetic data validation via re-segmentation

3D reconstruction

Interactive visualization

The design emphasizes reproducibility, interpretability, and clinical relevance, while remaining computationally feasible.

2. Model Rundown
Segmentation Model

Architecture: U-Net or nnU-Net

Input: Multimodal MRI volumes (T1, T1ce, T2, FLAIR)

Output: 4-class voxel-wise segmentation mask

Background

Edema

Non-enhancing tumor / necrosis

Enhancing tumor

This model establishes the anatomical foundation for the digital twin.

Generative Model

Architecture: Conditional diffusion model

Conditioning:

Tumor segmentation masks

MRI volumes

Or both (configurable)

Output: Synthetic MRI volumes containing realistic GBM morphology

The diffusion model generates anatomically plausible tumor-bearing MRI data aligned with learned tumor structure.

3. Purpose

The primary objective is to create a synthetic, MRI-derived 3D digital twin of glioblastoma that enables:

Anatomically faithful tumor representation

Consistent tumor segmentation across real and synthetic data

Simulation-ready spatial structures for modeling therapy transport

Interactive visualization of therapeutic spread, uptake, and decay

This framework is intended as a research and exploratory tool, supporting hypothesis testing and treatment strategy exploration rather than direct clinical deployment.

4. Dataset

Source: BraTS (Brain Tumor Segmentation Challenge)

Modalities Used:

T1

T1ce

T2

FLAIR

Preprocessing Steps

All MRI data undergo standardized preprocessing:

Skull stripping

Multimodal co-registration

Intensity normalization

Resampling to a consistent voxel resolution

These steps ensure spatial alignment and stable training behavior across models.

5. Training
Segmentation Training

Train the U-Net / nnU-Net on preprocessed BraTS data

Optimize for multi-class voxel-wise accuracy

Output segmentation masks used both independently and as conditioning inputs

Diffusion Model Training

Train a conditional diffusion model using:

Real MRI volumes

Corresponding segmentation masks

The model learns to generate MRI volumes that preserve tumor geometry and tissue contrast patterns

Training is staged to ensure segmentation performance is established before generative modeling.

6. Test Results

Evaluation focuses on pipeline consistency and anatomical plausibility, not clinical prediction.

Key validation steps include:

Visual inspection of synthetic MRI realism

Re-segmentation of synthetic MRI using the trained U-Net

Comparison of tumor morphology and class coherence between real and synthetic outputs

This ensures the synthetic data remains compatible with downstream processing and visualization.

7. Final Overview

This project delivers a minimal but extensible GBM digital twin pipeline that integrates:

Real MRI-derived tumor anatomy

Synthetic MRI generation via diffusion models

Robust segmentation consistency

3D reconstruction and visualization readiness

Purely a personal research project - not intended for clinical use

