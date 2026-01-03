# Pyramid Vision Transformer (PVT): From Scratch (PyTorch)

This repository contains a **clean, from-scratch implementation of the Pyramid Vision Transformer (PVT)** in PyTorch.

The goal of this project is **architectural understanding**, not reproducing benchmark numbers. The implementation focuses on **hierarchical token processing and efficient attention**, which are the core ideas behind PVT.

---

##  What is Pyramid Vision Transformer?

Vanilla Vision Transformers (ViT) process images using a **single-scale, flat token representation**, which works well for classification but struggles with:
- Object-level reasoning
- Localization
- Change detection
- Dense vision tasks

**PVT fixes this by introducing a pyramid (hierarchical) structure**, similar to CNN backbones.

### Key idea:
> Instead of keeping the same resolution throughout the network, PVT gradually reduces spatial resolution while increasing semantic richness.

---

## Core Concepts Implemented

This implementation includes the **essential building blocks of PVT**:

### 1. Hierarchical Patch Embedding
- Each stage downsamples the input
- Spatial resolution decreases stage by stage
- Channel dimension increases

### 2. Spatial Reduction Attention (SRA)
- Queries operate at full resolution
- Keys and values are spatially downsampled
- Reduces attention complexity from O(N²) to near-linear

### 3. Multi-Stage Transformer Encoder
- Each stage has its own transformer blocks
- Mimics CNN feature pyramids (low → mid → high level features)

---

## Architecture Overview

```
Input Image
   ↓
Stage 1: High-resolution tokens (fine details)
   ↓
Stage 2: Medium-resolution tokens
   ↓
Stage 3: Low-resolution semantic tokens
   ↓
Stage 4: Global semantic representation
```

Each stage outputs token features that can be used independently or fused for downstream tasks.

---

## Output of the Model

The model returns a **list of multi-scale token representations**:

- Stage 1: Fine-grained spatial tokens
- Stage 2: Mid-level object features
- Stage 3: High-level semantic features
- Stage 4: Global scene representation

These outputs are ideal for:
- Change detection
- Object-level reasoning
- Scene understanding
- Vision-language grounding

---

## Why this implementation matters

This project is useful if you want to:
- Understand how ViT evolves into practical vision backbones
- Learn how attention can be made efficient for high-resolution images
- Build modern vision architectures for research projects
- Use PVT as a backbone for tasks beyond classification

---

##  What this repository does NOT include

- Pretrained weights
- Training scripts
- Dataset loaders
- Benchmark comparisons

This is intentional, the focus is **architecture clarity and learning**.

---

## Learning Takeaway

> Pyramid Vision Transformer shows that **hierarchy + attention** is essential for real-world vision problems.

If you understand this code, you understand **why modern vision transformers look the way they do**.

---

## Disclaimer

This is an **educational / research-oriented implementation** inspired by the PVT paper. It is not intended to reproduce official results.

---

