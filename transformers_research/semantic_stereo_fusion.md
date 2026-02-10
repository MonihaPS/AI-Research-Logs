# Semantic Stereo Fusion (FoundationStereo + SAM 2)

## 📖 Project Overview

This project integrates **FoundationStereo** (for high-quality depth estimation) and **SAM 2** (Segment Anything Model 2) to create a Semantic 3D Point Cloud.

The pipeline takes a pair of stereo images (Left/Right), computes disparity/depth, segments objects using SAM 2, and fuses this data to measure the distance of specific objects in the scene.

### Key Features

- **Stereo Depth Estimation**: Uses FoundationStereo with Hiera-Large backbone.
- **Semantic Segmentation**: Uses SAM 2.1 for state-of-the-art instance segmentation.
- **Sensor Fusion**: Overlays distance metrics onto segmented objects.
- **3D Reconstruction**: Generates colored point clouds (`.ply`) for visualization.

## ⚙️ Hardware Requirements

- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA RTX 3060/4060 or higher recommended (8GB+ VRAM).

> [!NOTE]
> Testing on RTX 4050 (6GB) encountered significant memory fragmentation issues due to the size of the ViT-Large backbone.

- **RAM**: 16GB minimum (32GB recommended).
- **CUDA**: 11.8 or 12.1.

## 🛠️ Installation Guide

### 1. Environment Setup

It is recommended to use `conda` to manage dependencies.

```bash
# Create environment
conda create -n foundation_stereo python=3.10 -y
conda activate foundation_stereo

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Dependencies

Install the required Python libraries:

```bash
pip install opencv-python matplotlib numpy pandas imageio trimesh einops joblib timm
```

### 3. Clone Repositories

This project relies on two major sub-modules. Ensure they are placed in the correct directory structure.

- **FoundationStereo**: [Official Repo](https://github.com/NVlabs/FoundationStereo)
- **SAM 2**: [Meta SAM 2 Repo](https://github.com/facebookresearch/segment-anything-2)

## 📂 Project Structure

Ensure your folder structure looks exactly like this before running the script:

```plaintext
/Project_Root
│
├── FoundationStereo/           # (Cloned Repo)
│   ├── scripts/
│   │   └── run_demo.py         # Main inference script
│   ├── assets/                 # Input Images
│   │   ├── left.jpg
│   │   ├── right.jpg
│   │   └── camera_params.txt
│   ├── pretrained_models/      # Weights
│   │   └── model_best_bp2.pth
│   └── semantic_stereo_fusion.py  <-- OUR MAIN SCRIPT
│
└── sam2_project/
    └── sam2_repo/              # (Cloned Repo)
        ├── checkpoints/
        │   └── sam2.1_hiera_large.pt
        └── configs/
```

## 🚀 Usage

Run the main fusion pipeline from the `FoundationStereo` directory:

```bash
python semantic_stereo_fusion.py
```

### The Pipeline Steps:

1.  **Device Check**: Automatically detects the NVIDIA GPU and configures memory allocation.
2.  **Depth Estimation**: Runs `run_demo.py` to generate `disp.npy` (Disparity Map) and `cloud.ply` (Point Cloud).
3.  **Segmentation**: Loads SAM 2 to generate instance masks for the left image.
4.  **Fusion**: Combines masks and depth maps to calculate the median distance of every detected object.

## ⚠️ Known Issues & Troubleshooting

### 1. CUDA OutOfMemory (OOM) on 6GB Cards

- **Symptom**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 6.00 GiB...`
- **Root Cause**: The Hiera-Large backbone used by FoundationStereo is extremely VRAM intensive. On 6GB cards (like RTX 3050/4050), PyTorch suffers from Memory Fragmentation. Even if 4GB is technically "free," there is no continuous block large enough for the model's activation layers.

**Attempted Fixes (Documented):**

- **Memory Mapping (`mmap=True`)**: Applied to `torch.load` to reduce RAM spikes.
- **Variable Expansion**: Used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Downscaling**: Reduced input scale to 0.25.
- **Iteration Reduction**: Reduced inference iterations from 16 to 6.

**Resolution**: If OOM persists on 6GB cards, the specific FoundationStereo step must be run on CPU or a cloud GPU (Colab/RunPod) with 12GB+ VRAM.

### 2. GPU Index Mismatch

- **Symptom**: Script runs on Intel Integrated Graphics instead of NVIDIA.
- **Fix**: Windows Task Manager often lists NVIDIA as GPU 1, but CUDA lists it as GPU 0. The script explicitly sets:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

## 📊 Outputs

All results are saved in the `outputs/` directory:

-   `outputs/stereo/`: Raw disparity maps (`.npy`) and point clouds (`.ply`).
-   `outputs/segmentation/`: Visualized masks from SAM 2.
-   `outputs/fusion/`: Final image with object distances overlaid.
-   `outputs/3d/`: Cleaned 3D Point Cloud files ready for MeshLab.