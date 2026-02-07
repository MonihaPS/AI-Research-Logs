# FoundationStereo – My Complete Deployment Log (Windows + Arducam IMX708)

This document lives inside my repo:  
**AI-Research-Logs/transformers_research/**

---

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Pretrained Models I Use](#pretrained-models-i-use)
- [Running on Official Samples](#running-on-official-samples)
- [Running on My Arducam Stereo Images](#running-on-my-arducam-stereo-images)
- [Camera Calibration & camera_params.txt](#camera-calibration--camera_paramstxt)
- [Outputs & How to View Them](#outputs--how-to-view-them)
- [References](#references)

---

## Installation & Setup

Tested on Windows 11 with NVIDIA RTX 4050 Laptop GPU.

```powershell
# 1. Clone the base repo (I started here)
cd D:\moni\bluvern
git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git
cd FoundationStereo

# 2. Fix for Windows – disable xformers
# Open environment.yml and comment this line:
# - xformers==0.0.28.post1

# 3. Use faster solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# 4. Create my environment
conda env create -f environment.yml -n stereo_depth
conda activate stereo_depth

# 5. Install flash-attn separately
pip install flash-attn --no-build-isolation

# 6. Install CUDA PyTorch & helper libraries
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pandas matplotlib seaborn tqdm pillow opencv-contrib-python
```
---

## Pretrained Models I Use

## Pretrained Models Used in This Project

| Model Name   | Description                              | Accuracy / Speed Trade-off          | Download Link                                                                                           | Saved Location in Project                              |
|--------------|------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| ViT-Large    | Best performing model (higher accuracy)  | Highest accuracy, more VRAM & slower| [Google Drive - 23-51-11](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) | `pretrained_models/23-51-11/model_best_bp2.pth`       |
| ViT-Small    | Faster inference, lower memory usage     | Slightly lower accuracy, faster     | [Google Drive - 11-33-40](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) | `pretrained_models/11-33-40/model_best_bp2.pth`       |

**How to use:**
- Download the `.pth` file from the respective Google Drive folder.
- Place it exactly in the folder path shown in the "Saved Location" column.
- Use the corresponding `--ckpt_dir` path in the run command (example shown below).

**Example run commands (choose one):**

```powershell
# Using ViT-Large (best quality)
--ckpt_dir .\pretrained_models\23-51-11\model_best_bp2.pth

# Using ViT-Small (faster, less VRAM)
--ckpt_dir .\pretrained_models\11-33-40\model_best_bp2.pth
```
---

## Running on Official Samples

```PowerShell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python scripts/run_demo.py `
  --left_file .\assets\left.png `
  --right_file .\assets\right.png `
  --ckpt_dir .\pretrained_models\23-51-11\model_best_bp2.pth `
  --out_dir .\outputs_official `
  --hiera 1 --valid_iters 32 --get_pc 1 --z_far 100.0 --denoise_cloud 1
```
---

## Running on My Arducam Stereo Images

```PowerShell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python scripts/run_demo.py `
  --left_file .\assets\left.jpg `
  --right_file .\assets\right.jpg `
  --intrinsic_file .\assets\camera_params.txt `
  --ckpt_dir .\pretrained_models\23-51-11\model_best_bp2.pth `
  --out_dir .\outputs_my_arducam `
  --scale 0.7 `
  --hiera 1 `
  --valid_iters 32 `
  --get_pc 1 `
  --z_far 50.0 `
  --denoise_cloud 1 `
  --remove_invisible 1
```
---

## Camera Calibration & camera_params.txt

My current working values (approximate for Arducam IMX708 Wide at ~1296×972):
```text
1001.5 0.0 648.0 0.0 1001.5 486.0 0.0 0.0 1.0
0.063
```

**Format explanation:**

Line 1: fx 0 cx 0 fy cy 0 0 1 (left camera intrinsics)

Line 2: Baseline in meters (distance between lens centers)

**For best accuracy** — I recommend full stereo calibration:

1.Print 7×10 chessboard pattern

2.Capture 20–30 simultaneous left/right pairs

3.Run OpenCV stereoCalibrate to get exact values

4.Update camera_params.txt

---

## Outputs & How to View Them

**vis.png** → Original image + disparity visualization (brighter = closer)

**cloud.ply** → Raw 3D point cloud

**cloud_denoise.ply** → Cleaner version after denoising


**Best viewers:**

**MeshLab** – lightweight & excellent for point clouds

**CloudCompare** – powerful analysis tools

---

## References

Original FoundationStereo team: Bowen Wen et al. (NVIDIA)

Paper: https://arxiv.org/abs/2501.09898

Official Repo: https://github.com/NVlabs/FoundationStereo

Arducam IMX708 Wide: https://www.arducam.com/product/arducam-for-raspberry-pi-camera-module-3-wide-imx708-manual-focus/

OpenCV Stereo Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

Project Demo Images & Results (#outputs--visualization)  
