# SAM 2: Advanced Automatic Object Segmentation & Detection

![SAM 2 Banner](https://github.com/facebookresearch/sam2/raw/main/assets/sam2_logo.png)

## 🏗️ Project Overview

This repository hosts a deployment of the **Segment Anything Model 2 (SAM 2)**, specifically tuned for **high-fidelity automatic object segmentation**. The project is designed to process static images and video frames to detect, segment, and visualize every distinct object in a scene—ranging from large structural elements (like keyboards and furniture) to small, handheld items (like boxes, cups, or tools).

Unlike the standard "out-of-the-box" implementation, this deployment features a **custom-tuned inference script** (`auto_segment_tuned.py`). This script utilizes a "Dense Search" configuration to overcome common segmentation challenges, such as:

*   **Over-segmentation:** Preventing a single object (e.g., a keyboard) from being broken into hundreds of tiny masks (individual keys).
*   **Under-segmentation:** Ensuring small objects held in hands or placed in cluttered backgrounds are not missed.
*   **Edge Precision:** Refining boundaries to separate overlapping objects (e.g., a box vs. a shirt).

This project uses the **SAM 2.1 Hiera Large** model checkpoint for maximum accuracy and is optimized for CUDA-enabled GPU environments.

---

## 📋 Table of Contents

1.  [Project Overview](#-project-overview)
2.  [Prerequisites & System Requirements](#-prerequisites--system-requirements)
3.  [Installation Guide](#-installation-guide)
    *   [Cloning the Repository](#step-1-cloning-the-repository)
    *   [Setting Up the Virtual Environment](#step-2-setting-up-the-virtual-environment)
    *   [Installing Dependencies](#step-3-installing-dependencies)
    *   [Downloading Model Checkpoints](#step-4-downloading-model-checkpoints)
4.  [Project Structure](#-project-structure)
5.  [Configuration Strategy (The "Why")](#-configuration-strategy-the-why)
    *   [Parameter Tuning Guide](#parameter-tuning-guide)
6.  [Usage Instructions](#-usage-instructions)
    *   [Running the Segmentation](#running-the-segmentation)
    *   [Understanding the Output](#understanding-the-output)
7.  [Troubleshooting & Common Issues](#-troubleshooting--common-issues)
8.  [Advanced Visualization](#-advanced-visualization)
9.  [License & Acknowledgements](#-license--acknowledgements)

---

## 💻 Prerequisites & System Requirements

Before setting up the project, ensure your system meets the following requirements to run the SAM 2 Large model effectively.

### **Hardware**

*   **GPU:** NVIDIA GPU with at least **8GB VRAM** (Recommended: 16GB+ for 4K images).
    *   *Note: CPU execution is supported but will be significantly slower (minutes vs. seconds).*
*   **RAM:** 16GB system RAM minimum.
*   **Storage:** At least 5GB of free space for model checkpoints and environment files.

### **Software**

*   **Operating System:** Windows 10/11 (with PowerShell) or Linux (Ubuntu 20.04+).
*   **Python:** Version **3.10** or **3.11** is recommended.
*   **CUDA:** CUDA Toolkit 11.8 or 12.1 (Must match your PyTorch installation).
*   **Git:** Required for cloning the repository.

---

## 🛠️ Installation Guide

Follow these steps strictly to set up a conflict-free environment.

### Step 1: Cloning the Repository

First, clone the official SAM 2 repository to your local machine.

```bash
# Clone the repository
git clone https://github.com/facebookresearch/sam2.git

# Navigate into the project folder
cd sam2
```

### Step 2: Setting Up the Virtual Environment

It is critical to use a virtual environment to prevent conflicts with other Python projects.

#### For Windows (PowerShell):

```powershell
# Create the virtual environment named 'venv'
python -m venv venv

# Activate the environment
.\venv\Scripts\activate
```

#### For Linux / Mac:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

You should see `(venv)` appear at the start of your command line prompt.

### Step 3: Installing Dependencies

We need to install PyTorch first, ensuring it matches your CUDA version.

1.  **Install PyTorch (CUDA 11.8 version):**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

2.  **Install Basic Libraries:**
    ```bash
    pip install matplotlib opencv-python jupyter pillow
    ```

3.  **Install SAM 2 in Editable Mode:**
    ```bash
    pip install -e .
    ```

> **CRITICAL FIX:** If you encounter an error related to `sympy` or `mpmath` during installation or execution, run the following command to force a clean reinstall of the math libraries:
> ```bash
> pip uninstall sympy mpmath -y
> pip install sympy mpmath
> ```

### Step 4: Downloading Model Checkpoints

The model weights are not included in the pip install. You must download them manually using the provided script.

#### For Linux / Git Bash:

```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

#### For Windows (Manual Download):

If the shell script does not work, download the following file manually and place it inside the `sam2/checkpoints/` folder:

*   **URL:** `https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt`
*   **Target File:** `sam2.1_hiera_large.pt`

---

## 📂 Project Structure

Here is how your project directory should look after installation:

```plaintext
sam2_project/
├── checkpoints/
│   ├── sam2.1_hiera_large.pt      # The heavy model weights file
│   ├── sam2.1_hiera_base_plus.pt
│   └── ...
├── configs/
│   ├── sam2.1/
│   │   ├── sam2.1_hiera_l.yaml    # Config file for Large model
│   │   └── ...
├── sam2/                          # Source code for the library
├── venv/                          # Your virtual environment
├── auto_segment_tuned.py          # <--- MAIN SCRIPT (Run this)
├── test_image.png                 # <--- YOUR INPUT IMAGE
├── dense_result1.png              # <--- OUTPUT RESULT
└── README.md                      # This documentation file
```

---

## ⚙️ Configuration Strategy (The "Why")

The core value of this deployment lies in the specific parameter tuning of the `SAM2AutomaticMaskGenerator`. The default settings are often too sparse for small objects or too aggressive for complex textures (like keyboards).

We utilize a "Dense Search" strategy. Below is a detailed breakdown of every parameter modified in `auto_segment_tuned.py`.

### Parameter Tuning Guide

| Parameter | Value | Description & Reasoning |
| :--- | :--- | :--- |
| **points_per_side** | `64` | **The Grid Density.** Default: 32. We doubled this to 64. This means the model places a grid of 64x64 (4,096) points across the image to probe for objects. <br><br> **Benefit:** drastically increases the chance of hitting small objects (like a box in a hand) that might fall between the cracks of a looser grid. |
| **crop_n_layers** | `0` | **The Zoom Factor.** Default: 0. We explicitly keep this at 0. If set higher, the model crops sections of the image and re-runs segmentation. This causes "over-segmentation" (e.g., seeing a keyboard key as a separate object from the keyboard). <br><br> **Benefit:** Preserves the "whole object" context. |
| **pred_iou_thresh** | `0.85` | **Confidence Threshold.** Default: 0.88. We lowered this slightly. This value represents how confident the model must be that a mask is "good" before keeping it. <br><br> **Benefit:** Allows the detection of slightly ambiguous objects (like items in motion or in shadows) that might otherwise be discarded. |
| **stability_score_thresh** | `0.90` | **Stability Threshold.** Default: 0.95. This measures how much the mask shape changes when the threshold is jittered. <br><br> **Benefit:** Lowering this allows us to keep masks for objects that have fuzzy boundaries (like hair, fur, or cloth). |
| **min_mask_region_area** | `500` | **Minimum Size (Pixels).** Any mask smaller than this area is deleted. <br><br> **Benefit:** Removes "salt and pepper" noise (tiny specks) while still keeping small valid objects like USB drives or pens. |
| **box_nms_thresh** | `0.7` | **Non-Maximum Suppression.** Controls how much two masks can overlap before one is removed. <br><br> **Benefit:** A value of 0.7 allows objects to be close to each other (like a hand holding a cup) without merging them into one blob. |

---

## 🚀 Usage Instructions

### Running the Segmentation

1.  **Prepare your Image:**
    *   Find the image you want to test.
    *   Rename it to `test_image.png`.
    *   Place it in the root folder (same folder as the .py script).

2.  **Execute the Script:**
    *   Open your terminal, ensure `venv` is active, and run:
    ```powershell
    python auto_segment_tuned.py
    ```

3.  **Wait for Processing:**
    *   **First Run:** It may take 10-20 seconds to load the model into GPU memory.
    *   **Inference:** The "Dense Search" (64 points) takes longer than a standard run. Expect 5-15 seconds depending on your GPU.

### Understanding the Output

*   **Console Output:**
    The script will print the number of masks found.
    ```text
    Found 42 masks.
    ```

*   **Visual Output:**
    A new file named `dense_result1.png` will be created.
    *   **Colors:** Each object is overlaid with a random color.
    *   **Transparency:** The alpha channel is set to 0.7, allowing you to see the object texture underneath the mask.
    *   **Layering:** Smaller objects are drawn on top of larger objects to ensure everything is visible.

---

## ❓ Troubleshooting & Common Issues

### 1. ModuleNotFoundError: No module named 'matplotlib'
*   **Cause:** The visualization library is missing.
*   **Solution:**
    ```bash
    pip install matplotlib
    ```

### 2. KeyboardInterrupt or Script hanging at import sympy
*   **Cause:** A known conflict between PyTorch and the mpmath library versions.
*   **Solution:**
    ```bash
    pip uninstall sympy mpmath -y
    pip install sympy mpmath
    ```

### 3. RuntimeError: CUDA out of memory
*   **Cause:** Your GPU does not have enough VRAM (8GB+) for the Large model with `points_per_side=64`.
*   **Solution:**
    *   Open `auto_segment_tuned.py` and change the config:
    *   Reduce `points_per_side` to `32`.
    *   **OR** change the checkpoint to the "Small" model (`sam2.1_hiera_small.pt`).

### 4. "The result is a blue/green square (Empty)"
*   **Cause:** The model didn't find any masks that met the confidence threshold.
*   **Solution:**
    *   Check if the image is too dark.
    *   Lower `pred_iou_thresh` to `0.6` in the script.
    *   Lower `stability_score_thresh` to `0.80`.

### 5. "My Keyboard is still 100 individual keys!"
*   **Cause:** The image resolution might be very high, making keys look like large objects.
*   **Solution:**
    *   Increase `min_mask_region_area` to `2000` or `3000`. This forces the model to ignore anything as small as a key.

---

## 🎨 Advanced Visualization Code

The visualization function in our script is custom-written for clarity. If you wish to use this visualization in other scripts, here is the isolated function:

```python
def show_anns(anns):
    if len(anns) == 0:
        return
    # Sort masks by area (Largest -> Smallest)
    # This ensures large backgrounds don't cover small details
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    # Create a blank RGBA image
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0 # Set alpha to 0 (Transparent)
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # Generate random RGB color
        mask_color = np.random.random(3)
        # Concatenate color with 0.7 Alpha (Opacity)
        color_mask = np.concatenate([mask_color, [0.7]])
        img[m] = color_mask
        
    ax.imshow(img)
```