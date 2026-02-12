# 1. Understanding DINOv2 as the Base Vision Model

First, let's start with DINOv2, which is the foundation of the entire workflow your team lead described. DINOv2 is a self-supervised vision transformer (ViT) model developed by Meta AI in 2023. It is designed as a "foundation model" for computer vision tasks, meaning it is pre-trained on a massive dataset (millions of images) in an unsupervised way (no labels needed), and it can be adapted for various downstream tasks like depth estimation, segmentation, classification, or object detection with minimal fine-tuning.

## Key Concepts in DINOv2

Self-Supervised Learning (SSL): DINOv2 is trained using a method called "Distillation with no labels" (DINO), which is an improved version of knowledge distillation. It uses a teacher-student network architecture where the student learns to match the teacher's representations of augmented views of the same image. This allows the model to learn rich visual features (e.g., edges, textures, semantics) without human-annotated labels.

**Architecture:** It is based on Vision Transformers (ViT) — models that divide an image into patches (like tokens in language models) and process them with self-attention layers. DINOv2 comes in different sizes (small, base, large, giant), with the base model having ~86 million parameters. It outputs feature embeddings (dense vectors representing image regions) that are highly generalizable.


## Pre-training Dataset: 

Trained on ~142 million images from a curated dataset (LVD-142M), focusing on diversity (natural scenes, objects, textures) and quality (removing duplicates, low-quality images).


**Output:** For an input image, DINOv2 produces multi-scale feature maps (e.g., at resolutions like 1/4, 1/8, 1/16 of the input) that capture local and global visual information. These features are used as a "backbone" for other tasks — that's why your workflow uses DINOv2 as the base for both depth and segmentation.

## Why DINOv2 is Used in This Workflow

1. It provides strong monocular priors (understanding from single images) that generalize well to new domains without retraining from scratch.
2. It is the backbone for advanced models like Depth Anything V2 (for depth) and SAM2 (for segmentation), making it easy to fuse tasks.
**Advantages:** High zero-shot performance, efficient (runs on RTX 4050), and adaptable with small datasets for fine-tuning.

## Deployment and Usage of DINOv2 Standalone

**Installation:** pip install dinov2 or clone https://github.com/facebookresearch/dinov2.

Basic Test (extract features): 
```Python
import torch
from dinov2.models.vision_transformer import vit_base
model = vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
model.eval().to("cuda")
image = torch.rand(1, 3, 518, 518).to("cuda")  # test input
features = model(image)
print(features.shape)  # Output: torch.Size([1, 1024]) or multi-scale
```

**Parameters:** Layer size (e.g., 12 layers for base), patch size (14 or 16), input resolution (518x518 for high-res).
Training Data Needs: Pre-trained on 142M images — for fine-tuning, 10K–100K task-specific images are sufficient (e.g., 10K for depth).

This completes the explanation of DINOv2. Now, moving to the next concept.

# 2. Using DINOv2 for Depth Estimation (Disparity Map)

Your workflow uses DINOv2 for depth estimation, which aligns with models like Depth Anything V2 (DAv2), a monocular depth model that uses DINOv2 as its backbone. FoundationStereo (which is deployed) is a stereo depth model, but the workflow seems to shift to monocular depth (single image) using DINOv2, possibly for fusing with segmentation. You mentioned retraining FoundationStereo with your own images, but the diagram points to DINOv2 for depth. I'll explain DINOv2-based depth, and how it relates to disparity maps.

## Key Concepts in DINOv2 for Depth

Monocular Depth Estimation: Unlike stereo (which needs left/right images to compute disparity = pixel shift, then depth = focal * baseline / disparity), monocular depth estimates depth from one image using learned priors (e.g., shadows, textures, object sizes).

### DAv2 Architecture: 

DINOv2 backbone extracts features, then a decoder head (e.g., convolutional layers) predicts a depth map. It is trained on 62M images with pseudo-labels from teacher models like MiDaS or DINO.

**Disparity Map in Context:** For monocular models, "disparity" is often inverse depth (1/depth) — closer to how stereo disparity works. DINOv2-based models output a relative depth map, which can be converted to metric with scale/shift alignment.

**Why DINOv2?:** Its self-supervised features are robust to domain shifts (e.g., indoor/outdoor, different cameras), making it better for zero-shot depth than traditional CNNs.

**Retraining with Your Own Images:** As per your team lead, retrain the depth head on DINOv2 backbone using your Arducam stereo images (use ground-truth depth from FoundationStereo or LiDAR if available).

## Deployment and Usage of DINOv2 for Depth

**Installation:** pip install depth-anything-v2 (or clone https://github.com/DepthAnything/Depth-Anything-V2).

Basic Test (get depth):
```python
from depth_anything_v2.dinov2 import DepthAnythingV2
model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
model.load_state_dict(torch.load('depth_anything_v2_vitb.pth'))
model.eval().to("cuda")
image = cv2.imread("test_image.png")
depth = model.infer_image(image)
cv2.imwrite("depth_map.png", depth)
```

**Parameters:** Encoder size (vitb = base, vitl = large), feature dimensions (64–256), decoder heads (4 for multi-scale).
Training Data Needs: Pre-trained on 62M images. For retraining (as your team lead wants): 5K–50K images with depth labels (use your Arducam stereo pairs with FoundationStereo as label generator). Batch size 8–16, epochs 5–20, learning rate 1e-5, loss = L1 or Scale-Invariant Loss.

This completes the explanation of DINOv2 for depth. Now, moving to the next concept.

# 3. Using DINOv2 for Segmentation (Segmentation Masks)

Your workflow uses DINOv2 for segmentation, which aligns with SAM2 (Segment Anything Model 2), a segmentation model that uses DINOv2 as its image encoder backbone. SAM2 is designed for "segment anything" — it can generate masks for objects based on prompts (points, boxes, text). You have already deployed SAM2, but the team lead wants to retrain it with your own input images for better performance on your Arducam data.
Key Concepts in DINOv2 for Segmentation

**Prompt-Based Segmentation:** SAM2 takes an image and a prompt (e.g., point on an object) and outputs binary masks (segmentation maps) for that object. It can do "automatic" segmentation (find all objects) or interactive (user clicks).

**Architecture:** DINOv2 backbone extracts features, then a lightweight decoder (Transformer-based) generates masks. SAM2 improves on SAM1 with better video support and efficiency, but for images, it's similar.

**Segmentation Masks:** Output is a binary image (H x W, 1 = object, 0 = background) or multiple masks for multi-object.
Why DINOv2?: Its features are semantic and robust, allowing SAM2 to generalize to unseen objects without training on specific classes.

**Retraining with Your Own Images:** Retrain the decoder head on DINOv2 backbone using your Arducam images with mask labels (use SAM2's auto-mask as pseudo-labels or manual annotation).

## Deployment and Usage of DINOv2 for Segmentation

**Installation:** Already done (clone https://github.com/facebookresearch/segment-anything-2, pip install -e .).

Basic Test (get masks):
```Python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "sam2.1_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
image = cv2.imread("test_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)
point_coords = np.array([[500, 375]])  # example point
point_labels = np.array([1])
masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
cv2.imwrite("mask.png", masks[0] * 255)
```
**Parameters:** Model size (small/large), prompt type (point/box/mask), multimask_output (True for 3 quality levels).
Training Data Needs: Pre-trained on SA-1B (1B masks). For retraining: 1K–10K images with mask labels (use LabelStudio or SAM2 pseudo-labeling). Batch size 4–8, epochs 5–10, learning rate 1e-4, loss = BCE or Focal Loss.

This completes the explanation of DINOv2 for segmentation. Now, moving to the next concept.

# 4. Fusing Depth Heads with Segmentation Heads to Get Masks

Your workflow's final step is fusing depth heads with segmentation heads to get enhanced masks. This means combining the depth model (from DINOv2 + depth decoder) with the segmentation model (DINOv2 + segmentation decoder) into a single system that uses depth information to improve segmentation, or vice versa (e.g., depth-aware masks or segmented depth maps).
Key Concepts in Fusing Depth and Segmentation

**Head Fusion:** Both tasks share the DINOv2 backbone, but have separate "heads" (decoders). Fusion can be:
Feature-level: Concatenate depth and segmentation features from the backbone and train a joint decoder.

**Output-level:** Use depth map to refine segmentation masks (e.g., remove background with depth thresholds) or use masks to segment depth (per-object depth).

**Why Fuse?:** Depth helps segmentation in low-texture areas (e.g., walls), while segmentation helps depth by providing object boundaries. Result: Depth-aware masks (e.g., segment only foreground at certain depths).

**Common Methods:** Use a multi-task learning setup (train with joint loss: depth loss + segmentation loss). Or use models like Depth-Aware SAM (custom SAM2 with depth input).

**Retraining the Fused Model:** Freeze DINOv2 backbone, train the fused heads on your images with paired depth/mask labels.

## Deployment and Usage of Fused Model

**Installation:** Use DAv2 and SAM2 repos, add custom fusion layer in code.

Basic Test (simple fusion):
```Python
# Assume depth_map from DINOv2 depth, mask from SAM2
fused_mask = mask & (depth_map < threshold)  # e.g., segment only close objects
cv2.imwrite("fused_mask.png", fused_mask * 255)
```
**Parameters:** Fusion type (early/late), joint loss weight (0.5 for depth, 0.5 for segmentation), threshold for depth filtering (e.g., 2.0 meters).

**Training Data Needs:** 5K–50K images with both depth and mask labels (use FoundationStereo for depth labels, SAM2 for mask pseudo-labels). Batch size 4, epochs 10–20, learning rate 1e-5, loss = L1 (depth) + BCE (segmentation).

This completes the explanation of fusion. Now, moving to the full workflow.

# 5. The Full Workflow, Parameters, and Training Data Needs

This is a multi-task vision system using DINOv2 as the shared backbone for depth and segmentation, then fusing them for enhanced masks. It leverages pre-trained models (FoundationStereo for stereo depth, SAM2 for segmentation) but requires retraining on your Arducam images for customization.

**Full Workflow Explanation**

DINOV2(Vision): Input stereo or monocular image → DINOv2 extracts shared features.

Depth Branch: DINOv2 features → depth head → disparity map (retrained on your images).

Segmentation Branch: DINOv2 features → segmentation head → masks (retrained on your images).

Fusion: Combine depth + masks → depth-aware masks (e.g., segment only objects at specific depths).

Output: Segmentation masks with depth context (e.g., "person at 1.5m").

**Overall Parameters**

Backbone: DINOv2 base/large (86M–300M params).

Input resolution: 518x518 (for DINOv2 high-res mode).

Optimizer: AdamW (lr=1e-5).

Loss: Joint (0.5 * depth L1 + 0.5 * segmentation BCE).

Batch size: 4–8 (RTX 4050 limit).

Epochs: 5–20 per task, 10 for fusion.

Hardware: RTX 4050 (6GB) → use small DINOv2 to avoid OOM.

**Training Data Needs**

Total: 5K–50K images per task (use your Arducam stereo pairs).

Depth Training: 5K–20K stereo pairs with disparity labels (use FoundationStereo as pseudo-labeler).

Segmentation Training: 1K–10K images with mask labels (use SAM2 auto-masks as pseudo-labels, refine with LabelStudio).

Fusion Training: 5K images with both depth + mask labels (combined dataset).

Source: Your Arducam captures (augment with flips, color jitter).

## Full Workflow 

| Concept | Explanation | Key Parameters | Training Data Needs | How to Retrain |
|---------|-------------|----------------|--------------------|---------------|
| DINOv2 (Base Vision Model) | DINOv2 is a self-supervised Vision Transformer (ViT) model from Meta AI (2023), serving as the shared feature extractor for both depth and segmentation tasks in this workflow. It learns rich visual representations (features like edges, textures, semantics) from millions of unlabeled images using teacher-student distillation (DINO method). As a foundation model, it provides generalizable features that can be adapted to downstream tasks with minimal data, bridging sim-to-real gaps. In this phase, DINOv2 processes the input image to produce multi-scale feature maps, which are used by both depth and segmentation heads. | - Model Size: Small (21M params), Base (86M), Large (303M), Giant (1B)<br>- Patch Size: 14 or 16<br>- Input Resolution: 518x518 (for high-res mode)<br>- Layers: 12 (base), 24 (large)<br>- Feature Dimensions: 768 (base), 1024 (large)<br>- Optimizer: AdamW (lr=1e-4)<br>- Batch Size: 8–16 on RTX 4050 | Pre-trained on 142M unlabeled images (LVD-142M). For retraining/fine-tuning the backbone: 10K–100K custom images (unlabeled for SSL, or labeled for task-specific). Augment with flips, color jitter. | Freeze early layers, fine-tune later layers or adapters on your Arducam images. Use DINO loss for SSL. Epochs: 5–10. Use hydra for config management. Example: `python train_dinov2.py --config configs/dinov2_base.yaml` (custom script needed). |
| Depth (Disparity Map) - FoundationStereo Deployed (Retrained) | Depth estimation in this workflow uses FoundationStereo (NVIDIA, 2025) as the stereo depth model, but the team lead suggests shifting to a DINOv2-based monocular depth head (like Depth Anything V2) for fusion. FoundationStereo computes disparity (pixel shift between left/right images) using a hybrid cost volume, attentive filtering, and iterative refinement (ConvGRU). It outputs a disparity map, convertible to depth (depth = fx * baseline / disparity). You have deployed it, tested with samples, and need to retrain on your Arducam images for custom scenes (e.g., better handling of wide FOV distortions). | - Max Disparity: 416<br>- Valid Iters: 22–32 (refinement steps)<br>- Scale: 0.7–1.0 (input resize)<br>- Hiera: 1 (high-res mode)<br>- Z Far: 50.0 (clip depth)<br>- Optimizer: AdamW (lr=1e-4)<br>- Loss: Smooth L1<br>- Batch Size: 4–8 on RTX 4050 | Pre-trained on 1M synthetic FSD dataset. For retraining: 5K–20K stereo pairs from your Arducam (use self-supervised labels or LiDAR if available). Augment with rotation, noise. | Retrain the decoder heads on DINOv2 backbone (freeze DINOv2). Use your Arducam stereo pairs as input, FoundationStereo outputs as labels. Epochs: 10–20. Script: Modify `scripts/run_demo.py` for training loop or use custom `train_depth.py`. |
| Segmentation (Segmentation Masks) - SAM2 Deployed (Retrained) | Segmentation uses SAM2 (Meta, 2024) as the model, with DINOv2 as the image encoder backbone. SAM2 generates binary masks for objects based on prompts (points, boxes, text). It uses feature maps from DINOv2 to predict masks via a lightweight Transformer decoder. You have deployed SAM2, tested with samples, and need to retrain on your Arducam images for better performance on your domain (e.g., wide-angle distortions, specific objects). Output is masks (H x W binary arrays) for fusion. | - Model Size: Tiny (38M), Small (46M), Base+ (80M), Large (224M)<br>- Prompt Type: Point/Box/Mask/Text<br>- Multimask Output: False (single mask)<br>- Optimizer: AdamW (lr=1e-4)<br>- Loss: BCE/Focal<br>- Batch Size: 4–8 on RTX 4050 | Pre-trained on SA-1B (1B masks). For retraining: 1K–10K images from your Arducam with mask labels (use SAM2 auto-masks as pseudo-labels or LabelStudio for manual). Augment with flips, crops. | Retrain the decoder heads on DINOv2 backbone (freeze DINOv2). Use your Arducam images as input, pseudo-masks as labels. Epochs: 5–10. Script: Use SAM2's training code or custom `train_sam2.py`. |
| Fusing Depth Heads with Segmentation Heads to Get Masks | Fusion combines the depth head (from DINOv2 + depth decoder) and segmentation head (DINOv2 + segmentation decoder) to produce enhanced masks. This can be feature-level (concatenate DINOv2 features and train a joint decoder) or output-level (use depth map to refine masks, e.g., segment only objects at certain depths). The fused system outputs depth-aware segmentation masks (e.g., masks with object distance labels). This improves accuracy in ambiguous areas (depth helps textureless segmentation, masks help depth boundaries). | - Fusion Type: Feature-level (concat features) or Output-level (post-process)<br>- Joint Loss Weight: 0.5 depth + 0.5 segmentation<br>- Depth Threshold: 2.0 m (for foreground)<br>- Optimizer: AdamW (lr=1e-5)<br>- Loss: L1 (depth) + BCE (segmentation)<br>- Batch Size: 4 on RTX 4050 | 5K–50K images with both depth and mask labels (use FoundationStereo for depth, SAM2 for masks). Augment with mixed tasks. | Freeze DINOv2, train joint heads on combined loss. Use your Arducam data. Epochs: 10–20. Script: Custom `train_fusion.py` fusing DAv2 and SAM2 decoders. |
| Full Workflow (Phase 1: Vision) | The full workflow starts with Phase 1 (Vision): Input stereo/monocular image → DINOv2 extracts shared features → split to depth branch (disparity map from FoundationStereo or DINOv2 head, retrained) and segmentation branch (masks from SAM2, retrained) → fuse heads to get depth-aware segmentation masks. This creates a multi-task system for segmented metric depth. Retrain all on your Arducam data for customization. Output: Masks with depth context (e.g., "person at 1.5m"). | - Backbone: DINOv2 base/large<br>- Input Res: 518x518<br>- Optimizer: AdamW (lr=1e-5)<br>- Joint Loss: 0.5 depth + 0.5 segmentation<br>- Batch Size: 4–8<br>- Epochs: 5–20 per task, 10 for fusion | 5K–50K Arducam images with paired depth/mask labels (pseudo-labeled from pre-trained models). Augment for diversity. | Freeze DINOv2, retrain heads separately then fuse. Use Hydra for configs. Script: `train_workflow.py`. |


# 6. Implementation Guide: Step-by-Step Procedure

## Step 1: Organize Your Project Structure
Your current structure is good, but let's add subfolders for clarity and to keep things modular. Run these commands in PowerShell from `D:\moni\bluvern`:

```powershell
# Create main project folder for this workflow (to avoid cluttering bluvern root)
mkdir stereo_fusion_project
cd stereo_fusion_project

# Create subfolders for datasets, models, scripts, and outputs
mkdir datasets
mkdir models
mkdir scripts
mkdir outputs
mkdir docs

# Copy your existing deployments into this project (to keep everything in one place)
copy ..\FoundationStereo -Destination foundation_stereo -Recurse -Force
copy ..\sam2_project -Destination sam2 -Recurse -Force
copy ..\VIT -Destination dinov2 -Recurse -Force  # Even if ignoring for now, to have it ready
copy ..\fusion_workflow.md -Destination docs
copy ..\semantic_stereo_fusion.md -Destination docs

# Create a main README.md for the full workflow
New-Item -ItemType File -Path README.md -Force
```

### Updated Structure After This Step:
```text
D:\moni\bluvern\stereo_fusion_project\
├── datasets/  (for training data)
├── models/  (for checkpoints)
├── scripts/  (for training/fusion code)
├── outputs/  (for results)
├── docs/  (fusion_workflow.md, semantic_stereo_fusion.md, README.md)
├── foundation_stereo/  (your deployed FoundationStereo)
├── sam2/  (your deployed SAM2)
└── dinov2/  (VIT folder, ignored for now)
```

> [!TIP]
> **Why this step?**
> Keeps everything centralized, easy to git commit, and avoids scattering files.

## Step 2: Set Up the Conda Environment
Reuse your `foundation_stereo` env since it's already configured with CUDA and dependencies for FoundationStereo and SAM2.

```powershell
# Reuse existing
conda activate foundation_stereo

# (Optional) If you want a new env for this project
conda create -n stereo_fusion -y python=3.10
conda activate stereo_fusion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install git+https://github.com/facebookresearch/dinov2.git
pip install flash-attn --no-build-isolation
pip install hydra-core opencv-python pandas matplotlib seaborn tqdm pillow
```

> [!TIP]
> **Why this step?**
> Ensures all dependencies (PyTorch, SAM2, DINOv2, Hydra for configs) are in place.

## Step 3: Deploy DINOv2 as the Base Vision Model
DINOv2 is the shared backbone for both depth and segmentation. Deploy it standalone to test feature extraction.
https://github.com/facebookresearch/dinov2

### Install DINOv2
```powershell
pip install git+https://github.com/facebookresearch/dinov2.git 
```

### Download DINOv2 checkpoint
```powershell
mkdir models/dinov2
cd models/dinov2
curl -L -O https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
cd ..\..
```

### Test DINOv2
Create `scripts/test_dinov2.py`:
```python
import torch
from dinov2.models.vision_transformer import dinov2_vitb14
from PIL import Image
from torchvision import transforms

model = dinov2_vitb14(pretrained=True).eval().cuda()
checkpoint = torch.load("models/dinov2/dinov2_vitb14_pretrain.pth")
model.load_state_dict(checkpoint, strict=False)

transform = transforms.Compose([
    transforms.Resize(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "test_image.png"  # your sample image
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0).cuda()

features = model(input_tensor)
print("Feature shape:", features.shape)  # e.g., torch.Size([1, 1024])
```

Run the test:
```powershell
copy D:\moni\bluvern\FoundationStereo\assets\left.png test_image.png
python scripts/test_dinov2.py
```

> [!TIP]
> **Why this step?**
> DINOv2 provides the shared features. Test it to ensure the backbone works.

## Step 4: Retrain FoundationStereo for Depth
Retrain the depth head on DINOv2 features for your specific domain.

### Prepare Dataset
1. Collect 5K–20K stereo pairs from your Arducam (left/right images).
2. **Generate labels:** Use pre-trained FoundationStereo to create disparity labels (pseudo-labeling).
3. **Folder structure:**
   - `datasets/my_arducam/left/`
   - `datasets/my_arducam/right/`
   - `datasets/my_arducam/disparity/`

### Modify FoundationStereo for Retraining
1. Update `scripts/train.py` (or create `scripts/retrain_depth.py`) to load your dataset.
2. **Config:** Use `configs/train_fsd.yaml` as base, change `dataset_path` to your folder.

### Run Retraining
```powershell
python scripts/train.py --cfg configs/train_fsd.yaml --dataset_path datasets/my_arducam --epochs 10 --batch_size 4 --lr 1e-5
```

### Dataset Details
- **Size:** 5K–20K pairs (start with 5K, add more if accuracy low).
- **Sources:** Your Arducam captures (indoor/outdoor, various lighting).
- **Labels:** Pseudo-disparity from pre-trained model or LiDAR if available.
- **Augmentation:** Flips, rotation, color jitter (use imgaug).

> [!TIP]
> **Why this step?**
> Retrains the depth head on DINOv2 features for your domain.

## Step 5: Retrain SAM2 for Segmentation
Retrain the segmentation head on DINOv2 features for your Arducam data.

### Prepare Dataset
1. Collect 1K–10K images from your Arducam.
2. **Generate labels:** Use pre-trained SAM2 for pseudo-masks, refine with LabelStudio.
3. **Folder structure:**
   - `datasets/my_arducam/images/`
   - `datasets/my_arducam/masks/`

### Modify SAM2 for Retraining
Use SAM2's training code (modify `sam2/train.py` or create `scripts/retrain_sam2.py`).

### Run Retraining
```powershell
python scripts/retrain_sam2.py --dataset_path datasets/my_arducam --epochs 5 --batch_size 4 --lr 1e-4
```

### Dataset Details
- **Size:** 1K–10K images with masks.
- **Sources:** Your Arducam images (diverse objects/scenes).
- **Labels:** Pseudo-masks from SAM2, manual refinement.
- **Augmentation:** Crops, flips, scale.

> [!TIP]
> **Why this step?**
> Retrains the segmentation head on DINOv2 features for your data.

## Step 6: Fuse Depth and Segmentation Heads
Produces depth-aware masks as final output.

### Create Fusion Script
Create `scripts/fusion.py` to:
1. Load DINOv2 backbone.
2. Attach depth and segmentation heads.
3. Train with joint loss.

### Run Fusion Training
```powershell
# Prepare Joint Dataset: Use 5K images with both depth and mask labels.
python scripts/fusion.py --dataset_path datasets/my_arducam --epochs 10 --batch_size 4 --lr 1e-5
```

### Dataset Details
- **Size:** 5K–50K joint labeled images.
- **Sources:** Your Arducam with pseudo-labels from earlier steps.

> [!TIP]
> **Why this step?**
> Produces depth-aware masks as final output.

## Step 7: Test and Evaluate

### Run Fused Model
```powershell
python scripts/fusion.py --mode test --input left.jpg right.jpg --output outputs/fused_masks.png
```

### Evaluation
Use metrics like IoU for masks and RMSE for depth.

### Overall Dataset Summary
- **Sources:** Arducam captures, synthetic (FSD for depth, SA-1B for segmentation).
- **Size:** 5K–50K per task, 5K for fusion.
- **Tools:** LabelStudio for manual labels, pseudo-labeling from pre-trained models.
- **Storage:** 100–500 GB (images + masks/depth).

