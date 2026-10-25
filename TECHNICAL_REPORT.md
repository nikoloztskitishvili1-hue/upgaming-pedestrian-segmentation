# Technical Report - Pedestrian Detection & Instance Segmentation

## 1. Problem definition

The project detects pedestrians in images and segments each one at pixel level (instance segmentation):
for every person it returns a bounding box and a mask. This is a building block for driver-assistance,
crowd counting and people-aware video analytics.

Scope: one class (`person`), still images and webcam frames; everything runs on a free Colab GPU,
and inference runs on CPU. Code: `main.ipynb` (full pipeline) and `app.py` (demo).

## 2. Data, model and approach (and why)

**Dataset - Penn-Fudan Pedestrian.** About 170 street images with pixel-level instance masks (one pixel value per
person). It is small enough to run several experiments on free hardware and it has real instance masks, not only boxes,
which fits a segmentation task. Split: 80 % train (136 images) / 20 % validation (34 images, 91 pedestrians), fixed seed 42.
Each person's mask is converted to a YOLO polygon label (largest contour of that person, normalised coordinates).

**Model - YOLOv8-seg (Ultralytics).** One-stage instance segmentation, pretrained on COCO, simple to fine-tune and
exportable to ONNX. I compared the nano (`n`, 3.4 M parameters) and small (`s`, 11.8 M) variants to see whether
extra capacity pays off on a very small dataset.

**Zero-shot baseline.** COCO already contains a `person` class, so the pretrained models can find pedestrians without
any training. To claim that fine-tuning helps, I first measured the untouched pretrained models on exactly the same
validation split.

## 3. Experiments and results

All runs: 40 epochs, image size 640, batch 8, seed 42, default Ultralytics augmentation. Only the listed factor changes.

| Experiment | Start weights | Change | Hypothesis |
|---|---|---|---|
| zero-shot n / s | COCO weights | no training | what do we get for free? |
| exp1_n_seg | yolov8n-seg | fine-tune all layers | small and fast may be enough for one class |
| exp2_s_seg | yolov8s-seg | larger model | more capacity -> better masks |
| exp3_s_seg_freeze | yolov8s-seg | first 10 layers frozen | with ~136 images, freezing may reduce overfitting |

Validation results (34 images; P = precision, R = recall):

| Model | mask P | mask R | mask mAP50 | **mask mAP50-95** | box mAP50-95 | Params (M) | Size (MB) | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| zero-shot yolov8n-seg | 0.966 | 0.931 | 0.978 | 0.699 | 0.837 | 3.41 | 7.1 | - |
| zero-shot yolov8s-seg | 0.925 | 0.943 | 0.970 | 0.694 | 0.866 | 11.82 | 23.9 | - |
| exp1_n_seg | 0.989 | 0.977 | 0.984 | 0.796 | 0.831 | 3.39 | 7.0 | 3.1 |
| exp2_s_seg | 0.947 | 0.956 | 0.984 | 0.800 | 0.839 | 11.79 | 23.8 | 3.7 |
| exp3_s_seg_freeze | 0.943 | 0.967 | 0.979 | **0.805** | 0.840 | 11.79 | 23.8 | 3.2 |

Full tables: `results/tables.md`; chart: `results/model_comparison.png`; training curves: `results/exp*_curves.png`.

**What the numbers show**

- **Fine-tuning improved the masks.** Mask mAP50-95 rose from about 0.70 (zero-shot) to about 0.80 on the same
  validation split (+0.10).
- **Finding pedestrians was already easy.** mAP50 is about 0.97-0.98 for every model, including the untrained ones.
  The gain comes from more precise mask boundaries at strict overlap thresholds (mAP50-95), and box mAP50-95 did not
  improve (zero-shot `s` 0.866 vs 0.840 after fine-tuning). A likely explanation, which I did not test, is that
  Penn-Fudan masks are annotated differently from COCO masks, so fine-tuning mostly adapts the boundary style.
- **The three fine-tuned models are practically equal** (0.796 / 0.800 / 0.805). With 34 validation images and 91
  pedestrians a difference of one point is within noise, and I ran each setup once, so I cannot claim that `s` beats `n`
  or that freezing helps. The selection rule (highest validation mask mAP50-95) picked `exp3_s_seg_freeze`,
  but the nano model is about 3.4x smaller and nearly as accurate, so it would be the more economical choice in production.
- **Curves:** losses decrease steadily; the first epochs are unstable and then settle. Mask mAP50-95 of exp1 and exp2
  is still rising slightly at epoch 40, while exp3 appears to plateau from about epoch 15-20.

## 4. Optimization (ONNX)

The selected model was exported to ONNX and compared with the PyTorch weights on the same CPU (Colab):
file size, end-to-end latency per image (pre-processing + inference + post-processing, mean over 20 validation images
after warm-up) and validation mask mAP50-95.

| Format | Size (MB) | CPU latency (ms / image) | mask mAP50-95 |
|---|---|---|---|
| PyTorch (.pt) | 23.8 | 403 | 0.805 |
| ONNX (.onnx) | 47.4 | 428 | 0.792 |

**Result: in this setup ONNX did not make the model faster or smaller.** It was about 6 % slower, twice as large and
1.4 points lower in mask mAP50-95. The size difference is most likely because the `.pt` checkpoint stores half-precision
weights while the ONNX file is full precision (not verified). I did not investigate the small accuracy drop; one possible
cause is that the ONNX model runs on a fixed 640x640 input. The practical value of ONNX here is portability: the demo
(`app.py`) runs inference without PyTorch. Not tried: INT8/FP16 quantization, OpenVINO, smaller input size, or exporting the nano model.

## 5. Problems found and how they were solved

- **Best model chosen by file name.** The first version picked "best" weights by sorting folder names. Now every model is
  evaluated on the validation split and the highest mask mAP50-95 is selected.
- **Silent fallback to untrained weights.** If no trained weights were found, the first version exported the untrained
  COCO model without failing. I removed this fallback.
- **No metrics and no baseline.** The first version never evaluated. Evaluation, a zero-shot baseline and result tables were added.
- **ONNX claim without measurement.** Size, latency and accuracy are now measured, and the demo uses the ONNX model.
- **Colour channels.** Ultralytics returns BGR images and Gradio expects RGB, which swapped red and blue in the demo; fixed with a channel flip.
- **Mask conversion.** Each pedestrian has its own pixel value, so the converter creates one polygon per pedestrian,
  drops tiny blobs, pairs images and masks by file name and rebuilds the dataset folder on every run.
- **ONNX export crashed in Colab** with `SameFileError`: Ultralytics already writes `best.onnx` next to `best.pt`, and my
  code tried to copy the file onto itself. Fixed by copying only when the paths differ.
- **GitHub upload limit.** The browser upload limit (25 MB per file) blocks the 47 MB ONNX file, so it is distributed
  through the repository's Releases page.

## 6. Error analysis

For every validation image I compared the number of detected pedestrians (confidence >= 0.25) with the number of
ground-truth pedestrians (`results/error_analysis.csv`, worst cases in `results/error_analysis.png`).

- 34 validation images, 91 ground-truth pedestrians, 100 detections.
- The count was exactly right in 23 of 34 images.
- 3 pedestrians were missed (in 2 images); 12 extra detections appeared (in 9 images).

What I see in the six worst images: (a) some "extra" detections are people who are visible in the photo but not
annotated in the dataset, so these are not necessarily model errors (for example near the image borders in `PennPed00048`);
(b) one image (`PennPed00087`) has two overlapping boxes on a single person; (c) small or distant people are sometimes
missed (`FudanPed00058`: 8 annotated, 6 detected).

I also tried the demo by hand on two photos of my own. In a dense crowd the model found 9 people at confidence 0.25
(12 at 0.05) while clearly more were visible, and in a street photo a traffic light was detected as a person.
These were informal checks, not measurements, and I did not compare the zero-shot COCO model on the same photos, so I cannot say
whether this comes from fine-tuning on a tiny dataset or is a general limit of the model.

Limitation: this analysis compares counts, not masks. A correct count does not guarantee good mask boundaries.

## 7. Strengths and weaknesses

**Strengths**
- Complete and reproducible pipeline (seeded split, scripted data conversion, one notebook, `requirements.txt`).
- Zero-shot baseline plus three controlled experiments, box and mask metrics on the same split.
- Optimization is measured, and the negative result (no speed-up) is reported honestly.
- Demo works with uploaded images, webcam snapshots and a live webcam stream, and uses the exported ONNX model.

**Weaknesses**
- Very small dataset (~170 images; 34 validation images). Metrics are noisy, each experiment ran once, and the same
  validation split was used for model selection and reporting (no separate test set), so the numbers are slightly optimistic.
- One dataset and one class: robustness to crowds, night scenes, other cameras and look-alike objects is untested
  and showed weaknesses in my informal demo checks.
- ONNX gave no speed or size benefit in this setup, and the small accuracy drop is unexplained.
- Error analysis is count-based, not IoU-based.

## 8. What I would improve with more time and resources

- Repeat each experiment with several seeds or k-fold cross-validation and report mean +- std; keep a separate test set.
- Add more diverse data (e.g. COCO persons or BDD100K), including images without pedestrians, and test crowds and look-alike objects systematically.
- Train longer (exp1/exp2 were still improving) and try augmentation/learning-rate experiments.
- Try INT8/FP16 quantization, OpenVINO or exporting the nano model, and compare latency and accuracy.
- IoU-based per-instance error analysis and tracking for video.

## 9. AI tools usage disclosure

- ChatGPT / Gemini were used for the first version of the notebook and report (script structure, debugging, formatting).
- Claude (Anthropic) was used afterwards to review the project against the assignment, find bugs and rewrite parts of the
  pipeline (data conversion, experiments, evaluation, ONNX benchmark, demo, README, report).
- Verification: I ran the complete notebook on Google Colab, checked the resulting tables, training curves and error-analysis
  images, fixed the ONNX export error that appeared during the run, and tested the demo by hand on my own photos.
  All numbers in this report come from my own run.
