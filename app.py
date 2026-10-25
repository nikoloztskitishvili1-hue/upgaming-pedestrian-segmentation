"""Gradio demo: pedestrian detection + instance segmentation.

Runs the exported ONNX model (models/best.onnx) through Ultralytics /
ONNX Runtime. Falls back to the PyTorch weights (models/best.pt) if the
ONNX file is missing.

Usage:
    python app.py                       # local, http://127.0.0.1:7860
    MODEL_PATH=models/best.pt python app.py
"""
import os

import gradio as gr
from ultralytics import YOLO

IMGSZ = 640  # must match the size used when exporting the ONNX model


def _load_model():
    path = os.environ.get("MODEL_PATH", "models/best.onnx")
    if not os.path.exists(path):
        fallback = "models/best.pt"
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"Model not found: {path} (and no {fallback}). "
                "Run main.ipynb first, or download the weights - see README."
            )
        print(f"[app] {path} not found, using {fallback}")
        path = fallback
    print(f"[app] loading model: {path}")
    return YOLO(path, task="segment")


model = _load_model()


def predict(image, conf):
    """image: PIL image -> (annotated RGB image, text summary)."""
    if image is None:
        return None, "No image."
    result = model.predict(image, conf=float(conf), imgsz=IMGSZ, verbose=False)[0]
    n = 0 if result.boxes is None else len(result.boxes)
    annotated_bgr = result.plot()  # Ultralytics returns BGR; Gradio expects RGB
    return annotated_bgr[..., ::-1], f"Detected pedestrians: {n}"


def _conf_slider():
    return gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")


def _outputs():
    return [gr.Image(type="numpy", label="Segmentation result"), gr.Textbox(label="Summary")]


def build_demo():
    upload = gr.Interface(
        fn=predict,
        inputs=[gr.Image(sources=["upload"], type="pil", label="Upload image"), _conf_slider()],
        outputs=_outputs(),
        title="Upload image",
    )
    snapshot = gr.Interface(
        fn=predict,
        inputs=[gr.Image(sources=["webcam"], type="pil", label="Webcam snapshot"), _conf_slider()],
        outputs=_outputs(),
        title="Webcam snapshot",
    )
    live = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(sources=["webcam"], type="pil", streaming=True, label="Live webcam"),
            _conf_slider(),
        ],
        outputs=_outputs(),
        live=True,
        title="Live stream",
    )
    return gr.TabbedInterface(
        [upload, snapshot, live],
        ["1. Upload image", "2. Webcam snapshot", "3. Live stream"],
        title="Pedestrian Detection & Instance Segmentation (YOLOv8-seg + ONNX)",
    )


if __name__ == "__main__":
    build_demo().launch()
