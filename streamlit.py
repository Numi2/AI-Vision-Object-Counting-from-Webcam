# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any, List
import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        import streamlit as st

        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind: List[int] = []
        self.model = None
        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict.get("model")
        self.count = 0  # Counter for crossing objects
        self.crossed_ids = set()  # Track IDs that crossed the line

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown("<style>MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)
        self.st.markdown(
            "<h1 style='text-align:center;'>YOLOv11 Webcam Object Counter</h1>", unsafe_allow_html=True
        )

    def sidebar(self):
        self.st.sidebar.title("Settings")
        self.source = self.st.sidebar.selectbox("Video", ("webcam", "video"))
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"
        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))
        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0

    def configure(self):
        models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", models)
        with self.st.spinner("Loading model..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
        class_names = list(self.model.names.values())
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(c) for c in selected_classes]

    def count_crossings(self, results, line_x=400):
        if not results or not results[0].boxes:
            return

        boxes = results[0].boxes
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
        else:
            ids = [i for i in range(len(boxes))]  # fallback: use index if no tracking

        xyxys = boxes.xyxy.cpu().numpy()
        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)

            obj_id = int(ids[i])
            if obj_id in self.crossed_ids:
                continue  # already counted

            if cx > line_x - 5 and cx < line_x + 5:
                self.count += 1
                self.crossed_ids.add(obj_id)

    def inference(self):
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("Could not open source.")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Failed to grab frame.")
                    break

                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                self.count_crossings(results, line_x=400)
                annotated_frame = results[0].plot()

                # Draw vertical line
                cv2.line(annotated_frame, (400, 0), (400, annotated_frame.shape[0]), (0, 255, 0), 2)

                # Draw count
                cv2.putText(
                    annotated_frame,
                    f"Count: {self.count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                if stop_button:
                    cap.release()
                    self.st.stop()

                self.org_frame.image(frame, channels="BGR")
                self.ann_frame.image(annotated_frame, channels="BGR")

            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model).inference()
