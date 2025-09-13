from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Model
video = 0  # Camera

results = model.track(
    source=video,
    show=True,
    tracker="bytetrack.yaml",
)
