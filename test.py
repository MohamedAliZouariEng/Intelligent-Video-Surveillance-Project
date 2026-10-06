import argparse
from collections import deque

import cv2
import imutils
import numpy as np
from keras.models import load_model

# ---------------- Settings ----------------
THRESHOLD = 0.000275     # recalibrate: slightly above the loss of a normal video
SMOOTH_WINDOW = 3        # average the last N chunk losses to avoid flicker
DISPLAY_WIDTH = 1000
FRAME_DELAY_MS = 30      # playback speed (higher = slower)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
# ------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="test_video1.mp4", help="path to the input video")
parser.add_argument("--model", default="saved_model.keras", help="path to the trained model")
parser.add_argument("--save", default=None, help="optional path to save the annotated video, e.g. output.mp4")
args = parser.parse_args()


def mean_squared_loss(x1, x2):
    difference = x1 - x2
    n_samples = difference.size
    distance = np.sqrt((difference ** 2).sum())
    return distance / n_samples


def draw_status(image, abnormal, loss, smooth, tick):
    """Draw the NORMAL / ABNORMAL sign on the frame."""
    h, w = image.shape[:2]
    color = RED if abnormal else GREEN
    label = "ABNORMAL ACTION" if abnormal else "NORMAL"

    # colored border (blinks when abnormal)
    if not abnormal or (tick // 5) % 2 == 0:
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, 8)

    # banner at the top
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), color, -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 4)

    # loss info at the bottom left
    info = f"loss: {loss:.6f}  smooth: {smooth:.6f}  threshold: {THRESHOLD:.6f}"
    cv2.putText(image, info, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return image


model = load_model(args.model)

cap = cv2.VideoCapture(args.video)
print("Video opened:", cap.isOpened())

writer = None
recent = deque(maxlen=SMOOTH_WINDOW)
tick = 0
stop = False

while cap.isOpened() and not stop:
    gray_frames = []
    display_frames = []

    # read a chunk of 10 frames
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        display_frames.append(imutils.resize(frame, width=DISPLAY_WIDTH))

        small = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        # OpenCV is BGR, so the weights are reversed
        gray = 0.1140 * small[:, :, 0] + 0.5870 * small[:, :, 1] + 0.2989 * small[:, :, 2]
        gray_frames.append(gray.astype(np.float32) / 255.0)  # same 0-1 scale as training

    if len(gray_frames) < 10:  # end of video / incomplete chunk
        break

    x = np.stack(gray_frames, axis=-1)       # (227, 227, 10)
    x = np.expand_dims(x, axis=0)            # (1, 227, 227, 10)
    x = np.expand_dims(x, axis=4)            # (1, 227, 227, 10, 1)

    output = model.predict(x, verbose=0)
    loss = mean_squared_loss(x, output)

    recent.append(loss)
    smooth = sum(recent) / len(recent)
    abnormal = smooth > THRESHOLD
    print(f"loss: {loss:.6f}  smooth: {smooth:.6f}  ->  {'ABNORMAL' if abnormal else 'NORMAL'}")

    # play the 10 frames of this chunk with the status sign on them
    for img in display_frames:
        tick += 1
        img = draw_status(img, abnormal, loss, smooth, tick)

        if args.save and writer is None:
            h, w = img.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(img)

        cv2.imshow("video", img)
        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord("q"):
            stop = True
            break

cap.release()
if writer is not None:
    writer.release()
    print("Saved annotated video to", args.save)
cv2.destroyAllWindows()