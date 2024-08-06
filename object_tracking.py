import cv2
import numpy as np
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from ultralytics import YOLO

def objectTracking(play_realtime=False, save_to_file=False):
    # Initialize
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    n_frame = 4001
    frames = []
    frames_draw = []
    bboxs = np.empty((n_frame,), dtype=np.ndarray)

    model = YOLO('D:/Dataset/emanweight.pt')
    object_detected = False
    frame_idx = 0
    startXs, startYs = None, None

    object_name = "clutch"  # Fixed name for detected objects

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not object_detected:
            # Detect objects in the current frame
            detections = model(frame)
            n_object = len(detections[0].boxes)

            if n_object > 0:
                object_detected = True
                # Initialize bounding boxes from detected objects
                bboxs[0] = np.empty((n_object, 4, 2), dtype=float)
                for i, bbox in enumerate(detections[0].boxes):
                    xyxy = bbox.xyxy[0].numpy()
                    x_min, y_min, x_max, y_max = xyxy
                    bboxs[0][i, :, :] = np.array(
                        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]).astype(float)

                startXs, startYs = getFeatures(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), bboxs[0], use_shi=False)
                first_frame = frame.copy()

                if save_to_file:
                    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0,
                                          (first_frame.shape[1], first_frame.shape[0]))

        if object_detected:
            frames.append(frame)
            if frame_idx > 0:
                print('Processing Frame', frame_idx)
                newXs, newYs = estimateAllTranslation(startXs, startYs, frames[frame_idx - 1], frames[frame_idx])
                Xs, Ys, bboxs[frame_idx] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[frame_idx - 1])

                # Update coordinates
                startXs = Xs
                startYs = Ys

                # Update feature points as required
                n_features_left = np.sum(Xs != -1)
                print('# of Features: %d' % n_features_left)
                if n_features_left < 15:
                    print('Generate New Features')
                    startXs, startYs = getFeatures(cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2GRAY), bboxs[frame_idx])

                # Print real-time coordinates and fixed name
                for j in range(n_object):
                    (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[frame_idx][j, :, :].astype(int))
                    print(f"Object {j} ({object_name}): Bounding Box - x: {xmin}, y: {ymin}, w: {boxw}, h: {boxh}")

                    for k in range(startXs.shape[0]):
                        if startXs[k, j] != -1 and startYs[k, j] != -1:
                            print(f"Feature {k} ({object_name}): x: {int(startXs[k, j])}, y: {int(startYs[k, j])}")

                # Draw bounding box, feature points, and fixed name for each object
                frame_draw = frames[frame_idx].copy()
                for j in range(n_object):
                    (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[frame_idx][j, :, :].astype(int))
                    frame_draw = cv2.rectangle(frame_draw, (xmin, ymin), (xmin + boxw, ymin + boxh), (255, 0, 0), 2)
                    for k in range(startXs.shape[0]):
                        if startXs[k, j] != -1 and startYs[k, j] != -1:
                            frame_draw = cv2.circle(frame_draw, (int(startXs[k, j]), int(startYs[k, j])), 3, (0, 0, 255), thickness=2)

                    # Draw object name
                    cv2.putText(frame_draw, object_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frames_draw.append(frame_draw)

                # Show the result in real-time
                if play_realtime:
                    cv2.imshow("win", frame_draw)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                if save_to_file:
                    out.write(frame_draw)

            frame_idx += 1
            if frame_idx >= n_frame:
                break

    if save_to_file and 'out' in locals():
        out.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    objectTracking(play_realtime=True, save_to_file=True)
