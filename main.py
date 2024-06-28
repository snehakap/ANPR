import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import LineString, Point, Polygon
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR
import os
import string
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


# Load the YOLO model
axleLicense_model = YOLO("C:\\Users\\Administrator\\Downloads\\best (3).pt")
cap = cv2.VideoCapture("C:\\Users\\Administrator\\Downloads\\26550692-preview.mp4")

reader = PaddleOCR(lang='en')

translation_table = str.maketrans('', '', '/\\;_!@#$%^&(),:*+=<>{}.-[]\'\' ')


def read_license_plate(license_plate_crop):
    detections = reader.ocr(license_plate_crop)
    combined_text = ''
    highest_confidence = 0

    if detections is not None:
        for detection in detections:
            if detection:
                for text_box in detection:
                    text, confidence = text_box[1]
                    combined_text += text
                    highest_confidence = max(highest_confidence, confidence)

        combined_text = combined_text.translate(translation_table).upper()

        if combined_text:
            return combined_text, highest_confidence
    return None, 0

# Initialize variables for axle tracking
polygon_pts = [(300, 713), (1000, 713), (1000, 450), (630, 340), (300, 480)]
axleLicense_counting_region = Polygon([(200, 620), (950, 620), (950, 370), (200, 370)])
axleLicense_track_history = defaultdict(lambda: deque(maxlen=18))
axleLicense_count_ids = set()
axleLicense_count = defaultdict(int)
current_licensePlate_number = None
highest_confidence_plates = defaultdict(lambda: ("", 0.0))

def axleLicense_process_frame(img):
    wheel_tracks = axleLicense_model.track(img, persist=True, conf=0.75, tracker="bytetrack.yaml")
    annotator = Annotator(img, 2)

    if wheel_tracks[0].boxes.id is not None:
        wheel_boxes = wheel_tracks[0].boxes.xyxy.cpu().numpy()
        wheel_ids = wheel_tracks[0].boxes.id.int().cpu().numpy()
        wheel_classids = wheel_tracks[0].boxes.cls.int().cpu().numpy()

        for wheel_box, wheel_id, classids in zip(wheel_boxes, wheel_ids, wheel_classids):
            wheelLicense_x1, wheelLicense_y1, wheelLicense_x2, wheelLicense_y2 = wheel_box.tolist()
            wheel_center = ((wheel_box[0] + wheel_box[2]) / 2, (wheel_box[1] + wheel_box[3]) / 2)
            track_line = axleLicense_track_history[wheel_id]
            track_line.append(wheel_center)

            if classids == 0:
                annotator.box_label(wheel_box, color=(0, 255, 0))
                plate = img[int(wheelLicense_y1):int(wheelLicense_y2), int(wheelLicense_x1):int(wheelLicense_x2)]
                detections, confidence = read_license_plate(plate)

                if detections is not None:
                    if wheel_id not in highest_confidence_plates or confidence > highest_confidence_plates[wheel_id][1]:
                        highest_confidence_plates[wheel_id] = (detections, confidence)

                    cv2.line(img, (int(wheelLicense_x1) + 30, int(wheelLicense_y1)),
                             (int(wheelLicense_x1) + 30, int(wheelLicense_y2) - 90), (0, 0, 0), 2)

                    # Draw the rectangle above the line
                    rectangle_top_left_x = int(wheelLicense_x1) - 120
                    rectangle_top_left_y = int(wheelLicense_y1) - 80  # Adjust Y to place above the line
                    rectangle_bottom_right_x = int(wheelLicense_x1) + 150
                    rectangle_bottom_right_y = int(wheelLicense_y1) - 50  # Adjust Y to place above the line

                    cv2.rectangle(img, (rectangle_top_left_x, rectangle_top_left_y),
                                  (rectangle_bottom_right_x, rectangle_bottom_right_y), (255, 255, 255), -1)
                    cv2.rectangle(img, (rectangle_top_left_x, rectangle_top_left_y),
                                  (rectangle_bottom_right_x, rectangle_bottom_right_y), (0, 0, 0))
                    cv2.putText(img, f"License Plate: {highest_confidence_plates[wheel_id][0]}",
                                (rectangle_top_left_x + 10, rectangle_top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 2)
            else:
                pass

# Output video file setu
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('LicensePlate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while 1:
    success, frame = cap.read()
    #frame = cv2.resize(frame, (1000, 600))
    if not success:
        break
    axleLicense_process_frame(frame)
    out.write(frame)
    '''cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''

cap.release()
out.release()
cv2.destroyAllWindows()
