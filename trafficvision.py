import cv2
import csv
import math
from collections import OrderedDict


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH =  r"input.mp4"

MIN_AREA = 1500

LINE_Y = 400

MAX_DISTANCE = 80

MAX_MISSED_FRAMES = 15

OUTPUT_CSV = "vehicle_events.csv"


# ============================================================
# CENTROID TRACKER
# ============================================================

class CentroidTracker:

    def __init__(self, max_distance=80, max_missed_frames=15):

        self.next_id = 1

        self.objects = OrderedDict()

        self.previous_positions = OrderedDict()

        self.missed_frames = OrderedDict()

        self.max_distance = max_distance

        self.max_missed_frames = max_missed_frames


    def register(self, centroid):

        object_id = self.next_id

        self.objects[object_id] = centroid

        self.previous_positions[object_id] = centroid

        self.missed_frames[object_id] = 0

        self.next_id += 1


    def deregister(self, object_id):

        del self.objects[object_id]
        del self.previous_positions[object_id]
        del self.missed_frames[object_id]


    def update(self, detected_centroids):

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if len(detected_centroids) == 0:

            for object_id in list(self.objects.keys()):

                self.missed_frames[object_id] += 1

                if self.missed_frames[object_id] > self.max_missed_frames:
                    self.deregister(object_id)

            return self.objects


        # ----------------------------------------------------
        # No currently tracked objects
        # ----------------------------------------------------

        if len(self.objects) == 0:

            for centroid in detected_centroids:
                self.register(centroid)

            return self.objects


        # ----------------------------------------------------
        # Match existing objects to new detections
        # ----------------------------------------------------

        object_ids = list(self.objects.keys())

        object_centroids = list(self.objects.values())


        distances = []

        for object_centroid in object_centroids:

            row = []

            for detected_centroid in detected_centroids:

                dx = object_centroid[0] - detected_centroid[0]

                dy = object_centroid[1] - detected_centroid[1]

                distance = math.sqrt(dx * dx + dy * dy)

                row.append(distance)

            distances.append(row)


        used_objects = set()

        used_detections = set()


        # ----------------------------------------------------
        # Greedy nearest-neighbor matching
        # ----------------------------------------------------

        while True:

            best_distance = float("inf")

            best_object_index = None

            best_detection_index = None


            for object_index in range(len(object_ids)):

                if object_index in used_objects:
                    continue


                for detection_index in range(len(detected_centroids)):

                    if detection_index in used_detections:
                        continue


                    distance = distances[
                        object_index
                    ][
                        detection_index
                    ]


                    if distance < best_distance:

                        best_distance = distance

                        best_object_index = object_index

                        best_detection_index = detection_index


            if best_object_index is None:
                break


            if best_distance > self.max_distance:
                break


            object_id = object_ids[best_object_index]

            new_centroid = detected_centroids[
                best_detection_index
            ]


            self.previous_positions[object_id] = (
                self.objects[object_id]
            )


            self.objects[object_id] = new_centroid

            self.missed_frames[object_id] = 0


            used_objects.add(best_object_index)

            used_detections.add(best_detection_index)


        # ----------------------------------------------------
        # Handle objects that were not matched
        # ----------------------------------------------------

        for object_index, object_id in enumerate(object_ids):

            if object_index not in used_objects:

                self.missed_frames[object_id] += 1

                if self.missed_frames[object_id] > self.max_missed_frames:

                    self.deregister(object_id)


        # ----------------------------------------------------
        # Register completely new detections
        # ----------------------------------------------------

        for detection_index, centroid in enumerate(
            detected_centroids
        ):

            if detection_index not in used_detections:

                self.register(centroid)


        return self.objects


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    print("ERROR: Could not open video.")

    print(VIDEO_PATH)

    exit()


print("Video opened successfully.")


# ============================================================
# BACKGROUND SUBTRACTOR
# ============================================================

background_subtractor = cv2.createBackgroundSubtractorMOG2(

    history=500,

    varThreshold=50,

    detectShadows=True
)


# ============================================================
# MORPHOLOGY KERNEL
# ============================================================

kernel = cv2.getStructuringElement(

    cv2.MORPH_ELLIPSE,

    (5, 5)
)


# ============================================================
# TRACKER
# ============================================================

tracker = CentroidTracker(

    max_distance=MAX_DISTANCE,

    max_missed_frames=MAX_MISSED_FRAMES
)


# ============================================================
# COUNTING STATE
# ============================================================

counted_ids = set()

vehicle_count = 0


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(

    OUTPUT_CSV,

    "w",

    newline=""
)


csv_writer = csv.writer(csv_file)


csv_writer.writerow([

    "track_id",

    "event",

    "frame_number",

    "centroid_x",

    "centroid_y"

])


# ============================================================
# FRAME COUNTER
# ============================================================

frame_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()


    if not success:

        print("Video finished.")

        break


    frame_number += 1


    # ========================================================
    # BACKGROUND SUBTRACTION
    # ========================================================

    mask = background_subtractor.apply(frame)


    # ========================================================
    # REMOVE SHADOWS
    # ========================================================

    _, mask = cv2.threshold(

        mask,

        200,

        255,

        cv2.THRESH_BINARY
    )


    # ========================================================
    # MORPHOLOGICAL OPENING
    # Removes small noise
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel
    )


    # ========================================================
    # MORPHOLOGICAL CLOSING
    # Connects broken regions
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    detected_objects = []

    detected_centroids = []


    # ========================================================
    # PROCESS EACH CONTOUR
    # ========================================================

    for contour in contours:

        area = cv2.contourArea(contour)


        # ----------------------------------------------------
        # Ignore small objects/noise
        # ----------------------------------------------------

        if area < MIN_AREA:

            continue


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(contour)


        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        M = cv2.moments(contour)


        if M["m00"] == 0:

            continue


        cx = int(M["m10"] / M["m00"])

        cy = int(M["m01"] / M["m00"])


        detected_objects.append(

            (x, y, w, h, cx, cy)
        )


        detected_centroids.append(

            (cx, cy)
        )


    # ========================================================
    # UPDATE TRACKER
    # ========================================================

    objects = tracker.update(

        detected_centroids
    )


    # ========================================================
    # DRAW VIRTUAL TRIPWIRE
    # ========================================================

    cv2.line(

        frame,

        (0, LINE_Y),

        (frame.shape[1], LINE_Y),

        (0, 255, 255),

        3
    )


    cv2.putText(

        frame,

        "VIRTUAL TRIPWIRE",

        (20, LINE_Y - 15),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )


    # ========================================================
    # DRAW DETECTED OBJECTS
    # ========================================================

    for x, y, w, h, cx, cy in detected_objects:

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )


        cv2.circle(

            frame,

            (cx, cy),

            5,

            (0, 0, 255),

            -1
        )


    # ========================================================
    # DRAW TRACKED OBJECTS
    # ========================================================

    for object_id, centroid in objects.items():

        cx, cy = centroid


        cv2.circle(

            frame,

            (cx, cy),

            6,

            (255, 0, 0),

            -1
        )


        cv2.putText(

            frame,

            f"Vehicle {object_id}",

            (cx + 10, cy),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        # ====================================================
        # GET PREVIOUS POSITION
        # ====================================================

        previous_position = tracker.previous_positions.get(

            object_id
        )


        if previous_position is None:

            continue


        previous_y = previous_position[1]


        # ====================================================
        # LINE CROSSING
        # ====================================================

        crossed_down = (

            previous_y < LINE_Y

            and cy >= LINE_Y
        )


        crossed_up = (

            previous_y > LINE_Y

            and cy <= LINE_Y
        )


        crossed = crossed_down or crossed_up


        # ====================================================
        # COUNT ONLY ONCE
        # ====================================================

        if crossed and object_id not in counted_ids:

            vehicle_count += 1

            counted_ids.add(object_id)


            event_direction = (

                "INBOUND"

                if crossed_down

                else

                "OUTBOUND"
            )


            csv_writer.writerow([

                object_id,

                event_direction,

                frame_number,

                cx,

                cy
            ])


            csv_file.flush()


            print(

                f"Vehicle {object_id} crossed "

                f"{event_direction}. "

                f"Total: {vehicle_count}"
            )


    # ========================================================
    # DISPLAY COUNT
    # ========================================================

    cv2.putText(

        frame,

        f"Vehicles Counted: {vehicle_count}",

        (30, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (255, 255, 255),

        3
    )


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(

        "TrafficVision - OpenCV Vehicle Tracking",

        frame
    )


    # ========================================================
    # DISPLAY MASK
    # ========================================================

    cv2.imshow(

        "MOG2 Foreground Mask",

        mask
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("==============================")

print("TRAFFICVISION FINAL RESULT")

print("==============================")

print(

    f"Total vehicles counted: {vehicle_count}"
)

print(

    f"CSV saved to: {OUTPUT_CSV}"
)
