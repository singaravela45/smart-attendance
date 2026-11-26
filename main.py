import csv
import os
import re
import shutil
import signal
import smtplib
import sys
import threading
import time
import uuid

from datetime import datetime, date
from email.message import EmailMessage
from pathlib import Path
from email.utils import parseaddr

import cv2
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "unknown_faces"
MODEL_DIR = BASE_DIR / "models"

HAAR_CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"

ATTENDANCE_FILE = LOG_DIR / "attendance.csv"
UNKNOWN_EVENTS_FILE = LOG_DIR / "unknown_events.csv"

# Create necessary directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

def load_env_file():
    """
    Load variables from .env without requiring python-dotenv.
    Existing system environment variables take priority.
    """

    env_file = BASE_DIR / ".env"

    if not env_file.exists():
        print("[WARNING] .env file not found.")
        return

    with open(env_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip().strip('"').strip("'")

            os.environ.setdefault(key, value)


load_env_file()


# ============================================================
# APPLICATION SETTINGS
# ============================================================

CAMERA_SETTING = os.getenv("CAMERA_INDEX", "auto").strip().lower()

FACE_CONFIDENCE_THRESHOLD = float(
    os.getenv("FACE_CONFIDENCE_THRESHOLD", "70")
)

UNKNOWN_FRAMES_REQUIRED = int(
    os.getenv("UNKNOWN_FRAMES_REQUIRED", "5")
)

UNKNOWN_CLEAR_FRAMES_REQUIRED = int(
    os.getenv("UNKNOWN_CLEAR_FRAMES_REQUIRED", "15")
)

ALERT_COOLDOWN_SECONDS = int(
    os.getenv("ALERT_COOLDOWN_SECONDS", "600")
)

# Face processing settings
FACE_WIDTH = 200
FACE_HEIGHT = 200

DETECTION_SCALE = 0.5

# Camera recovery settings
CAMERA_FAILURE_LIMIT = 10


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "").strip()


def valid_email_address(address):
    """Basic email address validation."""

    parsed_address = parseaddr(address)[1]

    if parsed_address != address:
        return False

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return bool(re.match(pattern, address))


def email_configured():
    """Check whether valid email credentials are configured."""

    placeholders = {
        "",
        "xxxx",
        "your_email",
        "your_password",
        "your_app_password",
        "your_16_character_gmail_app_password",
        "example@gmail.com"
    }

    sender = SENDER_EMAIL.lower()
    receiver = RECEIVER_EMAIL.lower()
    password = GMAIL_APP_PASSWORD.lower().replace(" ", "")

    if sender in placeholders:
        print("[EMAIL] Sender email is missing or a placeholder.")
        return False

    if receiver in placeholders:
        print("[EMAIL] Receiver email is missing or a placeholder.")
        return False

    if password in placeholders:
        print("[EMAIL] Gmail App Password is missing or a placeholder.")
        return False

    if not valid_email_address(SENDER_EMAIL):
        print("[EMAIL] Invalid sender email address.")
        return False

    if not valid_email_address(RECEIVER_EMAIL):
        print("[EMAIL] Invalid receiver email address.")
        return False

    if not password:
        print("[EMAIL] Gmail App Password is empty.")
        return False

    return True


EMAIL_ENABLED = email_configured()

if EMAIL_ENABLED:
    print("[EMAIL] Email notifications configured.")
else:
    print("[EMAIL] Email notifications disabled.")


# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================

def handle_shutdown(signum, frame):
    print("\n[SHUTDOWN] Shutdown signal received.")
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, handle_shutdown)


# ============================================================
# MIGRATE OLD ATTENDANCE FILE
# ============================================================

def migrate_old_attendance_file():
    """
    Move an existing attendance.csv from the project root
    into logs/ if the destination does not already exist.
    """

    old_file = BASE_DIR / "attendance.csv"

    if old_file.exists() and not ATTENDANCE_FILE.exists():

        shutil.move(str(old_file), str(ATTENDANCE_FILE))

        print("[MIGRATION] Old attendance.csv moved into logs/")

    elif old_file.exists() and ATTENDANCE_FILE.exists():

        print(
            "[WARNING] Both root attendance.csv and "
            "logs/attendance.csv exist. The root file was left untouched."
        )


migrate_old_attendance_file()


# ============================================================
# CSV INITIALIZATION
# ============================================================

ATTENDANCE_HEADERS = [
    "Name",
    "Date",
    "Time",
    "Status"
]

UNKNOWN_EVENT_HEADERS = [
    "Event ID",
    "Date",
    "Timestamp",
    "Photograph",
    "Recognition Confidence"
]


def initialize_csv_files():

    if not ATTENDANCE_FILE.exists():

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(ATTENDANCE_HEADERS)

    if not UNKNOWN_EVENTS_FILE.exists():

        with open(
            UNKNOWN_EVENTS_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(UNKNOWN_EVENT_HEADERS)


initialize_csv_files()


# ============================================================
# ATTENDANCE MANAGEMENT
# ============================================================

today_date = date.today().strftime("%Y-%m-%d")

attendance_today = set()


def load_today_attendance():
    """Load people who already have attendance recorded today."""

    if not ATTENDANCE_FILE.exists():
        return

    with open(
        ATTENDANCE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("Date") == today_date:
                attendance_today.add(row.get("Name", "").strip())


def record_attendance(person_name):
    """Record attendance only once per person per day."""

    if person_name in attendance_today:
        return

    now = datetime.now()

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            person_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    attendance_today.add(person_name)

    print(
        f"[ATTENDANCE] {person_name} marked present "
        f"at {now.strftime('%H:%M:%S')}"
    )


load_today_attendance()


# ============================================================
# FACE DETECTION MODEL
# ============================================================

face_detector = cv2.CascadeClassifier(
    str(HAAR_CASCADE_PATH)
)

if face_detector.empty():
    raise RuntimeError(
        f"Unable to load Haar Cascade model: {HAAR_CASCADE_PATH}"
    )


# ============================================================
# LBPH FACE RECOGNIZER
# ============================================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is unavailable. Install an OpenCV build "
        "that includes the contrib face module."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

label_to_name = {}


def preprocess_face(face):
    """Prepare a face image for LBPH recognition."""

    face = cv2.resize(
        face,
        (FACE_WIDTH, FACE_HEIGHT)
    )

    face = cv2.equalizeHist(face)

    return face


# ============================================================
# TRAINING DATASET
# ============================================================

def train_recognizer():

    training_faces = []
    training_labels = []

    label_to_name.clear()

    if not KNOWN_FACES_DIR.exists():
        raise RuntimeError("known_faces directory does not exist.")

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_files = sorted(
        file for file in KNOWN_FACES_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    )

    if not image_files:
        raise RuntimeError(
            "No training images found in known_faces."
        )

    name_to_label = {}

    for image_path in image_files:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[TRAINING] Could not read {image_path.name}")
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(detected_faces) == 0:
            print(
                f"[TRAINING] No face detected in {image_path.name}"
            )
            continue

        # Select the largest face in the training image.
        x, y, w, h = max(
            detected_faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y + h, x:x + w]

        face = preprocess_face(face)

        # Example: john_1.jpg -> john
        person_name = image_path.stem.rsplit("_", 1)[0]

        person_name = person_name.replace("_", " ").strip().title()

        if not person_name:
            continue

        if person_name not in name_to_label:

            label = len(name_to_label)

            name_to_label[person_name] = label
            label_to_name[label] = person_name

        label = name_to_label[person_name]

        training_faces.append(face)
        training_labels.append(label)

        print(
            f"[TRAINING] Loaded {image_path.name} "
            f"for {person_name}"
        )

    if not training_faces:
        raise RuntimeError(
            "No usable face images found for training."
        )

    recognizer.train(
        training_faces,
        np.array(training_labels, dtype=np.int32)
    )

    print("\n[TRAINING] Training completed.")

    for name, label in name_to_label.items():

        count = training_labels.count(label)

        print(
            f"[TRAINING] {name}: {count} training images"
        )

    print(
        f"[TRAINING] Total images: {len(training_faces)}"
    )


train_recognizer()


# ============================================================
# CAMERA MANAGEMENT
# ============================================================

def get_camera_indices():
    """Return camera indices in the order they should be tested."""

    if CAMERA_SETTING == "auto":
        return [0, 1]

    try:
        indices = [
            int(value.strip())
            for value in CAMERA_SETTING.split(",")
            if value.strip()
        ]

        if not indices:
            raise ValueError

        return list(dict.fromkeys(indices))

    except ValueError:
        print(
            "[CAMERA] Invalid CAMERA_INDEX setting. "
            "Using automatic detection."
        )

        return [0, 1]


def open_camera():
    """Find an available camera and verify that it produces a frame."""

    for camera_index in get_camera_indices():

        print(
            f"[CAMERA] Trying camera index {camera_index}..."
        )

        camera = cv2.VideoCapture(camera_index)

        if not camera.isOpened():
            camera.release()
            continue

        # Give the camera a moment to initialize.
        time.sleep(0.5)

        success, frame = camera.read()

        if success and frame is not None:

            print(
                f"[CAMERA] Camera {camera_index} connected."
            )

            return camera

        camera.release()

    return None


# ============================================================
# UNKNOWN EVENT LOGGING
# ============================================================

def generate_event_id():
    """Generate a unique ID for each unknown-person visit."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    unique_part = uuid.uuid4().hex[:8].upper()

    return f"UNKNOWN-{timestamp}-{unique_part}"


def save_unknown_event(face_image, confidence):
    """
    Save one unknown-person photograph and its CSV event record.

    This function is called once per continuous unknown-person visit.
    """

    now = datetime.now()

    event_id = generate_event_id()

    photograph_name = f"{event_id}.jpg"

    photograph_path = UNKNOWN_FACES_DIR / photograph_name

    saved = cv2.imwrite(
        str(photograph_path),
        face_image
    )

    if not saved:
        print("[EVENT] Failed to save unknown-person photograph.")
        return None, None

    with open(
        UNKNOWN_EVENTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            event_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            photograph_name,
            f"{confidence:.2f}"
        ])

    print("\n[UNKNOWN EVENT] New event recorded.")

    print(f"Event ID: {event_id}")
    print(f"Date: {now.strftime('%Y-%m-%d')}")
    print(f"Timestamp: {now.strftime('%H:%M:%S')}")
    print(f"Photograph: {photograph_name}")
    print(f"Recognition confidence: {confidence:.2f}")

    return event_id, photograph_path


# ============================================================
# EMAIL NOTIFICATIONS
# ============================================================

def send_email_alert(photograph_path, event_id, confidence):
    """Send an unknown-person email with the saved photograph."""

    try:

        message = EmailMessage()

        message["Subject"] = "Unknown Person Detected"

        message["From"] = SENDER_EMAIL
        message["To"] = RECEIVER_EMAIL

        message.set_content(
            f"""
An unknown person was detected by the Smart Attendance System.

Event ID: {event_id}

Date: {datetime.now().strftime('%Y-%m-%d')}

Recognition confidence: {confidence:.2f}

Please find the photograph attached.
"""
        )

        with open(photograph_path, "rb") as image_file:

            image_data = image_file.read()

        message.add_attachment(
            image_data,
            maintype="image",
            subtype="jpeg",
            filename=photograph_path.name
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=15
        ) as smtp:

            smtp.starttls()

            smtp.login(
                SENDER_EMAIL,
                GMAIL_APP_PASSWORD
            )

            smtp.send_message(message)

        print(
            f"[EMAIL] Alert sent successfully for {event_id}."
        )

    except Exception as error:

        print(
            f"[EMAIL ERROR] Could not send alert for "
            f"{event_id}: {error}"
        )

        # No queue or automatic retry is created.


# ============================================================
# EMAIL COOLDOWN MANAGEMENT
# ============================================================

last_email_attempt = 0.0

email_thread = None


def trigger_email_alert(photograph_path, event_id, confidence):
    """
    Allow one email attempt every configured cooldown period.

    Logging is independent of email notification.
    """

    global last_email_attempt
    global email_thread

    if not EMAIL_ENABLED:
        print("[EMAIL] Skipped: email is not configured.")
        return

    current_time = time.monotonic()

    if (
        current_time - last_email_attempt
        < ALERT_COOLDOWN_SECONDS
    ):

        remaining = int(
            ALERT_COOLDOWN_SECONDS
            - (current_time - last_email_attempt)
        )

        print(
            f"[EMAIL] Alert suppressed by cooldown. "
            f"Remaining: {remaining} seconds."
        )

        return

    if email_thread is not None and email_thread.is_alive():

        print(
            "[EMAIL] Previous email is still being processed. "
            "Skipping this alert."
        )

        return

    # Record the attempt time before starting the thread.
    # A failed attempt does not create a queued email.
    last_email_attempt = current_time

    email_thread = threading.Thread(
        target=send_email_alert,
        args=(photograph_path, event_id, confidence),
        daemon=True
    )

    email_thread.start()


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(face_image):

    processed_face = preprocess_face(face_image)

    label, confidence = recognizer.predict(
        processed_face
    )

    if confidence <= FACE_CONFIDENCE_THRESHOLD:

        person_name = label_to_name.get(label)

        if person_name is not None:
            return person_name, confidence

    return None, confidence


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("\n========================================")
    print("       SMART ATTENDANCE SYSTEM")
    print("========================================\n")

    camera = open_camera()

    if camera is None:

        raise RuntimeError(
            "No working camera found. "
            "Check CAMERA_INDEX and camera connection."
        )

    unknown_frame_count = 0
    unknown_clear_count = 0

    unknown_event_active = False

    camera_failure_count = 0

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:

                camera_failure_count += 1

                print(
                    f"[CAMERA] Frame capture failed "
                    f"({camera_failure_count}/"
                    f"{CAMERA_FAILURE_LIMIT})."
                )

                if camera_failure_count >= CAMERA_FAILURE_LIMIT:

                    print(
                        "[CAMERA] Attempting camera recovery..."
                    )

                    camera.release()

                    camera = open_camera()

                    camera_failure_count = 0

                    # Reset visit tracking after camera recovery.
                    unknown_frame_count = 0
                    unknown_clear_count = 0
                    unknown_event_active = False

                    if camera is None:

                        print(
                            "[CAMERA] Camera unavailable. "
                            "Retrying in 5 seconds."
                        )

                        time.sleep(5)

                        camera = open_camera()

                        if camera is None:
                            continue

                time.sleep(0.1)
                continue

            camera_failure_count = 0

            # Convert original frame to grayscale.
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Downscale only for faster face detection.
            small_gray = cv2.resize(
                gray,
                None,
                fx=DETECTION_SCALE,
                fy=DETECTION_SCALE
            )

            detected_faces = face_detector.detectMultiScale(
                small_gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
            )

            unknown_faces = []

            # =================================================
            # PROCESS EACH DETECTED FACE
            # =================================================

            for (small_x, small_y, small_w, small_h) in detected_faces:

                # Convert coordinates to original frame size.
                x = int(small_x / DETECTION_SCALE)
                y = int(small_y / DETECTION_SCALE)

                w = int(small_w / DETECTION_SCALE)
                h = int(small_h / DETECTION_SCALE)

                # Keep coordinates within image boundaries.
                x = max(0, x)
                y = max(0, y)

                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w <= 0 or h <= 0:
                    continue

                face_image = gray[
                    y:y + h,
                    x:x + w
                ]

                person_name, confidence = recognize_face(
                    face_image
                )

                if person_name is not None:

                    # Known person.
                    color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"{person_name} ({confidence:.1f})",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    record_attendance(person_name)

                else:

                    # Unknown person.
                    color = (0, 0, 255)

                    unknown_faces.append(
                        (face_image.copy(), confidence)
                    )

                    cv2.putText(
                        frame,
                        f"Unknown ({confidence:.1f})",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # Draw rectangle around unknown face.
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2
                    )

                    continue

                # Draw rectangle around recognized face.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

            # =================================================
            # UNKNOWN PERSON VISIT TRACKING
            # =================================================

            if unknown_faces:

                unknown_frame_count += 1
                unknown_clear_count = 0

                # LBPH uses lower confidence/distance values
                # for closer matches. Choose the lowest distance
                # among the currently unknown faces.
                best_unknown_face, best_confidence = min(
                    unknown_faces,
                    key=lambda item: item[1]
                )

                # Create one event after the required number
                # of consecutive frames containing an unknown face.
                if (
                    unknown_frame_count >= UNKNOWN_FRAMES_REQUIRED
                    and not unknown_event_active
                ):

                    unknown_event_active = True

                    event_id, photograph_path = save_unknown_event(
                        best_unknown_face,
                        best_confidence
                    )

                    if event_id is not None:

                        trigger_email_alert(
                            photograph_path,
                            event_id,
                            best_confidence
                        )

            else:

                unknown_frame_count = 0

                if unknown_event_active:

                    unknown_clear_count += 1

                    # End the visit after enough consecutive
                    # frames without an unknown person.
                    if (
                        unknown_clear_count
                        >= UNKNOWN_CLEAR_FRAMES_REQUIRED
                    ):

                        unknown_event_active = False
                        unknown_clear_count = 0

                        print(
                            "[UNKNOWN EVENT] Unknown-person visit ended."
                        )

            # =================================================
            # HEADLESS OPERATION
            # =================================================

            # No cv2.imshow() is used.
            # This application can run without a display.

    except KeyboardInterrupt:

        print("\n[APPLICATION] Stopping application...")

    finally:

        camera.release()

        if email_thread is not None and email_thread.is_alive():

            print(
                "[APPLICATION] Waiting briefly for email operation..."
            )

            email_thread.join(timeout=5)

        print("[CAMERA] Camera released.")

        print("[APPLICATION] Smart Attendance System stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(f"\n[FATAL ERROR] {error}")

        sys.exit(1)
