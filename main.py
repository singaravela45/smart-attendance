
#!/usr/bin/env python3
"""Smart attendance system: face recognition, CSV attendance, email alerts."""

import csv
import os
import re
import signal
import smtplib
import sys
import threading
import time

from datetime import date, datetime
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

def load_env_file(env_path):
    if not env_path.is_file():
        return

    with open(env_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, separator, value = line.partition("=")

            if separator:
                os.environ.setdefault(
                    key.strip(),
                    value.strip().strip('"').strip("'")
                )


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        print(f"[WARNING] Invalid integer for {name}; using {default}.")
        return default


def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        print(f"[WARNING] Invalid number for {name}; using {default}.")
        return default


load_env_file(BASE_DIR / ".env")


SENDER_EMAIL = os.getenv("SENDER_EMAIL", "").strip()
SENDER_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "").strip()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

KNOWN_FACES_DIR = BASE_DIR / "known_faces"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"


# Camera configuration:
# auto = test camera 0 and then camera 1.
# 0 or 1 = manually select a camera.
# 0,1 = manually specify camera priority.

CAMERA_SETTING = os.getenv("CAMERA_INDEX", "auto").strip().lower()


# Face recognition.
FACE_CONFIDENCE_THRESHOLD = env_float(
    "FACE_CONFIDENCE_THRESHOLD", 70
)

# Unknown face detection.
UNKNOWN_FRAMES_REQUIRED = env_int(
    "UNKNOWN_FRAMES_REQUIRED", 5
)

UNKNOWN_CLEAR_FRAMES_REQUIRED = env_int(
    "UNKNOWN_CLEAR_FRAMES_REQUIRED", 15
)

# Global cooldown between all unauthorized-person email alerts.
ALERT_COOLDOWN_SECONDS = 600  # 10 minutes

# Logging intervals.
RECOGNIZED_LOG_INTERVAL = 10
UNKNOWN_LOG_INTERVAL = 5

# Camera recovery.
MAX_READ_FAILURES = 10
MAX_CAMERA_RETRIES = 5
CAMERA_RETRY_DELAY = 2.0

# Attendance retry.
ATTENDANCE_RETRY_SECONDS = 10

FACE_WIDTH = 200
FACE_HEIGHT = 200

DETECT_SCALE = 0.5

ATTENDANCE_MARKED = "marked"
ATTENDANCE_ALREADY = "already"
ATTENDANCE_ERROR = "error"


# ==========================================
# EMAIL VALIDATION
# ==========================================

def valid_email_address(address):
    if not address:
        return False

    _, parsed_address = parseaddr(address)

    if parsed_address != address:
        return False

    return bool(re.fullmatch(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
        address
    ))


def email_configured():
    placeholders = {
        "your mail id",
        "your email",
        "your_email@gmail.com",
        "your app password",
        "your password",
        "your receiver email"
    }

    if not valid_email_address(SENDER_EMAIL):
        return False

    if not valid_email_address(RECEIVER_EMAIL):
        return False

    if not SENDER_PASSWORD:
        return False

    if SENDER_EMAIL.lower() in placeholders:
        return False

    if RECEIVER_EMAIL.lower() in placeholders:
        return False

    if SENDER_PASSWORD.lower() in placeholders:
        return False

    return True


def validate_email_configuration():
    if not SENDER_EMAIL or not RECEIVER_EMAIL:
        print("[WARNING] Email addresses are missing.")
        return False

    if not valid_email_address(SENDER_EMAIL):
        print("[WARNING] Invalid SENDER_EMAIL format.")
        return False

    if not valid_email_address(RECEIVER_EMAIL):
        print("[WARNING] Invalid RECEIVER_EMAIL format.")
        return False

    if not SENDER_PASSWORD:
        print("[WARNING] Gmail app password is missing.")
        return False

    if not email_configured():
        print("[WARNING] Placeholder email settings detected.")
        return False

    print("[INFO] Email configuration validated.")
    return True


# ==========================================
# INITIALIZE OPENCV
# ==========================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is not available. OpenCV was built without "
        "the contrib modules. Enable the contrib PACKAGECONFIG "
        "for OpenCV in Yocto."
    )


face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError(
        f"Could not load Haar Cascade model: {CASCADE_PATH}"
    )


def preprocess_face(face):
    face = cv2.resize(face, (FACE_WIDTH, FACE_HEIGHT))
    face = cv2.equalizeHist(face)
    return face


def detect_faces(gray_image, min_size):
    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size
    )

    return list(faces)


def detect_largest_face(gray_image):
    faces = detect_faces(gray_image, (60, 60))

    if not faces:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    return x, y, w, h


# ==========================================
# LOAD AND TRAIN KNOWN FACES
# ==========================================

def load_known_faces():
    training_faces = []
    training_labels = []
    label_to_name = {}

    if not KNOWN_FACES_DIR.is_dir():
        print(
            f"[ERROR] known_faces directory not found: "
            f"{KNOWN_FACES_DIR}"
        )
        return None, {}

    current_label = 0

    for path in sorted(KNOWN_FACES_DIR.iterdir()):

        if path.suffix.lower() not in (
            ".jpg", ".jpeg", ".png"
        ):
            continue

        try:
            # john_1.jpg, john_2.jpg -> John

            name_part = path.stem
            parts = name_part.rsplit("_", 1)

            if len(parts) == 2 and parts[1].isdigit():
                name_part = parts[0]

            name = name_part.replace("_", " ").title()

            image = cv2.imread(str(path))

            if image is None:
                print(f"[WARNING] Cannot read {path.name}")
                continue

            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            face_location = detect_largest_face(gray)

            if face_location is None:
                print(
                    f"[WARNING] No face found in {path.name}"
                )
                continue

            x, y, w, h = face_location

            face = preprocess_face(
                gray[y:y + h, x:x + w]
            )

            if name not in label_to_name:
                label_to_name[name] = current_label
                current_label += 1

            training_faces.append(face)
            training_labels.append(label_to_name[name])

            print(
                f"[INFO] Loaded {path.name} -> {name}"
            )

        except Exception as e:
            print(
                f"[ERROR] Failed to load {path.name}: {e}"
            )

    if not training_faces:
        return None, {}

    print(
        f"[INFO] Training with {len(training_faces)} images "
        f"for {len(label_to_name)} people."
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    recognizer.train(
        training_faces,
        np.array(training_labels, dtype=np.int32)
    )

    print("[INFO] Face training completed.")

    return recognizer, {
        label: name
        for name, label in label_to_name.items()
    }


# ==========================================
# ATTENDANCE MANAGEMENT
# ==========================================

def mark_attendance(name):
    today = date.today().isoformat()

    if ATTENDANCE_FILE.exists():
        try:
            with open(
                ATTENDANCE_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                for row in csv.DictReader(file):
                    if (
                        row.get("Name") == name
                        and row.get("Date") == today
                    ):
                        return ATTENDANCE_ALREADY

        except (OSError, csv.Error) as e:
            print(f"[ERROR] Attendance read failed: {e}")
            return ATTENDANCE_ERROR

    current_time = datetime.now().strftime("%H:%M:%S")

    write_header = (
        not ATTENDANCE_FILE.exists()
        or ATTENDANCE_FILE.stat().st_size == 0
    )

    try:
        with open(
            ATTENDANCE_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if write_header:
                writer.writerow([
                    "Name", "Date", "Time", "Status"
                ])

            writer.writerow([
                name,
                today,
                current_time,
                "Present"
            ])

        print(
            f"[ATTENDANCE] {name} marked present "
            f"at {current_time}"
        )

        return ATTENDANCE_MARKED

    except OSError as e:
        print(f"[ERROR] Attendance write failed: {e}")
        return ATTENDANCE_ERROR


# ==========================================
# EMAIL ALERT
# ==========================================

def send_email_alert(image_frame):
    """Send an email alert in a background thread."""

    if not email_configured():
        print(
            "[ERROR] Email not configured. "
            "Please check your .env file."
        )
        return False

    try:
        success, encoded_image = cv2.imencode(
            ".jpg", image_frame
        )

        if not success:
            print("[ERROR] Could not encode alert image.")
            return False

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        msg = EmailMessage()

        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = (
            "ALERT: Unauthorized Person Detected!"
        )

        msg.set_content(
            "An unknown face was detected.\n\n"
            f"Detection time: {timestamp}\n"
            "Captured image is attached."
        )

        msg.add_attachment(
            encoded_image.tobytes(),
            maintype="image",
            subtype="jpeg",
            filename="unauthorized_detected.jpg"
        )

        with smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT,
            timeout=20
        ) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(
                SENDER_EMAIL,
                SENDER_PASSWORD
            )

            server.send_message(msg)

        print("[SUCCESS] Alert email sent.")
        return True

    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
        return False


def start_alert_thread(frame):
    thread = threading.Thread(
        target=send_email_alert,
        args=(frame.copy(),),
        daemon=True
    )

    thread.start()
    return thread


# ==========================================
# CAMERA MANAGEMENT
# ==========================================

def get_camera_indices():
    """
    auto -> [0, 1]
    0    -> [0]
    1    -> [1]
    0,1  -> [0, 1]
    """

    if CAMERA_SETTING == "auto":
        return [0, 1]

    try:
        indices = [
            int(value.strip())
            for value in CAMERA_SETTING.split(",")
        ]

        if not indices or any(index < 0 for index in indices):
            raise ValueError

        return list(dict.fromkeys(indices))

    except ValueError:
        print(
            "[WARNING] Invalid CAMERA_INDEX setting. "
            "Using automatic camera detection."
        )
        return [0, 1]


def open_camera():
    """Try available camera indexes and return the first working camera."""

    for index in get_camera_indices():

        print(f"[INFO] Trying camera index {index}...")

        capture = cv2.VideoCapture(index)

        if not capture.isOpened():
            capture.release()
            print(
                f"[WARNING] Camera {index} could not be opened."
            )
            continue

        ret, frame = capture.read()

        if ret and frame is not None and frame.size > 0:
            print(
                f"[SUCCESS] Camera {index} is active."
            )
            return capture, index

        print(
            f"[WARNING] Camera {index} opened "
            "but cannot provide frames."
        )

        capture.release()

    return None, None


def handle_sigterm(signum, frame):
    raise KeyboardInterrupt


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    recognizer, label_to_name = load_known_faces()

    if recognizer is None:
        print("[ERROR] No usable authorized faces.")
        return 1

    signal.signal(signal.SIGTERM, handle_sigterm)

    validate_email_configuration()

    capture = None
    active_camera_index = None

    read_failures = 0
    reopen_failures = 0

    alert_thread = None

    unknown_streak = 0
    unknown_clear_streak = 0

    # One alert per unknown-person visit.
    unknown_visit_alerted = False

    # Global cooldown shared by every unknown person.
    # 0 means an alert can be sent immediately at startup.
    last_alert_time = 0.0

    attendance_done = {}
    attendance_retry_at = {}
    last_logged = {}

    last_unknown_log = 0.0

    print("[INFO] Smart Attendance System started.")
    print("[INFO] Camera auto-detection enabled.")
    print("[INFO] Global alert cooldown: 10 minutes.")
    print("[INFO] Running in headless mode.")
    print("[INFO] Press Ctrl+C to stop.")

    try:

        while True:

            # ----------------------------------
            # CAMERA CONNECTION / RECONNECTION
            # ----------------------------------

            if capture is None:

                capture, active_camera_index = open_camera()

                if capture is None:

                    reopen_failures += 1

                    if reopen_failures >= MAX_CAMERA_RETRIES:
                        print(
                            "[ERROR] No working camera found. "
                            "Exiting."
                        )
                        return 1

                    print(
                        f"[WARNING] Camera unavailable "
                        f"({reopen_failures}/"
                        f"{MAX_CAMERA_RETRIES}). Retrying..."
                    )

                    time.sleep(CAMERA_RETRY_DELAY)
                    continue

                reopen_failures = 0
                read_failures = 0

                print(
                    f"[INFO] Using camera "
                    f"index {active_camera_index}."
                )

            ret, frame = capture.read()

            if not ret or frame is None:

                read_failures += 1

                if read_failures >= MAX_READ_FAILURES:

                    print(
                        "[ERROR] Camera frames unavailable. "
                        "Reopening camera."
                    )

                    capture.release()
                    capture = None
                    active_camera_index = None

                else:
                    time.sleep(0.1)

                continue

            read_failures = 0

            # ----------------------------------
            # FACE DETECTION
            # ----------------------------------

            gray_full = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            frame_h, frame_w = gray_full.shape[:2]

            gray_small = cv2.resize(
                gray_full,
                None,
                fx=DETECT_SCALE,
                fy=DETECT_SCALE
            )

            detected_faces = detect_faces(
                gray_small,
                (40, 40)
            )

            now = time.time()
            today = date.today()

            unknown_in_frame = False
            unknown_confidences = []

            # ----------------------------------
            # FACE RECOGNITION
            # ----------------------------------

            for (x, y, w, h) in detected_faces:

                x0 = max(
                    int(x / DETECT_SCALE), 0
                )

                y0 = max(
                    int(y / DETECT_SCALE), 0
                )

                x1 = min(
                    int((x + w) / DETECT_SCALE),
                    frame_w
                )

                y1 = min(
                    int((y + h) / DETECT_SCALE),
                    frame_h
                )

                face = gray_full[y0:y1, x0:x1]

                if face.size == 0:
                    continue

                face = preprocess_face(face)

                label, confidence = recognizer.predict(face)

                # Lower LBPH confidence means a better match.

                if (
                    confidence < FACE_CONFIDENCE_THRESHOLD
                    and label in label_to_name
                ):

                    name = label_to_name[label]

                    # Attendance: once per person per day.

                    if (
                        attendance_done.get(name) != today
                        and now >= attendance_retry_at.get(name, 0)
                    ):

                        status = mark_attendance(name)

                        if status == ATTENDANCE_ERROR:

                            attendance_retry_at[name] = (
                                now + ATTENDANCE_RETRY_SECONDS
                            )

                        else:
                            attendance_done[name] = today

                    # Throttled recognized-face logging.

                    if (
                        now - last_logged.get(name, 0)
                        >= RECOGNIZED_LOG_INTERVAL
                    ):

                        print(
                            f"[RECOGNIZED] {name} "
                            f"(Confidence: {confidence:.2f})"
                        )

                        last_logged[name] = now

                else:

                    unknown_in_frame = True
                    unknown_confidences.append(confidence)

            # ----------------------------------
            # UNKNOWN FACE CONFIDENCE LOGGING
            # ----------------------------------

            if unknown_in_frame:

                if now - last_unknown_log >= UNKNOWN_LOG_INTERVAL:

                    best_unknown_confidence = min(
                        unknown_confidences
                    )

                    print(
                        "[UNKNOWN] Face detected. "
                        f"Confidence: "
                        f"{best_unknown_confidence:.2f} "
                        f"(Threshold: "
                        f"{FACE_CONFIDENCE_THRESHOLD:.2f})"
                    )

                    last_unknown_log = now

            # ----------------------------------
            # UNKNOWN VISIT TRACKING
            # ----------------------------------

            if unknown_in_frame:

                unknown_streak += 1
                unknown_clear_streak = 0

            else:

                unknown_streak = 0

                if unknown_visit_alerted:

                    unknown_clear_streak += 1

                    if (
                        unknown_clear_streak
                        >= UNKNOWN_CLEAR_FRAMES_REQUIRED
                    ):

                        unknown_visit_alerted = False
                        unknown_clear_streak = 0

                        print(
                            "[INFO] Previous unknown-person "
                            "visit ended. Alert system reset."
                        )

            # ----------------------------------
            # GLOBAL 10-MINUTE ALERT COOLDOWN
            # ----------------------------------

            if (
                unknown_streak >= UNKNOWN_FRAMES_REQUIRED
                and not unknown_visit_alerted
            ):

                # Mark this visit as alerted even when
                # the global cooldown prevents an email.
                unknown_visit_alerted = True

                current_time = time.time()

                if (
                    current_time - last_alert_time
                    >= ALERT_COOLDOWN_SECONDS
                ):

                    print(
                        "[ALERT] Unauthorized person detected! "
                        "Sending email alert."
                    )

                    # Start global cooldown immediately.
                    last_alert_time = current_time

                    if email_configured():

                        if (
                            alert_thread is None
                            or not alert_thread.is_alive()
                        ):

                            alert_thread = start_alert_thread(frame)

                        else:

                            print(
                                "[WARNING] Previous email is still "
                                "being sent. Alert skipped."
                            )

                    else:

                        print(
                            "[WARNING] Email unavailable. "
                            "Alert recorded in application log only."
                        )

                else:

                    remaining = int(
                        ALERT_COOLDOWN_SECONDS
                        - (current_time - last_alert_time)
                    )

                    print(
                        "[INFO] Alert suppressed. "
                        f"Global cooldown active: "
                        f"{remaining} seconds remaining."
                    )

            time.sleep(0.03)

    except KeyboardInterrupt:

        print("\n[INFO] Stopping application...")

    finally:

        if capture is not None:
            capture.release()

        print("[INFO] Camera released.")
        print("[INFO] Smart Attendance System stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())