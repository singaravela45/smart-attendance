#!/usr/bin/env python3
"""Smart attendance system: face recognition, CSV attendance, email alerts."""
import csv
import os
import signal
import smtplib
import sys
import threading
import time
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

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

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

KNOWN_FACES_DIR = BASE_DIR / "known_faces"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"

# Camera index (your camera is 1; override with CAMERA_INDEX in .env if needed).
CAMERA_INDEX = env_int("CAMERA_INDEX", 1)

# LBPH confidence: LOWER means a better match. Calibrate with real data.
FACE_CONFIDENCE_THRESHOLD = env_float("FACE_CONFIDENCE_THRESHOLD", 70)

# Seconds between alert emails.
COOLDOWN_SECONDS = env_int("COOLDOWN_SECONDS", 30)

# An unknown face must appear in this many consecutive frames before alerting.
UNKNOWN_FRAMES_REQUIRED = env_int("UNKNOWN_FRAMES_REQUIRED", 5)

FACE_WIDTH = 200
FACE_HEIGHT = 200

# Detection runs on a downscaled frame for speed; recognition uses full-res crops.
DETECT_SCALE = 0.5

# Camera recovery.
MAX_READ_FAILURES = 10        # consecutive bad reads before reopening the camera
MAX_CAMERA_RETRIES = 5        # reopen attempts before exiting (systemd can restart)
CAMERA_RETRY_DELAY = 2.0      # seconds between reopen attempts

# Logging / retry throttles.
RECOGNIZED_LOG_INTERVAL = 10  # seconds between "[RECOGNIZED]" lines per person
ATTENDANCE_RETRY_SECONDS = 10 # wait before retrying a failed attendance write

# mark_attendance() results
ATTENDANCE_MARKED = "marked"
ATTENDANCE_ALREADY = "already"
ATTENDANCE_ERROR = "error"


# ==========================================
# INITIALIZE OPENCV
# ==========================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is not available. OpenCV was built without the contrib "
        "modules. In Yocto, enable the 'contrib' PACKAGECONFIG for opencv."
    )

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError(f"Could not load Haar Cascade model: {CASCADE_PATH}")

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

    # Select the largest detected face.
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return x, y, w, h


def load_known_faces():
    training_faces = []
    training_labels = []
    label_to_name = {}

    if not KNOWN_FACES_DIR.is_dir():
        print(f"[ERROR] known_faces directory not found: {KNOWN_FACES_DIR}")
        return None, {}

    current_label = 0

    for path in sorted(KNOWN_FACES_DIR.iterdir()):

        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        try:
            # john_1.jpg and john_2.jpg -> John
            name_part = path.stem
            parts = name_part.rsplit("_", 1)

            if len(parts) == 2 and parts[1].isdigit():
                name_part = parts[0]

            name = name_part.replace("_", " ").title()

            image = cv2.imread(str(path))

            if image is None:
                print(f"[WARNING] Cannot read {path.name}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_location = detect_largest_face(gray)

            if face_location is None:
                print(f"[WARNING] No face found in {path.name}")
                continue

            x, y, w, h = face_location
            face = preprocess_face(gray[y:y + h, x:x + w])

            # One label per person.
            if name not in label_to_name:
                label_to_name[name] = current_label
                current_label += 1

            training_faces.append(face)
            training_labels.append(label_to_name[name])

            print(f"[INFO] Loaded {path.name} -> {name}")

        except Exception as e:
            print(f"[ERROR] Failed to load {path.name}: {e}")

    if not training_faces:
        return None, {}

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

    return recognizer, {label: name for name, label in label_to_name.items()}


# ==========================================
# ATTENDANCE
# ==========================================

def mark_attendance(name):
    """Return ATTENDANCE_MARKED, ATTENDANCE_ALREADY, or ATTENDANCE_ERROR."""

    today = date.today().isoformat()

    if ATTENDANCE_FILE.exists():
        try:
            with open(
                ATTENDANCE_FILE, "r", newline="", encoding="utf-8"
            ) as file:
                for row in csv.DictReader(file):
                    if row.get("Name") == name and row.get("Date") == today:
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
            ATTENDANCE_FILE, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            if write_header:
                writer.writerow(["Name", "Date", "Time", "Status"])

            writer.writerow([name, today, current_time, "Present"])

        print(f"[ATTENDANCE] {name} marked present at {current_time}")
        return ATTENDANCE_MARKED

    except OSError as e:
        print(f"[ERROR] Attendance write failed: {e}")
        return ATTENDANCE_ERROR


# ==========================================
# EMAIL ALERT
# ==========================================

def email_configured():
    return bool(SENDER_EMAIL and SENDER_PASSWORD and RECEIVER_EMAIL)


def send_email_alert(image_frame):
    """Send the alert email. Runs in a background thread."""

    if not email_configured():
        print(
            "[ERROR] Email not configured "
            "(SENDER_EMAIL, GMAIL_APP_PASSWORD, RECEIVER_EMAIL)."
        )
        return False

    try:
        success, encoded_image = cv2.imencode(".jpg", image_frame)

        if not success:
            print("[ERROR] Could not encode alert image.")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "ALERT: Unauthorized Person Detected!"
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

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("[SUCCESS] Alert email sent.")
        return True

    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
        return False


def start_alert_thread(frame):
    """Send the email in the background so the camera loop never blocks."""
    thread = threading.Thread(
        target=send_email_alert,
        args=(frame.copy(),),
        daemon=True
    )
    thread.start()
    return thread


# ==========================================
# CAMERA
# ==========================================

def open_camera():
    capture = cv2.VideoCapture(CAMERA_INDEX)

    if capture.isOpened():
        return capture

    capture.release()
    return None


def handle_sigterm(signum, frame):
    # Lets `systemctl stop` shut down cleanly through the same path as Ctrl+C.
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

    if not email_configured():
        print("[WARNING] Email is not fully configured; alerts will only be logged.")

    capture = None
    read_failures = 0
    reopen_failures = 0

    last_alert_time = 0.0
    alert_thread = None
    unknown_streak = 0

    attendance_done = {}      # name -> date already handled
    attendance_retry_at = {}  # name -> time.time() after which to retry
    last_logged = {}          # name -> last "[RECOGNIZED]" log time

    print(f"[INFO] Smart Attendance System started (camera index {CAMERA_INDEX}).")
    print("[INFO] Running in headless mode.")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        while True:

            # ---- Camera (re)connection ----
            if capture is None:
                capture = open_camera()

                if capture is None:
                    reopen_failures += 1

                    if reopen_failures >= MAX_CAMERA_RETRIES:
                        print("[ERROR] Could not open camera. Exiting.")
                        return 1

                    print(
                        f"[WARNING] Camera unavailable "
                        f"({reopen_failures}/{MAX_CAMERA_RETRIES}); retrying..."
                    )
                    time.sleep(CAMERA_RETRY_DELAY)
                    continue

                reopen_failures = 0
                read_failures = 0

            ret, frame = capture.read()

            if not ret:
                read_failures += 1

                if read_failures >= MAX_READ_FAILURES:
                    print("[ERROR] Camera frames unavailable; reopening camera.")
                    capture.release()
                    capture = None
                else:
                    time.sleep(0.1)

                continue

            read_failures = 0

            # ---- Detection on a small frame, recognition on full-res crops ----
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_h, frame_w = gray_full.shape[:2]

            gray_small = cv2.resize(
                gray_full, None, fx=DETECT_SCALE, fy=DETECT_SCALE
            )

            detected_faces = detect_faces(gray_small, (40, 40))

            now = time.time()
            today = date.today()
            unknown_in_frame = False

            for (x, y, w, h) in detected_faces:

                # Map the box back to full-resolution coordinates.
                x0 = max(int(x / DETECT_SCALE), 0)
                y0 = max(int(y / DETECT_SCALE), 0)
                x1 = min(int((x + w) / DETECT_SCALE), frame_w)
                y1 = min(int((y + h) / DETECT_SCALE), frame_h)

                face = gray_full[y0:y1, x0:x1]

                if face.size == 0:
                    continue

                face = preprocess_face(face)
                label, confidence = recognizer.predict(face)

                if (
                    confidence < FACE_CONFIDENCE_THRESHOLD
                    and label in label_to_name
                ):
                    name = label_to_name[label]

                    # Attendance: once per person per day, retried on error.
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

                    # Throttled logging.
                    if now - last_logged.get(name, 0) >= RECOGNIZED_LOG_INTERVAL:
                        print(
                            f"[RECOGNIZED] {name} "
                            f"(Confidence: {confidence:.2f})"
                        )
                        last_logged[name] = now

                else:
                    unknown_in_frame = True

            # ---- Unknown-face alerting with temporal smoothing ----
            if unknown_in_frame:
                unknown_streak += 1
            else:
                unknown_streak = 0

            if (
                unknown_streak >= UNKNOWN_FRAMES_REQUIRED
                and now - last_alert_time >= COOLDOWN_SECONDS
                and (alert_thread is None or not alert_thread.is_alive())
            ):
                print("[ALERT] Unauthorized person detected!")

                # Start the cooldown on every attempt (success or failure) so a
                # broken network or bad password can't cause a retry storm.
                last_alert_time = now
                unknown_streak = 0

                if email_configured():
                    alert_thread = start_alert_thread(frame)

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