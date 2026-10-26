
import cv2
import face_recognition
import numpy as np
import os
import csv
import smtplib
import time
import io

from datetime import datetime, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ==========================================
# CONFIGURATION
# ==========================================

SENDER_EMAIL = "hitmanfanvela@gmail.com"
SENDER_PASSWORD = "qddf nhbx ttfb rmxn"
RECEIVER_EMAIL = "singaravelaramesh@gmail.com"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

COOLDOWN_SECONDS = 30
FACE_TOLERANCE = 0.5
CAMERA_INDEX = 0


# ==========================================
# LOAD KNOWN FACES
# ==========================================

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    if not os.path.isdir(known_faces_dir):
        print(f"[ERROR] Folder '{known_faces_dir}' not found.")
        return known_encodings, known_names

    for filename in os.listdir(known_faces_dir):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(known_faces_dir, filename)

        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"[WARNING] No face found in {filename}")
                continue

            if len(encodings) > 1:
                print(
                    f"[WARNING] Multiple faces in {filename}; "
                    "using the first face."
                )

            name = os.path.splitext(filename)[0].replace("_", " ").title()

            known_encodings.append(encodings[0])
            known_names.append(name)

            print(f"[INFO] Loaded authorized person: {name}")

        except Exception as e:
            print(f"[ERROR] Could not load {filename}: {e}")

    return known_encodings, known_names


# ==========================================
# ATTENDANCE MANAGEMENT
# ==========================================

def mark_attendance(name):
    today = date.today().isoformat()

    # Check whether this person is already marked today.
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(
                ATTENDANCE_FILE, "r", newline="", encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)

                for row in reader:
                    if (
                        row.get("Name") == name
                        and row.get("Date") == today
                    ):
                        return False

        except (OSError, csv.Error) as e:
            print(f"[ERROR] Could not read attendance file: {e}")
            return False

    current_time = datetime.now().strftime("%H:%M:%S")
    file_exists = (
        os.path.exists(ATTENDANCE_FILE)
        and os.path.getsize(ATTENDANCE_FILE) > 0
    )

    try:
        with open(
            ATTENDANCE_FILE, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Name", "Date", "Time", "Status"])

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
        return True

    except OSError as e:
        print(f"[ERROR] Could not write attendance: {e}")
        return False


# ==========================================
# EMAIL ALERT
# ==========================================

def send_email_alert(image_frame):
    """
    Encodes the captured frame in memory and emails it.
    No unauthorized-person image is saved to disk.
    """

    if not SENDER_PASSWORD:
        print(
            "[ERROR] GMAIL_APP_PASSWORD environment variable "
            "is not configured."
        )
        return False

    try:
        # Encode image as JPEG in memory.
        # Nothing is written to the filesystem.
        success, encoded_image = cv2.imencode(
            ".jpg", image_frame
        )

        if not success:
            print("[ERROR] Could not encode captured image.")
            return False

        image_bytes = encoded_image.tobytes()

        # Build email
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = (
            "ALERT: Unauthorized Person Detected!"
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        body = (
            "An unknown face was detected by the smart "
            "attendance system.\n\n"
            f"Detection time: {timestamp}\n"
            "The captured image is attached to this email.\n"
            "The image was not saved to disk by the program."
        )

        msg.attach(MIMEText(body, "plain"))

        image_attachment = MIMEImage(
            image_bytes,
            _subtype="jpeg",
            name="unauthorized_detected.jpg"
        )

        msg.attach(image_attachment)

        # Send email securely using SMTP with TLS.
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(SENDER_EMAIL, SENDER_PASSWORD)

            server.sendmail(
                SENDER_EMAIL,
                [RECEIVER_EMAIL],
                msg.as_string()
            )

        print("[SUCCESS] Unauthorized alert email sent.")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


# ==========================================
# FACE RECOGNITION
# ==========================================

def main():
    known_encodings, known_names = load_known_faces(
        KNOWN_FACES_DIR
    )

    if not known_encodings:
        print(
            "[ERROR] No usable authorized faces found. "
            "Add clear face images to known_faces/."
        )
        return

    # Validate email configuration before starting.
    if not SENDER_PASSWORD:
        print(
            "[WARNING] Email password not configured. "
            "Unknown-face email alerts will not work."
        )

    video_capture = cv2.VideoCapture(CAMERA_INDEX)

    if not video_capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    last_alert_time = 0.0

    # Avoid repeatedly marking the same face in each frame.
    attendance_marked_this_session = set()

    print("[INFO] Smart Attendance System started.")
    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("[ERROR] Could not read camera frame.")
                break

            # Reduce image size to improve processing speed.
            small_frame = cv2.resize(
                frame, (0, 0), fx=0.25, fy=0.25
            )

            rgb_small_frame = cv2.cvtColor(
                small_frame, cv2.COLOR_BGR2RGB
            )

            face_locations = face_recognition.face_locations(
                rgb_small_frame
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            current_time = time.time()

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                name = "UNAUTHORIZED"

                face_distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )

                if len(face_distances) > 0:
                    best_match_index = int(
                        np.argmin(face_distances)
                    )

                    if (
                        face_distances[best_match_index]
                        < FACE_TOLERANCE
                    ):
                        name = known_names[best_match_index]

                # Convert coordinates to original frame size.
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == "UNAUTHORIZED":
                    color = (0, 0, 255)
                    status = "Unknown"

                    # Global cooldown for unknown-face alerts.
                    if (
                        current_time - last_alert_time
                        >= COOLDOWN_SECONDS
                    ):
                        print(
                            "[ALERT] Unauthorized person detected!"
                        )

                        # Pass the frame directly; do not save it.
                        if send_email_alert(frame):
                            last_alert_time = current_time

                else:
                    color = (0, 255, 0)

                    if name not in attendance_marked_this_session:
                        marked = mark_attendance(name)

                        # Avoid repeated file checks during this run.
                        # Existing attendance is also protected by CSV.
                        attendance_marked_this_session.add(name)

                        if marked:
                            status = "Present - Marked"
                        else:
                            status = "Present - Already Marked"
                    else:
                        status = "Present"

                # Draw face box.
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    2
                )

                # Draw label background.
                label_top = max(0, bottom - 55)

                cv2.rectangle(
                    frame,
                    (left, label_top),
                    (right, bottom),
                    color,
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    name,
                    (left + 6, max(20, bottom - 30)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    frame,
                    status,
                    (left + 6, max(15, bottom - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

            cv2.imshow(
                "Smart Attendance Tracking System",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Smart Attendance System stopped.")


if __name__ == "__main__":
    main()

