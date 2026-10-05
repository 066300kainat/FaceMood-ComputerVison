
import cv2
import time
import numpy as np
import mediapipe as mp
import onnxruntime as ort

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FACEMOOD
# Real-Time Facial Expression Recognition
# MediaPipe + ONNX FER+
# ============================================================


# ============================================================
# CONFIG
# ============================================================

LANDMARK_MODEL = "models/face_landmarker.task"
EMOTION_MODEL = "models/emotion-ferplus-8.onnx"

EMOTIONS = [
    "Neutral",
    "Happy",
    "Surprise",
    "Sad",
    "Angry",
    "Disgust",
    "Fear",
    "Contempt"
]


# ============================================================
# LOAD MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=LANDMARK_MODEL
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# LOAD ONNX EMOTION MODEL
# ============================================================

emotion_session = ort.InferenceSession(
    EMOTION_MODEL,
    providers=["CPUExecutionProvider"]
)

emotion_input = emotion_session.get_inputs()[0].name


# ============================================================
# STATE
# ============================================================

emotion_history = deque(maxlen=2)

current_emotion = "Neutral"
current_confidence = 0.0

fps = 0.0
previous_time = time.perf_counter()

frame_timestamp = 0

last_console_time = 0.0


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# ============================================================
# PREPROCESS FACE FOR FER+
# ============================================================

def preprocess_face(face):

    if face is None or face.size == 0:
        return None

    # --------------------------------------------------------
    # Convert BGR -> grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Resize to FER+ input size
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Convert to float32
    #
    # IMPORTANT:
    # No histogram equalization.
    # No [-1, 1] normalization.
    # --------------------------------------------------------

    gray = gray.astype(np.float32)

    # --------------------------------------------------------
    # Model input:
    #
    # [64,64]
    #      ->
    # [1,1,64,64]
    # --------------------------------------------------------

    tensor = gray.reshape(
        1,
        1,
        64,
        64
    )

    return tensor


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(face):

    tensor = preprocess_face(face)

    if tensor is None:
        return "Neutral", 0.0, None

    # --------------------------------------------------------
    # ONNX inference
    # --------------------------------------------------------

    output = emotion_session.run(
        None,
        {
            emotion_input: tensor
        }
    )

    logits = output[0][0]

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = softmax(logits)

    # --------------------------------------------------------
    # Highest probability
    # --------------------------------------------------------

    index = int(
        np.argmax(probabilities)
    )

    emotion = EMOTIONS[index]

    confidence = float(
        probabilities[index]
    )

    return (
        emotion,
        confidence,
        probabilities
    )


# ============================================================
# DRAW DYNAMIC EMOJI / AVATAR
# ============================================================

def draw_emoji_face(
    frame,
    center,
    radius,
    emotion
):

    cx, cy = center

    # ========================================================
    # FACE
    # ========================================================

    cv2.circle(
        frame,
        center,
        radius,
        (60, 210, 245),
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        center,
        radius,
        (35, 35, 35),
        5,
        cv2.LINE_AA
    )

    # ========================================================
    # COMMON EYE POSITIONS
    # ========================================================

    left_eye = (
        cx - int(radius * 0.35),
        cy - int(radius * 0.18)
    )

    right_eye = (
        cx + int(radius * 0.35),
        cy - int(radius * 0.18)
    )

    eye_w = int(radius * 0.15)
    eye_h = int(radius * 0.22)

    # ========================================================
    # HAPPY
    # ========================================================

    if emotion == "Happy":

        # Happy closed eyes

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    max(3, eye_h // 3)
                ),
                0,
                0,
                180,
                (35, 35, 35),
                7,
                cv2.LINE_AA
            )

        # Big smile

        cv2.ellipse(
            frame,
            (
                cx,
                cy + int(radius * 0.18)
            ),
            (
                int(radius * 0.43),
                int(radius * 0.30)
            ),
            0,
            0,
            180,
            (35, 35, 35),
            -1,
            cv2.LINE_AA
        )

        # Teeth

        cv2.ellipse(
            frame,
            (
                cx,
                cy + int(radius * 0.12)
            ),
            (
                int(radius * 0.30),
                int(radius * 0.13)
            ),
            0,
            0,
            180,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    # ========================================================
    # SAD
    # ========================================================

    elif emotion == "Sad":

        # Eyes

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    eye_h
                ),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(eye_w * 0.35),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        # Sad eyebrows

        cv2.line(
            frame,
            (
                cx - int(radius * 0.52),
                cy - int(radius * 0.43)
            ),
            (
                cx - int(radius * 0.18),
                cy - int(radius * 0.52)
            ),
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (
                cx + int(radius * 0.18),
                cy - int(radius * 0.52)
            ),
            (
                cx + int(radius * 0.52),
                cy - int(radius * 0.43)
            ),
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

        # Frown

        cv2.ellipse(
            frame,
            (
                cx,
                cy + int(radius * 0.38)
            ),
            (
                int(radius * 0.38),
                int(radius * 0.25)
            ),
            0,
            180,
            360,
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

        # Tears

        for x, y in [left_eye, right_eye]:

            pts = np.array([
                [
                    x,
                    y + int(radius * 0.18)
                ],
                [
                    x - int(radius * 0.07),
                    y + int(radius * 0.35)
                ],
                [
                    x,
                    y + int(radius * 0.48)
                ],
                [
                    x + int(radius * 0.07),
                    y + int(radius * 0.35)
                ]
            ])

            cv2.fillPoly(
                frame,
                [pts],
                (200, 225, 255)
            )

    # ========================================================
    # ANGRY
    # ========================================================

    elif emotion == "Angry":

        # Eyes

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    eye_h
                ),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(eye_w * 0.38),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        # Angry eyebrows

        cv2.line(
            frame,
            (
                cx - int(radius * 0.52),
                cy - int(radius * 0.48)
            ),
            (
                cx - int(radius * 0.12),
                cy - int(radius * 0.30)
            ),
            (35, 35, 35),
            9,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (
                cx + int(radius * 0.12),
                cy - int(radius * 0.30)
            ),
            (
                cx + int(radius * 0.52),
                cy - int(radius * 0.48)
            ),
            (35, 35, 35),
            9,
            cv2.LINE_AA
        )

        # Angry mouth

        cv2.line(
            frame,
            (
                cx - int(radius * 0.35),
                cy + int(radius * 0.30)
            ),
            (
                cx,
                cy + int(radius * 0.20)
            ),
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (
                cx,
                cy + int(radius * 0.20)
            ),
            (
                cx + int(radius * 0.35),
                cy + int(radius * 0.30)
            ),
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

    # ========================================================
    # SURPRISE
    # ========================================================

    elif emotion == "Surprise":

        for x, y in [left_eye, right_eye]:

            cv2.circle(
                frame,
                (x, y),
                int(radius * 0.18),
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(radius * 0.07),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        # Open mouth

        cv2.ellipse(
            frame,
            (
                cx,
                cy + int(radius * 0.27)
            ),
            (
                int(radius * 0.20),
                int(radius * 0.29)
            ),
            0,
            0,
            360,
            (35, 35, 35),
            -1,
            cv2.LINE_AA
        )

    # ========================================================
    # FEAR
    # ========================================================

    elif emotion == "Fear":

        for x, y in [left_eye, right_eye]:

            cv2.circle(
                frame,
                (x, y),
                int(radius * 0.18),
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(radius * 0.07),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        cv2.ellipse(
            frame,
            (
                cx,
                cy + int(radius * 0.28)
            ),
            (
                int(radius * 0.28),
                int(radius * 0.22)
            ),
            0,
            0,
            360,
            (35, 35, 35),
            -1,
            cv2.LINE_AA
        )

    # ========================================================
    # DISGUST
    # ========================================================

    elif emotion == "Disgust":

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    eye_h
                ),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(eye_w * 0.35),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        pts = np.array([
            [
                cx - int(radius * 0.35),
                cy + int(radius * 0.20)
            ],
            [
                cx - int(radius * 0.12),
                cy + int(radius * 0.28)
            ],
            [
                cx + int(radius * 0.12),
                cy + int(radius * 0.18)
            ],
            [
                cx + int(radius * 0.35),
                cy + int(radius * 0.20)
            ]
        ])

        cv2.polylines(
            frame,
            [pts],
            False,
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

    # ========================================================
    # CONTEMPT
    # ========================================================

    elif emotion == "Contempt":

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    eye_h
                ),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(eye_w * 0.35),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        # Smirk

        cv2.ellipse(
            frame,
            (
                cx + int(radius * 0.08),
                cy + int(radius * 0.25)
            ),
            (
                int(radius * 0.35),
                int(radius * 0.15)
            ),
            -8,
            0,
            180,
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )

    # ========================================================
    # NEUTRAL
    # ========================================================

    else:

        for x, y in [left_eye, right_eye]:

            cv2.ellipse(
                frame,
                (x, y),
                (
                    eye_w,
                    eye_h
                ),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                int(eye_w * 0.35),
                (35, 35, 35),
                -1,
                cv2.LINE_AA
            )

        # Straight mouth

        cv2.line(
            frame,
            (
                cx - int(radius * 0.30),
                cy + int(radius * 0.25)
            ),
            (
                cx + int(radius * 0.30),
                cy + int(radius * 0.25)
            ),
            (35, 35, 35),
            7,
            cv2.LINE_AA
        )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    landmarker.close()

    raise SystemExit


# Camera resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 65)
print("FaceMood - Real-Time Facial Expression Recognition")
print("=" * 65)
print("Engine: MediaPipe Face Landmarker + ONNX FER+")
print("Q = Quit")
print("R = Reset emotion")
print("=" * 65)
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            print("Camera frame error.")

            break


        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Keep clean copy
        #
        # This is important:
        # face crop is taken from original frame before
        # landmarks / labels / panel are drawn.
        # ----------------------------------------------------

        original_frame = frame.copy()

        height, width = frame.shape[:2]


        # ====================================================
        # FPS
        # ====================================================

        now = time.perf_counter()

        dt = now - previous_time

        if dt > 0:

            instant_fps = 1.0 / dt

            fps = (
                fps * 0.90
                +
                instant_fps * 0.10
            )

        previous_time = now


        # ====================================================
        # MEDIAPIPE INPUT
        # ====================================================

        frame_timestamp += 33

        rgb = cv2.cvtColor(
            original_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # MEDIAPIPE LANDMARK INFERENCE
        # ====================================================

        landmark_start = time.perf_counter()

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        landmark_time = (
            time.perf_counter()
            -
            landmark_start
        ) * 1000


        emotion_time = 0.0


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # =================================================
            # LANDMARK COORDINATES
            # =================================================

            xs = np.array([
                lm.x * width
                for lm in landmarks
            ])

            ys = np.array([
                lm.y * height
                for lm in landmarks
            ])


            # =================================================
            # FACE BOUNDING BOX
            # =================================================

            x_min = int(np.min(xs))
            x_max = int(np.max(xs))

            y_min = int(np.min(ys))
            y_max = int(np.max(ys))


            face_w = x_max - x_min
            face_h = y_max - y_min


            # =================================================
            # SQUARE FACE CROP
            # =================================================

            box_size = max(
                face_w,
                face_h
            )

            # Padding around face

            box_size = int(
                box_size * 1.35
            )


            # Face center

            center_x = int(
                (x_min + x_max) / 2
            )

            center_y = int(
                (y_min + y_max) / 2
            )


            # Square coordinates

            x1 = (
                center_x
                -
                box_size // 2
            )

            y1 = (
                center_y
                -
                box_size // 2
            )

            x2 = (
                center_x
                +
                box_size // 2
            )

            y2 = (
                center_y
                +
                box_size // 2
            )


            # =================================================
            # CLAMP TO IMAGE
            # =================================================

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )


            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (80, 220, 120),
                3,
                cv2.LINE_AA
            )


            # =================================================
            # FACE CROP
            # =================================================

            face_crop = original_frame[
                y1:y2,
                x1:x2
            ]


            # =================================================
            # EMOTION INFERENCE
            # =================================================

            emotion_start = time.perf_counter()

            (
                detected_emotion,
                confidence,
                probabilities
            ) = predict_emotion(
                face_crop
            )

            emotion_time = (
                time.perf_counter()
                -
                emotion_start
            ) * 1000


            # =================================================
            # RAW TOP 3 PREDICTIONS
            # =================================================

            if probabilities is not None:

                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]


                current_console_time = time.perf_counter()


                # Print approximately once per second

                if (
                    current_console_time
                    -
                    last_console_time
                    >= 1.0
                ):

                    top_text = " | ".join(
                        [
                            (
                                f"{EMOTIONS[i]}="
                                f"{probabilities[i] * 100:.1f}%"
                            )
                            for i in top_indices
                        ]
                    )

                    print(
                        "Predictions:",
                        top_text
                    )

                    last_console_time = (
                        current_console_time
                    )


            # =================================================
            # EMOTION SMOOTHING
            # =================================================

            emotion_history.append(
                detected_emotion
            )


            values, counts = np.unique(
                emotion_history,
                return_counts=True
            )


            stable_index = int(
                np.argmax(counts)
            )


            stable_emotion = str(
                values[stable_index]
            )


            current_emotion = (
                stable_emotion
            )


            # -------------------------------------------------
            # Confidence smoothing
            # -------------------------------------------------

            current_confidence = (
                current_confidence * 0.50
                +
                confidence * 0.50
            )


            # =================================================
            # DRAW 468 LANDMARKS
            # =================================================

            for lm in landmarks:

                px = int(
                    lm.x * width
                )

                py = int(
                    lm.y * height
                )

                # Only draw visible points

                if (
                    0 <= px < width
                    and
                    0 <= py < height
                ):

                    cv2.circle(
                        frame,
                        (
                            px,
                            py
                        ),
                        1,
                        (100, 220, 255),
                        -1
                    )


            # =================================================
            # FACE EMOTION LABEL
            # =================================================

            cv2.putText(
                frame,
                current_emotion,
                (
                    x1,
                    max(
                        35,
                        y1 - 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 220, 120),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                f"{current_confidence * 100:.1f}%",
                (
                    x1,
                    max(
                        65,
                        y1 + 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # NO FACE
        # ====================================================

        else:

            current_emotion = "Neutral"

            current_confidence = 0.0

            emotion_history.clear()


        # ====================================================
        # RIGHT SIDE PANEL
        # ====================================================

        panel_x = int(
            width * 0.70
        )


        overlay = frame.copy()


        cv2.rectangle(
            overlay,
            (
                panel_x,
                0
            ),
            (
                width,
                height
            ),
            (20, 25, 35),
            -1
        )


        frame = cv2.addWeighted(
            overlay,
            0.82,
            frame,
            0.18,
            0
        )


        # ====================================================
        # PANEL TITLE
        # ====================================================

        cv2.putText(
            frame,
            "FACEMOOD",
            (
                panel_x + 35,
                50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "LIVE EXPRESSION",
            (
                panel_x + 35,
                80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (160, 200, 230),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # DYNAMIC EMOJI
        # ====================================================

        emoji_x = int(
            panel_x
            +
            (width - panel_x) / 2
        )


        emoji_y = int(
            height * 0.37
        )


        emoji_radius = int(
            min(
                width,
                height
            ) * 0.15
        )


        draw_emoji_face(
            frame,
            (
                emoji_x,
                emoji_y
            ),
            emoji_radius,
            current_emotion
        )


        # ====================================================
        # EMOTION TEXT
        # ====================================================

        emotion_text = (
            current_emotion.upper()
        )


        text_size = cv2.getTextSize(
            emotion_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            2
        )[0]


        text_x = int(
            emoji_x
            -
            text_size[0] / 2
        )


        cv2.putText(
            frame,
            emotion_text,
            (
                text_x,
                int(height * 0.66)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Confidence: "
                f"{current_confidence * 100:.1f}%"
            ),
            (
                panel_x + 35,
                int(height * 0.72)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # CONFIDENCE BAR
        # ====================================================

        bar_x = (
            panel_x + 35
        )

        bar_y = int(
            height * 0.76
        )


        bar_width = (
            width
            -
            panel_x
            -
            70
        )


        bar_height = 14


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (80, 80, 80),
            2
        )


        fill = int(
            bar_width
            *
            min(
                1.0,
                max(
                    0.0,
                    current_confidence
                )
            )
        )


        if fill > 0:

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y
                ),
                (
                    bar_x + fill,
                    bar_y + bar_height
                ),
                (80, 220, 120),
                -1
            )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                panel_x + 35,
                int(height * 0.84)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 220, 240),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            (
                f"Landmarks: "
                f"{landmark_time:.1f} ms"
            ),
            (
                panel_x + 35,
                int(height * 0.88)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            (
                f"Emotion: "
                f"{emotion_time:.1f} ms"
            ),
            (
                panel_x + 35,
                int(height * 0.92)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            frame,
            "ONNX + MEDIAPIPE ONLINE",
            (
                25,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 220, 120),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Q: Quit    R: Reset",
            (
                25,
                height - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "FaceMood - Real-Time Emotion AI",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        if key == ord("r"):

            emotion_history.clear()

            current_emotion = "Neutral"

            current_confidence = 0.0

            print("Emotion reset.")


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("FaceMood closed successfully.")

