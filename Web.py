# pylint: disable=E1101
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import ssl

app = Flask(__name__)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                        min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Función para capturar el video y procesar los frames
def generar_frames():
    vid = cv2.VideoCapture(0)
    vid.set(3, 1080)  # Ajustar el ancho de la cámara

    while True:
        success, frame = vid.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Voltear para efecto espejo
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos
        hand_results = hands.process(RGBframe)
        if hand_results.multi_hand_landmarks:
            for idx, (handLm, handType) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                hand_label = "Derecha" if handType.classification[0].label == "Right" else "Izquierda"

                for id, lm in enumerate(handLm.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 0:  # Muñeca
                        cv2.putText(frame, hand_label, (cx - 30, cy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                mp_draw.draw_landmarks(frame, handLm, mp_hands.HAND_CONNECTIONS)

        # Detectar rostros
        face_results = face_detection.process(RGBframe)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                     int(bboxC.width * w), int(bboxC.height * h)

                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.putText(frame, "Rostro", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    vid.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='server.crt', keyfile='server.key')
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=context)
