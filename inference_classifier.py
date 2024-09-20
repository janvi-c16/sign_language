import pickle  #for loading the trained model
import cv2   #to access camera
import mediapipe as mp   #for hand landmarks
import numpy as np   #for numerical operations
from sklearn.preprocessing import StandardScaler  #for normalizing the feature data

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Fit the scaler to a dummy array of zeros with 42 features
scaler = StandardScaler()
scaler.fit([np.zeros(42)])

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Labels dictionary for numbers (0-9) and alphabets (A-Z)
labels_dict = {i: chr(48 + i) if i < 10 else chr(55 + i) for i in range(36)}  #using ascii values

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Adjust data_aux to contain only 42 features
        data_aux = data_aux[:42]  # Ensure the length is 42
        data_aux = scaler.transform([data_aux])  #Standardizes the feature vector to have zero mean and unit variance, which improves the performance and convergence of many machine learning algorithms.
        prediction = model.predict(data_aux)


        predicted_character = prediction[0]  # Directly use the predicted character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
