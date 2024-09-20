import os
import pickle #uses to change format of objects
import mediapipe as mp #using to make hand landmarks
import cv2   #image processing

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands    #access to the hands module
mp_drawing = mp.solutions.drawing_utils  #provide drawing utilities.
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'

data = []   #Initializes empty lists data and labels
labels = []

# Process each image in the data directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []   #store auxiliary data
        x_ = []         # coordinates of hand landmarks for the current image.
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  #read current image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert img format

        results = hands.process(img_rgb)   #marking the hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):  #normalize kr dega agar screen mai kahi bhi aa rha hoga haath vo detect krlega
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save the processed data into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)    #Serializes the data and labels lists into the pickle file

# Initialize the webcam
#cap = cv2.VideoCapture(0)

# Capture a frame to determine its dimensions
#ret, frame = cap.read()

# Get the dimensions of the captured frame
#if ret:
   # image_height, image_width, _ = frame.shape
   # print(f"Frame dimensions: {image_width}x{image_height}")

# Release the webcam
#cap.release()

print("Data processing complete. Data saved to 'data.pickle'.")
