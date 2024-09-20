import os #to make changes with directories
import cv2  #to access camera

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  #make a directory if not available in the system
    os.makedirs(DATA_DIR)

# Define the numbers and alphabets to capture (0-9, A-Z)
characters = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

dataset_size_per_character = 100

cap = cv2.VideoCapture(0)

for character in characters:
    character_dir = os.path.join(DATA_DIR, character)
    if not os.path.exists(character_dir):
        os.makedirs(character_dir)

    print('Collecting data for character: {}'.format(character))

    # Display a message prompting the user to get ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready to capture "{}". Press "Q" to continue.'.format(character), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images for the current character
    for i in range(dataset_size_per_character):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(character_dir, '{}_{}.jpg'.format(character, i)), frame)

cap.release()   #Releases the webcam resource.
cv2.destroyAllWindows()  #Closes all OpenCV windows.
