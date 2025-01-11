import cv2
import numpy as np
import seetaface


face_detector = seetaface.Detector('seeta_fd_frontal_v1.0.bin')  
face_recognizer = seetaface.FaceRecognizer('seeta_fr_v1.0.bin')  


image_path = './1.jpg'  


image = cv2.imread(image_path)


if image is None:
    print(f"Error: Could not open or find the image at {image_path}")
    exit()


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_detector.detect(gray)

for face in faces:

    x, y, w, h = face[:4]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face_img = image[y:y + h, x:x + w]
    feature = face_recognizer.extract(face_img)

    print("Face feature vector length:", len(feature))

cv2.imshow('Face Recognition', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
