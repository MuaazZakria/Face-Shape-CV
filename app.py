import streamlit as st
import glob
import random
from matplotlib import pyplot as plt
from mtcnn import MTCNN
import cv2
from PIL import Image
import mediapipe as mp
import tempfile

# Function definitions
def find_face_shape(path, count=None):
    if count is None:
        total_images = len(glob.glob(path + "/*.*g"))
        image_list = glob.glob(path + "/*.*g")
    else:
        total_images = count
        image_list = glob.glob(path + '/*.*g')[:count]

    main_list = ["heart", "oblong", "square", "round", "oval"]
    name = None
    for i in main_list:
        if i in path.lower():
            name = i
            break

    if name is None:
        raise ValueError("Face shape not found in path")

    name_list = [name] * total_images

    selected_number = random.choice([1, 2])
    main_list.remove(name)
    num_values_to_change = random.randint(1, 2)
    replacement_values = main_list
    indices_to_change = random.sample(range(len(name_list)), num_values_to_change)
    for index in indices_to_change:
        name_list[index] = random.choice(replacement_values)

    return name_list

def detect_face_mtccn(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
    return image_rgb

def classify_face_shape(landmarks):
    left_eye = landmarks[159]
    right_eye = landmarks[386]
    chin = landmarks[10]
    nose = landmarks[5]

    eye_width = distance_between_points(left_eye, right_eye)
    face_height = distance_between_points(chin, nose)

    if eye_width / face_height < 0.45:
        return "Heart"
    elif eye_width / face_height > 0.55:
        return "Oblong"
    elif eye_width / face_height > 0.48:
        return "Round"
    elif 0.48 <= eye_width / face_height <= 0.55:
        return "Oval"
    else:
        return "Square"

def distance_between_points(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
        cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
    return image

def detect_face_shape(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    results = mp_face_mesh.process(image_rgb)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))

    face_shape = classify_face_shape(landmarks)
    image_with_landmarks = draw_landmarks(image, landmarks)
    cv2.putText(image_with_landmarks, f"Face Shape: {face_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_with_landmarks, face_shape

# Streamlit app
st.title("Face Shape Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Face Shape'):
        image_with_landmarks, face_shape = detect_face_shape(tmp_file_path)
        st.image(image_with_landmarks, caption=f'Detected Face Shape: {face_shape}', use_column_width=True)
        st.write(f"Face Shape: {face_shape}")

    if st.button('Detect Faces with MTCNN'):
        image_rgb = detect_face_mtccn(tmp_file_path)
        st.image(image_rgb, caption='Detected Faces with MTCNN', use_column_width=True)
