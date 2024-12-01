import cv2
import streamlit as st
import numpy as np

# Charger le fichier Haar Cascade
HAAR_CASCADE_PATH = "./haarcascade_frontalface_default.xml"
try:
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Le fichier Haar Cascade n'a pas été chargé correctement.")
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier Haar Cascade : {e}")

def video_streaming(scaleFactor, minNeighbors, rectangle_color, video_source="./test_video.mp4"):
    """
    Fonction pour capturer un flux vidéo, détecter des visages,
    et capturer automatiquement 3 images.
    """
    cap = cv2.VideoCapture(video_source)  # Source vidéo
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la source vidéo.")
        return

    stframe = st.empty()  # Placeholder pour le flux vidéo
    captured_images = []  # Liste pour stocker les images capturées
    max_photos = 3  # Nombre maximum de photos à capturer

    while len(captured_images) < max_photos:
        ret, frame = cap.read()
        if not ret:
            st.warning("Fin de la vidéo ou erreur de lecture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if len(captured_images) < max_photos:
                captured_images.append(frame.copy())
                st.write(f"Photo capturée ({len(captured_images)}/{max_photos})")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

    if captured_images:
        st.write("Images capturées :")
        for idx, img in enumerate(captured_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Photo {idx + 1}", channels="RGB")

def app():
    st.title("Détection de visages avec capture automatique")
    st.markdown("""
    ### Instructions :
    1. Cliquez sur **"Lancer le flux vidéo"** pour démarrer la détection des visages.
    2. Les visages détectés seront entourés de rectangles, et **3 photos** seront automatiquement capturées.
    3. Une fois 3 photos capturées, le flux vidéo s'arrête automatiquement.
    """)

    rectangle_color = st.color_picker("Couleur des rectangles", "#00FF00")
    minNeighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)
    scaleFactor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)
    video_source = st.text_input("Chemin de la vidéo (par défaut : vidéo de test)", "./test_video.mp4")

    if st.button("Lancer le flux vidéo"):
        video_streaming(scaleFactor, minNeighbors, rectangle_color, video_source=video_source)

if __name__ == "__main__":
    app()
