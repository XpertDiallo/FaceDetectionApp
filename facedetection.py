import cv2
import streamlit as st
import numpy as np

# Charger le classificateur Haar Cascade
# Assurez-vous que le fichier XML est placé dans le même répertoire que ce script
HAAR_CASCADE_PATH = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def video_streaming(scaleFactor, minNeighbors, rectangle_color):
    """
    Fonction pour capturer un flux vidéo, détecter des visages et capturer automatiquement 3 images.
    """
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la webcam.")
        return

    # Initialisation des variables
    stframe = st.empty()  # Placeholder Streamlit pour afficher le flux vidéo
    captured_images = []  # Liste pour stocker les images capturées
    max_photos = 3  # Nombre maximum de photos à capturer

    while len(captured_images) < max_photos:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture vidéo.")
            break

        # Convertir en niveaux de gris pour la détection des visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        # Dessiner des rectangles autour des visages détectés
        color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Capturer automatiquement une photo si un visage est détecté
            if len(captured_images) < max_photos:
                captured_images.append(frame.copy())
                st.write(f"Photo capturée ({len(captured_images)}/{max_photos})")

        # Convertir le frame pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Afficher le flux vidéo dans Streamlit
        stframe.image(frame_rgb, channels="RGB")

    # Libérer les ressources
    cap.release()

    # Afficher les images capturées
    if captured_images:
        st.write("Images capturées :")
        for idx, img in enumerate(captured_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Photo {idx + 1}", channels="RGB")

# Interface principale de l'application Streamlit
def app():
    st.title("Détection de visages avec capture automatique")
    st.markdown("""
    ### Instructions :
    1. Cliquez sur **"Lancer le flux vidéo"** pour démarrer la détection des visages.
    2. Les visages détectés seront entourés de rectangles et **3 photos** seront automatiquement capturées.
    3. Une fois 3 photos capturées, le flux vidéo s'arrête automatiquement.
    """)

    # Paramètres de détection
    rectangle_color = st.color_picker("Couleur des rectangles", "#00FF00")
    minNeighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)
    scaleFactor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)

    # Bouton pour lancer la détection
    if st.button("Lancer le flux vidéo"):
        video_streaming(scaleFactor, minNeighbors, rectangle_color)

if __name__ == "__main__":
    app()
