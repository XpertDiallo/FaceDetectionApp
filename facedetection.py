import cv2
import streamlit as st
import numpy as np
import os
from PIL import Image

# Dossier pour stocker les captures
CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Charger le classificateur Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def video_streaming(scaleFactor, minNeighbors, rectangle_color):
    """
    Fonction pour capturer une seule photo lorsque visage détecté,
    et effacer les anciennes images affichées tout en conservant les sauvegardes.
    """
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la webcam.")
        return

    stframe = st.empty()  # Placeholder Streamlit pour afficher le flux vidéo
    stop_button = st.button("Arrêter le flux")
    captured = False  # Indique si une capture a été réalisée

    while not captured:
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

            if not captured:  # Si aucune capture n'a encore été réalisée
                save_capture(frame)
                st.write("Photo capturée.")
                captured = True

        # Convertir le frame pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Afficher le flux vidéo dans Streamlit
        stframe.image(frame_rgb, channels="RGB")

        # Sortir si l'utilisateur clique sur "Arrêter le flux"
        if stop_button:
            st.write("Arrêt du flux vidéo.")
            break

    # Libérer les ressources
    cap.release()

def save_capture(frame):
    """
    Sauvegarde une seule capture dans le dossier des captures avec un nom unique.
    """
    photo_number = len([f for f in os.listdir(CAPTURE_FOLDER) if f.endswith(".jpg")]) + 1
    file_path = os.path.join(CAPTURE_FOLDER, f"photo_{photo_number}.jpg")
    cv2.imwrite(file_path, frame)

def view_captures():
    """
    Affiche toutes les photos sauvegardées dans le dossier des captures.
    """
    st.write("### Historique des captures")
    photos = [f for f in os.listdir(CAPTURE_FOLDER) if f.endswith(".jpg")]
    if not photos:
        st.write("Aucune photo capturée pour le moment.")
    else:
        for photo in photos:
            file_path = os.path.join(CAPTURE_FOLDER, photo)
            img = Image.open(file_path)
            st.image(img, caption=photo, use_column_width=True)

def delete_captures():
    """
    Supprime toutes les photos dans le dossier des captures.
    """
    photos = [f for f in os.listdir(CAPTURE_FOLDER) if f.endswith(".jpg")]
    if not photos:
        st.write("Aucune capture à supprimer.")
    else:
        for photo in photos:
            os.remove(os.path.join(CAPTURE_FOLDER, photo))
        st.write("Toutes les captures ont été supprimées.")

# Interface Streamlit
def app():
    st.title("Détection de visages avec gestion des captures")
    st.markdown("""
    ### Instructions :
    1. Cliquez sur **"Lancer le flux vidéo"** pour afficher la vidéo en temps réel.
    2. Lorsqu'un visage est détecté, une seule photo sera capturée et sauvegardée.
    3. Cliquez sur **"Visualiser"** pour consulter l'historique des photos capturées.
    4. Cliquez sur **"Supprimer les anciennes captures"** pour supprimer toutes les photos sauvegardées.
    """)

    # Paramètres de détection
    rectangle_color = st.color_picker("Couleur des rectangles", "#00FF00")
    minNeighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)
    scaleFactor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)

    # Boutons pour lancer la détection, visualiser ou supprimer les photos
    if st.button("Lancer le flux vidéo"):
        video_streaming(scaleFactor, minNeighbors, rectangle_color)

    if st.button("Visualiser"):
        view_captures()

    if st.button("Supprimer les anciennes captures"):
        delete_captures()

if __name__ == "__main__":
    app()
