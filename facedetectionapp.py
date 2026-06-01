"""
Application Streamlit modernisée pour la détection et la description d'images.

Cette version étend la fonctionnalité de l'application originale de détection de
visages : elle capture une photo dès qu'un visage est détecté via une webcam,
enregistre l'image et affiche une description succincte de la scène.  La
description est générée localement à partir du nombre de visages détectés et
d'une analyse heuristique des couleurs et de la luminosité de l'image afin
d'offrir une indication sur l'environnement (verdoyant, ciel/eau, sombre…)
sans dépendre de modèles de deep learning externes.  L'interface Streamlit
a également été modernisée grâce à une palette de couleurs, des icônes
émoticônes et un agencement soigné à l'aide de colonnes.  Les paramètres de
détection sont accessibles via la barre latérale.
"""

import os
from typing import Dict

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Répertoire où seront stockées les captures
CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Charger le classificateur Haar Cascade pour la détection de visages
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# -----------------------------------------------------------------------------
#  Fonctions utilitaires pour l'analyse et la description des images
# -----------------------------------------------------------------------------
def analyse_couleurs(frame: np.ndarray) -> Dict[str, float]:
    """Analyse basique de la couleur et de la luminosité d'une image.

    L'image est convertie en espace HSV pour calculer les moyennes de teinte
    (H), saturation (S) et valeur (V).  Une version en niveaux de gris est
    également générée pour estimer la luminosité moyenne.  Ces statistiques
    simples servent ensuite à décrire grossièrement l'environnement (ex. : une
    dominante de vert suggère un environnement naturel).

    Args:
        frame: Image BGR lue avec OpenCV.

    Returns:
        Un dictionnaire contenant les moyennes des canaux H, S, V et de la
        luminosité en niveaux de gris.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    avg_h = float(np.mean(h))
    avg_s = float(np.mean(s))
    avg_v = float(np.mean(v))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_gray = float(np.mean(gray))
    return {"avg_h": avg_h, "avg_s": avg_s, "avg_v": avg_v, "avg_gray": avg_gray}


def generer_description(frame: np.ndarray, face_count: int) -> str:
    """Génère une description textuelle simple de l'image.

    La description combine le nombre de visages détectés avec une analyse
    élémentaire de l'éclairage et des couleurs dominantes afin de donner une
    indication générale du contenu de la scène.

    Args:
        frame: Image BGR provenant de la capture vidéo.
        face_count: Nombre de visages détectés dans l'image.

    Returns:
        Une chaîne de caractères décrivant brièvement la scène.
    """
    stats = analyse_couleurs(frame)
    descriptions = []
    # Décrire le nombre de visages
    if face_count == 0:
        descriptions.append("Aucun visage détecté.")
    elif face_count == 1:
        descriptions.append("Un visage détecté.")
    else:
        descriptions.append(f"{face_count} visages détectés.")
    # Analyse de la luminosité (moyenne des niveaux de gris)
    if stats["avg_gray"] < 60:
        descriptions.append("L'image est sombre.")
    elif stats["avg_gray"] > 200:
        descriptions.append("L'image est très lumineuse.")
    # Analyse des teintes dominantes (seulement si la saturation est suffisante)
    h = stats["avg_h"]
    s = stats["avg_s"]
    if s > 50:
        if 50 < h < 90:
            descriptions.append(
                "Présence importante de vert : l'environnement semble naturel ou verdoyant."
            )
        elif 90 <= h < 150:
            descriptions.append(
                "Dominante de bleu : cela évoque un ciel dégagé ou un plan d'eau."
            )
        elif h < 20 or h > 170:
            descriptions.append("Dominante de rouge/orange dans l'image.")
    else:
        descriptions.append(
            "Les couleurs semblent neutres ou peu saturées, indiquant un décor simple."
        )
    return " ".join(descriptions)


# -----------------------------------------------------------------------------
#  Fonctions de flux vidéo et de gestion des captures
# -----------------------------------------------------------------------------
def video_streaming(scaleFactor: float, minNeighbors: int, rectangle_color: str) -> None:
    """Capture le flux vidéo, détecte les visages et enregistre une photo.

    Dès qu'un visage est détecté, l'image est enregistrée, un rectangle est
    dessiné autour des visages, une description est générée et affichée.  Le
    flux vidéo continue jusqu'à ce qu'une capture soit réalisée ou que
    l'utilisateur clique sur « Arrêter ».

    Args:
        scaleFactor: Paramètre scaleFactor pour la détection des visages.
        minNeighbors: Paramètre minNeighbors pour la détection des visages.
        rectangle_color: Couleur hexadécimale (#RRGGBB) pour les rectangles.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la webcam.")
        return

    stframe = st.empty()
    stop_button = st.button("⏹️ Arrêter le flux", key="stop")
    captured = False

    # Convertir la couleur hexadécimale en tuple BGR pour OpenCV
    color = tuple(int(rectangle_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    while not captured:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture vidéo.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors
        )
        # Dessiner les rectangles autour des visages
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Afficher le flux dans l'interface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        # Si un visage est détecté et aucune capture n'a été faite
        if len(faces) > 0 and not captured:
            # Sauvegarder l'image
            file_path = save_capture(frame)
            # Générer la description
            description = generer_description(frame, len(faces))
            st.success("Photo capturée !")
            st.markdown(f"**Description :** {description}")
            captured = True
        # Permettre à l'utilisateur de stopper manuellement
        if stop_button:
            st.info("Flux vidéo arrêté par l'utilisateur.")
            break
    cap.release()


def save_capture(frame: np.ndarray) -> str:
    """Enregistre une capture dans le dossier dédié avec un nom unique.

    Args:
        frame: Image BGR à sauvegarder.

    Returns:
        Le chemin du fichier enregistré.
    """
    photo_number = (
        len([f for f in os.listdir(CAPTURE_FOLDER) if f.lower().endswith(".jpg")]) + 1
    )
    file_path = os.path.join(CAPTURE_FOLDER, f"photo_{photo_number}.jpg")
    cv2.imwrite(file_path, frame)
    return file_path


def view_captures() -> None:
    """Affiche les photos déjà capturées dans une galerie déroulante."""
    st.subheader("🖼️ Historique des captures")
    photos = [f for f in os.listdir(CAPTURE_FOLDER) if f.lower().endswith(".jpg")]
    if not photos:
        st.write("Aucune photo capturée pour le moment.")
    else:
        for photo in sorted(photos):
            file_path = os.path.join(CAPTURE_FOLDER, photo)
            img = Image.open(file_path)
            st.image(img, caption=photo, use_column_width=True)


def delete_captures() -> None:
    """Supprime toutes les photos présentes dans le dossier de captures."""
    photos = [f for f in os.listdir(CAPTURE_FOLDER) if f.lower().endswith(".jpg")]
    if not photos:
        st.warning("Aucune capture à supprimer.")
    else:
        for photo in photos:
            os.remove(os.path.join(CAPTURE_FOLDER, photo))
        st.success("Toutes les captures ont été supprimées.")


# -----------------------------------------------------------------------------
#  Interface Streamlit principale
# -----------------------------------------------------------------------------
def app() -> None:
    """Fonction principale pour lancer l'interface Streamlit."""
    # Configuration de la page (icône emoji et mise en page centrée)
    st.set_page_config(
        page_title="Vision IA – Détection et description", page_icon="📷", layout="centered"
    )
    st.title("🎯 Vision IA : Détection & Description d'Image")
    st.markdown(
        """
        Cette application capture automatiquement une photo dès qu'un visage est détecté,
        puis fournit une courte description basée sur le nombre de personnes et les
        couleurs dominantes de l'image. Utilisez les paramètres dans la barre
        latérale pour ajuster la détection.
        """
    )
    # Barre latérale pour les paramètres de détection
    with st.sidebar:
        st.header("🔧 Paramètres de détection")
        rectangle_color = st.color_picker("Couleur des rectangles", "#00FF00")
        minNeighbors = st.slider("minNeighbors", 1, 10, 5, 1)
        scaleFactor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.1)
        st.markdown("---")
        st.header("📂 Captures")
        # Actions secondaires dans la barre latérale
        if st.button("🖼️ Voir les captures"):
            view_captures()
        if st.button("🗑️ Supprimer les captures"):
            delete_captures()

    # Zone centrale pour lancer le flux vidéo
    st.markdown("---")
    st.subheader("🎥 Lancer l'analyse en temps réel")
    if st.button("📸 Démarrer la caméra"):
        video_streaming(scaleFactor, minNeighbors, rectangle_color)


if __name__ == "__main__":
    app()