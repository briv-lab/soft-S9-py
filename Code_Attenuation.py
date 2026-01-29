#Linear frequency transposition
import numpy as np
from scipy.io import wavfile
from scipy import signal

def transposition_frequentielle_lineaire(chemin_entree, chemin_sortie, freq_source, freq_cible, gain_transposition=0.5):
    """S
    Applique une transposition fréquentielle linéaire sur un fichier audio.
    
    Paramètres:
    - chemin_entree : Le fichier audio original (ex: 'voix_originale.wav')
    - chemin_sortie : Le fichier sauvegardé après traitement
    - freq_source   : La fréquence (Hz) où commence la zone inaudible (à déplacer)
    - freq_cible    : La fréquence (Hz) où l'on veut déplacer ces sons (zone audible)
    - gain_transposition : Volume des sons transposés (0.0 à 1.0) pour éviter la saturation
    """
    
    # 1. Chargement du fichier audio
    # fs = fréquence d'échantillonnage, data = les données audio
    fs, data = wavfile.read(chemin_entree)
    
    nyquist = fs / 2

    if freq_cible >= freq_source:
        freq_cible = freq_source * 0.8  # force la cible en dessous

    if freq_source >= nyquist:
        freq_source = nyquist - 100

    # Conversion en mono si le fichier est stéréo (pour simplifier le traitement)
    if len(data.shape) > 1:
        data = data[:, 0]
        
    # Normalisation du signal entre -1 et 1 pour faciliter les calculs
    data = data / np.max(np.abs(data))

    # 2. Passage dans le domaine fréquentiel (STFT - Short Time Fourier Transform)
    # Cela transforme le son en un spectrogramme (Fréquence vs Temps)
    # f = tableau des fréquences, t = temps, Zxx = matrice complexe du spectrogramme
    f, t, Zxx = signal.stft(data, fs, nperseg=1024)

    # 3. Calcul des indices correspondants aux fréquences
    # On cherche quel "index" du tableau correspond à nos fréquences en Hz
    index_source = np.searchsorted(f, freq_source)
    index_cible = np.searchsorted(f, freq_cible)
    
    # Calcul de la largeur de la bande à copier (on prend tout jusqu'à la limite de Nyquist)
    largeur_bande = len(f) - index_source
    
    # Création d'une copie du spectrogramme pour le modifier
    Zxx_modifie = Zxx.copy()

    # 4. Transposition (Le Cœur du traitement)
    # On prend la partie haute (source) et on l'ajoute à la partie basse (cible)
    # On s'assure de ne pas dépasser la taille du tableau
    limite = min(largeur_bande, len(f) - index_cible)
    
    segment_source = Zxx[index_source : index_source + limite, :]
    
    # On ajoute le signal transposé au signal existant dans la zone cible
    Zxx_modifie[index_cible : index_cible + limite, :] += segment_source * gain_transposition

    #on supprime les fréquences hautes
    Zxx_modifie[index_source:, :] *= 0.2

    # 5. Retour au domaine temporel (ISTFT - Inverse Short Time Fourier Transform)
    _, signal_sortie = signal.istft(Zxx_modifie, fs)

    # 6. Sauvegarde du fichier
    # On remet à l'échelle pour le format WAV 16-bit
    signal_sortie = np.int16(signal_sortie / np.max(np.abs(signal_sortie)) * 32767)
    wavfile.write(chemin_sortie, fs, signal_sortie)
    
    print(f"Traitement terminé. Fichier sauvegardé sous : {chemin_sortie}")
    print(f"Transposition de {freq_source}Hz vers {freq_cible}Hz effectuée.")
# --- Exemple d'utilisation ---
if __name__ == "__main__":
    print("Ce module est destiné à être utilisé via l'interface Dash.")