import numpy as np
from scipy.io import wavfile
from scipy import signal

def transposition_non_lineaire(chemin_entree, chemin_sortie, freq_source, freq_cible, gain_transposition=1.0, **kwargs):
    """
    Applique une transposition non-linéaire (ex: logarithmique) d'une bande haute vers une bande basse.
    Cela permet de caser plus d'informations spectrales dans une bande cible plus étroite.
    """
    fs, data = wavfile.read(chemin_entree)
    if len(data.shape) > 1:
        data = data[:, 0]
        
    data = data / np.max(np.abs(data))
    
    f, t, Zxx = signal.stft(data, fs, nperseg=1024)
    Zxx_new = Zxx.copy()
    
    idx_source = np.searchsorted(f, freq_source)
    idx_cible = np.searchsorted(f, freq_cible)
    
    # On définit une plage source à transposer (jusqu'à Nyquist)
    nb_bins_source = len(f) - idx_source
    
    # Pour chaque bin de la zone source high-freq
    for i in range(nb_bins_source):
        src_bin_idx = idx_source + i
        if src_bin_idx >= len(f): break
        
        # Mapping non linéaire : Logarithmique
        # On mappe l'intervalle [0, nb_bins_source] vers un intervalle plus court en utilisant log
        # offset = log(1 + i * alpha)
        # On choisit un alpha pour qu'une large bande rentre dans une bande plus petite
        
        alpha = 0.1 # Facteur de compression non-linéaire
        offset_non_lin = int(np.log(1 + i * alpha) * 20) # Scaling arbitraire pour l'exemple
        
        dest_bin_idx = idx_cible + offset_non_lin
        
        if dest_bin_idx < len(f) and dest_bin_idx < idx_source:
             Zxx_new[dest_bin_idx, :] += Zxx[src_bin_idx, :] * gain_transposition

    # On atténue les hautes fréquences originales
    Zxx_new[idx_source:, :] *= 0.1

    _, signal_sortie = signal.istft(Zxx_new, fs)
    signal_sortie = np.int16(signal_sortie / np.max(np.abs(signal_sortie)) * 32767)
    wavfile.write(chemin_sortie, fs, signal_sortie)
    print(f"Transposition non-linéaire terminée : {chemin_sortie}")
