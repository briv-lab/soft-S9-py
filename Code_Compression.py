import numpy as np
from scipy.io import wavfile
from scipy import signal

def compression_frequentielle(chemin_entree, chemin_sortie, freq_source, freq_cible=None, gain_transposition=1.0, **kwargs):
    """
    Applique une compression fréquentielle sur les hautes fréquences.
    A partir de la frequence de coupure (freq_source), les fréquences sont comprimées.
    L'argument freq_cible est ignoré mais présent pour compatibilité.
    """
    freq_coupure = freq_source
    ratio_compression = 2.0 # Valeur par défaut ou calculée

    fs, data = wavfile.read(chemin_entree)
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Normalisation
    data = data / np.max(np.abs(data))
    
    # STFT
    f, t, Zxx = signal.stft(data, fs, nperseg=1024)
    Zxx_compile = np.zeros_like(Zxx)
    
    idx_coupure = np.searchsorted(f, freq_coupure)
    
    # Copie des fréquences basses inchangées
    Zxx_compile[:idx_coupure, :] = Zxx[:idx_coupure, :]
    
    # Compression des hautes fréquences
    for i in range(idx_coupure, len(f)):
        freq_orig = f[i]
        # Formule de compression : f_new = fc + (f_orig - fc) / ratio
        freq_new = freq_coupure + (freq_orig - freq_coupure) / ratio_compression
        
        idx_new = np.searchsorted(f, freq_new)
        
        # Sommation de l'énergie dans le nouveau bin
        if idx_new < len(f):
            Zxx_compile[idx_new, :] += Zxx[i, :] * gain_transposition
            
    # ISTFT
    _, signal_sortie = signal.istft(Zxx_compile, fs)
    
    # Normalisation et sauvegarde
    signal_sortie = np.int16(signal_sortie / np.max(np.abs(signal_sortie)) * 32767)
    wavfile.write(chemin_sortie, fs, signal_sortie)
    print(f"Compression terminée : {chemin_sortie}")
