# ===================================================
# Interface interactive Dash pour fichiers audio
# L'adresse Dash à ouvrir sur un navigateur est : http://127.0.0.1:8051/
# ===================================================

import dash
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.io import wavfile
import os
import Code_Transposition_Lineaire as audio_algo  # ton module de traitement
import Code_Compression
import Code_Transposition_Non_Lineaire

# ---------------------------------------------------
# CONFIGURATION : fichiers sources et méthodes
# ---------------------------------------------------

# Récupère le chemin absolu du dossier où se trouve ce script (compatible Mac & PC)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

audio_sources = {
    "Emma": os.path.join(BASE_DIR, "audios sources/Emma_recette.wav"),
    "Briac": os.path.join(BASE_DIR, "audios sources/Briac_recette.wav")
}

# Méthodes disponibles (tu peux ajouter plus de fonctions depuis Code_Transposition_Lineaire)
attenuation_methods = {
    "Transposition linéaire": audio_algo.transposition_frequentielle_lineaire,
    "Compression Fréquentielle": Code_Compression.compression_frequentielle,
    "Transposition Non-Linéaire": Code_Transposition_Non_Lineaire.transposition_non_lineaire
}

# Fichier temporaire pour afficher le traitement (dans le même dossier)
TEMP_FILE = os.path.join(BASE_DIR, "temp_output.wav")

# ---------------------------------------------------
# FONCTION POUR LIRE UN WAV ET CALCULER FFT
# ---------------------------------------------------

def get_fft(wav_path):
    if not os.path.exists(wav_path):
        return np.array([]), np.array([])

    fs, data = wavfile.read(wav_path)
    
    # Si stéréo, prendre le premier canal
    if len(data.shape) > 1:
        data = data[:, 0]
    
    N = len(data)
    fft_vals = np.fft.rfft(data)
    fft_freq = np.fft.rfftfreq(N, 1/fs)
    amplitude = np.abs(fft_vals)
    
    return fft_freq, amplitude

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interface Audio - Réhabilitation Auditive"),
    
    html.Div([
        html.Label("Choisir audio source :"),
        dcc.Dropdown(
            id="source-dropdown",
            options=[{"label": k, "value": k} for k in audio_sources.keys()],
            value="Emma"
        ),
    ], style={"width": "30%", "display": "inline-block", "margin-right": "20px"}),
    
    html.Div([
        html.Label("Méthode d'atténuation :"),
        dcc.Dropdown(
            id="method-dropdown",
            options=[{"label": k, "value": k} for k in attenuation_methods.keys()],
            value="Transposition linéaire"
        ),
    ], style={"width": "30%", "display": "inline-block"}),
    
    html.Div([
        html.Label("Fréquence de destination de la transposition (Hz) :"),
        dcc.Slider(
            id="freq-slider",
            min=500,
            max=8000,
            step=100,
            value=4000,
            marks={500:"500", 2000:"2k", 4000:"4k", 6000:"6k", 8000:"8k"}
        )
    ], style={"width":"90%", "margin-top":"20px"}),
    
    html.Div([
        dcc.Graph(id="graph-source"),
        dcc.Graph(id="graph-treated")
    ])
])

# ---------------------------------------------------
# CALLBACK POUR METTRE À JOUR LES GRAPHIQUES
# ---------------------------------------------------

@app.callback(
    Output("graph-source", "figure"),
    Output("graph-treated", "figure"),
    Input("source-dropdown", "value"),
    Input("method-dropdown", "value"),
    Input("freq-slider", "value")
)
def update_graphs(source, method, freq_max):
    src_file = audio_sources[source]
    
    # --- FFT audio source ---
    f_src, amp_src = get_fft(src_file)
    fig_src = go.Figure()
    if len(f_src) > 0:
        fig_src.add_trace(go.Scatter(
            x=f_src, y=amp_src, mode="lines", line=dict(color="blue")
        ))
    fig_src.update_layout(
        title=f"Audio Source : {source}",
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Amplitude"
    )
    
    # --- Traitement dynamique ---
    treated_file = TEMP_FILE
    try:
        # Appel de la fonction de traitement depuis Code_Transposition_Lineaire
        attenuation_func = attenuation_methods[method]
        # Exemple pour transposition linéaire : freq_source fixé à 4000Hz, freq_cible dépend du slider
        attenuation_func(
            chemin_entree=src_file,
            chemin_sortie=treated_file,
            freq_source=freq_max,              # seuil de perte auditive
            freq_cible=freq_max * 0.4,         # zone audible proportionnelle
            gain_transposition=0.8
        )



    except Exception as e:
        print("Erreur traitement audio :", e)
    
    # --- FFT audio traité ---
    f_tr, amp_tr = get_fft(treated_file)
    fig_tr = go.Figure()
    if len(f_tr) > 0:
        fig_tr.add_trace(go.Scatter(
            x=f_tr, y=amp_tr, mode="lines", line=dict(color="red")
        ))
    fig_tr.update_layout(
        title=f"Audio Traité : {method}, Fréquence max {freq_max}Hz",
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Amplitude"
    )
    
    return fig_src, fig_tr

# ---------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=8051)  # ici on change le port
