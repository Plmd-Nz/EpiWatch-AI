import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------------------------------------------
st.set_page_config(page_title="EpiWatch AI - RDC", layout="wide")

st.title("🚨 EpiWatch AI — Veille Épidémiologique Collaborative")
st.caption("Détection précoce de signaux faibles sanitaires par IA (Gemma 2 9B) à Bukavu")

# -------------------------------------------------------------------
# 1. GESTION DE LA CLÉ API (Fichier .env ou Entrée Manuelle)
# -------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# Si la clé n'est pas trouvée dans .env, on propose la saisie dans la barre latérale
if not api_key:
    api_key = st.sidebar.text_input("Clé API Google AI Studio", type="password")

if not api_key:
    st.warning("⚠️ Veuillez configurer votre clé API dans le fichier `.env` ou la saisir dans le panneau latéral de gauche pour démarrer.")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_ID = "gemma-4-26b-a4b-it"

# -------------------------------------------------------------------
# 2. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES (BUKAVU)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    QUARTIERS = {
        "Kadutu": {"lat": -2.5050, "lon": 28.8450},
        "Ibanda": {"lat": -2.5180, "lon": 28.8600},
        "Bagira": {"lat": -2.4800, "lon": 28.8300},
        "Panzi":  {"lat": -2.5400, "lon": 28.8700}
    }
    MALADIES = ["Choléra", "Mpox", "Rougeole", "Paludisme grave"]
    SOURCES = [
        ("Centre de Santé", 1.0),
        ("Pharmacie", 0.7),
        ("Relais Communautaire", 0.8),
        ("Signalement Citoyen", 0.4)
    ]
    
    np.random.seed(42)
    records = []
    start_date = datetime.now() - timedelta(days=7)

    for day in range(7):
        current_date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
        for quartier, coords in QUARTIERS.items():
            # Cluster anormal de Choléra sur Kadutu
            is_outbreak = (quartier == "Kadutu" and day >= 4)
            num_signals = np.random.randint(12, 22) if is_outbreak else np.random.randint(2, 5)
            
            for _ in range(num_signals):
                idx = np.random.choice(len(SOURCES))
                source_name, weight = SOURCES[idx]
                maladie = "Choléra" if is_outbreak and np.random.rand() > 0.3 else np.random.choice(MALADIES)
                
                records.append({
                    "date": current_date,
                    "quartier": quartier,
                    "latitude": coords["lat"] + np.random.normal(0, 0.0015),
                    "longitude": coords["lon"] + np.random.normal(0, 0.0015),
                    "maladie_suspectee": maladie,
                    "source": source_name,
                    "poids_confiance": weight,
                    "nombre_cas": np.random.randint(1, 3)
                })

    df = pd.DataFrame(records)
    df['score_impact'] = df['nombre_cas'] * df['poids_confiance']
    return df, QUARTIERS

df_health, QUARTIERS = load_data()

# Moteur d'Agrégation
synthesis = df_health.groupby(['quartier', 'maladie_suspectee']).agg(
    total_cas=('nombre_cas', 'sum'),
    score_confiance=('score_impact', 'sum'),
    sources_uniques=('source', 'nunique')
).reset_index().sort_values(by='score_confiance', ascending=False)

# -------------------------------------------------------------------
# 3. INTERFACE UTILISATEUR
# -------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

# --- COLONNE 1 : CARTOGRAPHIE FOLIUM ---
with col1:
    st.subheader("🗺️ Carte des Signaux Sanitaires (Bukavu)")
    m = folium.Map(location=[-2.5100, 28.8500], zoom_start=13, tiles="CartoDB positron")
    
    # Couche HeatMap
    heat_data = [[row['latitude'], row['longitude'], row['score_impact']] for _, row in df_health.iterrows()]
    HeatMap(heat_data, radius=16, blur=11).add_to(m)

    # Marqueurs par Quartier
    for q, coords in QUARTIERS.items():
        cas_q = df_health[df_health['quartier'] == q]['nombre_cas'].sum()
        color = "red" if q == "Kadutu" else "blue"
        folium.CircleMarker(
            location=[coords['lat'], coords['lon']],
            radius=6 + (cas_q / 4),
            popup=f"Quartier: {q} | Cas: {cas_q}",
            color=color, fill=True, fill_color=color
        ).add_to(m)

    st_folium(m, width=600, height=450)

# --- COLONNE 2 : ANALYSE GEMMA 2 ---
with col2:
    st.subheader("🤖 Analyse d'Agent IA (Gemma 2)")
    quartier_sel = st.selectbox("Sélectionner un quartier à analyser :", list(QUARTIERS.keys()))
    
    if st.button("Lancer l'analyse épidémiologique"):
        with st.spinner("Analyse du signal faible par Gemma 2..."):
            data_q = synthesis[synthesis['quartier'] == quartier_sel].to_json(orient="records")
            
            prompt = f"""
            Vous êtes EpiWatch AI, un assistant d'aide à la décision en veille épidémiologique pour la RDC.
            Examinez les données sanitaires récentes pour le quartier : {quartier_sel}.
            
            Données agrégées multi-sources (Centres de santé, Pharmacies, Relais communautaires, Citoyens) :
            {data_q}
            
            Veuillez fournir une analyse structurée en 3 points :
            1. **Évaluation du signal faible :** S'agit-il d'une anomalie/cluster ou d'un bruit statistique ? (Prenez en compte le poids de confiance des sources).
            2. **Hypothèses contextuelles :** Quelles raisons environnementales ou sanitaires locales pourraient expliquer cette hausse (ex. accès à l'eau, pluies) ? *Formulez des hypothèses, pas des certitudes absolues.*
            3. **Plan d'Action Réflexe :** Proposez 3 actions prioritaires et concrètes pour les relais communautaires et les autorités de santé.
            """
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                )
            )
            st.markdown(response.text)