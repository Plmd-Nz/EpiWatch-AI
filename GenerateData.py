import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration des quartiers de Bukavu et coordonnees
QUARTIERS = {
    "Kadutu": {"lat": -2.5050, "lon": 28.8450},
    "Ibanda": {"lat": -2.5180, "lon": 28.8600},
    "Bagira": {"lat": -2.4800, "lon": 28.8300},
    "Panzi":  {"lat": -2.5400, "lon": 28.8700}
}

MALADIES = ["Cholera", "Mpox", "Rougeole", "Paludisme grave"]
SOURCES = [
    ("Centre de Sante", 1.0),
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
        # Creation d'une anomalie simulee sur Kadutu (ex: hausse de Cholera à J-2 et J-1)
        is_outbreak = (quartier == "Kadutu" and day >= 4)
        
        num_signals = np.random.randint(10, 25) if is_outbreak else np.random.randint(2, 6)
        
        for _ in range(num_signals):
            source_name, weight = SOURCES[np.random.choice(len(SOURCES))]
            maladie = "Cholera" if is_outbreak and np.random.rand() > 0.3 else np.random.choice(MALADIES)
            
            records.append({
                "date": current_date,
                "quartier": quartier,
                "latitude": coords["lat"] + np.random.normal(0, 0.002),
                "longitude": coords["lon"] + np.random.normal(0, 0.002),
                "maladie_suspectee": maladie,
                "symptomes": "Diarrhee aigue, deshydratation" if maladie == "Cholera" else "Fievre, eruptions",
                "source": source_name,
                "poids_confiance": weight,
                "nombre_cas": np.random.randint(1, 4)
            })

df = pd.DataFrame(records)
df.to_csv("donnees_sante_bukavu.csv", index=False)
chemin_complet = os.path.abspath("donnees_sante_bukavu.csv")
print(f"✅ Fichier genere avec succès à l'emplacement :\n{chemin_complet}")