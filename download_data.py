import pandas as pd
import os

# Charger le CSV avec la dernière colonne = liens
df = pd.read_csv("full_dataset.csv")

# Extraire la colonne 'download_link'
links = df["download_link"].dropna()

# Télécharger chaque lien avec wget
for url in links:
    os.system(f'wget -r -N -c -np --user mathisumons --ask-password "{url}"')

