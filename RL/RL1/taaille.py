import os

file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"
print("📦 Taille du fichier :", round(os.path.getsize(file_path) / 1_000_000, 2), "Mo")

import pandas as pd
df = pd.read_csv(file_path)
print("🧮 Dimensions du fichier :", df.shape)
