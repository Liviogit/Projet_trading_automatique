import pandas as pd

# Charger le fichier CSV avec l'index 'Datetime'
df = pd.read_csv('Data/cac40_stock_data.csv', header=[0, 1], index_col=0)

# Vérifier les premières lignes du DataFrame
print("Avant transformation:")
print(df.head())

# Réorganiser les données en format long en utilisant 'stack()' sur les colonnes multi-index
df_long = df.stack(level=['Price', 'Ticker']).reset_index()

# Renommer les colonnes pour plus de clarté
df_long.columns = ['Datetime', 'Price', 'Ticker', 'Value']

# Vérifier les premières lignes après transformation
print("\nAprès transformation en format long:")
print(df_long.head())

# Sauvegarder le DataFrame transformé si nécessaire
df_long.to_csv('Data/cac40_clean_format.csv', index=False)

print("Les données ont été réorganisées et enregistrées dans 'cac40_long_format.csv'.")