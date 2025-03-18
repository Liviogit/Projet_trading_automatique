import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Chargement et préparation des données
# Supposons que votre dataset est dans un fichier CSV 'trading_data.csv'
df = pd.read_csv('trading_data.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Calcul des rendements logarithmiques journaliers
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

# Création de quelques variables explicatives simples (par exemple, rendements décalés)
df['lag1'] = df['log_return'].shift(1)
df['lag2'] = df['log_return'].shift(2)
df.dropna(inplace=True)

# 2. Séparation des données en ensembles d'entraînement et de test
features = ['lag1', 'lag2']
target = 'log_return'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Entraînement d'un modèle d'apprentissage automatique (régression linéaire)
model = LinearRegression()
model.fit(X_train, y_train)

# Calcul de l'erreur résiduelle pour déterminer la volatilité non expliquée
y_pred = model.predict(X_train)
residuals = y_train - y_pred
residual_std = residuals.std()
print("Erreur résiduelle (std) =", residual_std)

# 4. Simulation de Monte Carlo
# Paramètres de la simulation
n_simulations = 1000       # nombre de simulations
forecast_horizon = 30      # nombre de jours dans le futur
last_known_price = df['Close'].iloc[-1]

# Pour lancer la simulation, nous devons définir des conditions initiales
# On part du dernier rendement observé pour initialiser les variables décalées
last_lag1 = df['log_return'].iloc[-1]
last_lag2 = df['log_return'].iloc[-2]

# Stockage des chemins simulés
simulated_paths = np.zeros((forecast_horizon, n_simulations))

for sim in range(n_simulations):
    price = last_known_price
    lag1 = last_lag1
    lag2 = last_lag2
    for t in range(forecast_horizon):
        # Préparation des variables pour la prédiction
        X_new = np.array([[lag1, lag2]])
        # Prédiction du rendement
        predicted_return = model.predict(X_new)[0]
        # Ajout d'un bruit aléatoire basé sur l'écart-type des résidus (pour simuler l'incertitude)
        simulated_return = predicted_return + np.random.normal(0, residual_std)
        # Calcul du nouveau prix
        price = price * np.exp(simulated_return)
        simulated_paths[t, sim] = price
        # Mise à jour des lags pour la prochaine itération
        lag2 = lag1
        lag1 = simulated_return

# 5. Visualisation des simulations
plt.figure(figsize=(12,6))
plt.plot(simulated_paths, color='gray', alpha=0.2)
plt.title("Simulation Monte Carlo des Prix Futurs")
plt.xlabel("Jours dans le futur")
plt.ylabel("Prix simulé")
plt.show()

# 6. Statistiques sur la simulation
# Par exemple, calculer la moyenne et les quantiles du prix à la fin de l'horizon
final_prices = simulated_paths[-1, :]
mean_final_price = np.mean(final_prices)
quantiles = np.percentile(final_prices, [5, 50, 95])
print("Prix final moyen :", mean_final_price)
print("Quantiles 5%, 50%, 95% :", quantiles)

#Explications:

#Préparation des données :
#Le code charge les données de trading, les trie par date, puis calcule les rendements logarithmiques journaliers. Des variables explicatives simples (les rendements décalés) sont créées pour servir d’entrée au modèle.

#Modèle d’apprentissage automatique :
#Une régression linéaire est entraînée pour prédire le rendement actuel à partir des rendements précédents. On calcule ensuite l’écart-type des résidus pour quantifier l’incertitude de la prédiction.

#Simulation Monte Carlo :
#Pour chaque simulation, le modèle prédit le rendement futur, auquel on ajoute un terme aléatoire (basé sur l’écart-type des résidus). Ce rendement est utilisé pour actualiser le prix en simulant son évolution sur un horizon donné. Les variables décalées (lags) sont mises à jour à chaque étape.

#Visualisation et analyse :
#Le résultat est visualisé par un graphique montrant tous les chemins simulés, et des statistiques (moyenne, quantiles) sont calculées sur le prix final pour évaluer l’incertitude des prédictions.