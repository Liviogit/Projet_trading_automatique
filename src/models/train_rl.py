import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv
from utils.data_preprocessing import load_data, preprocess_data

# Charger et prétraiter les données
df = load_data()
data_scaled, _ = preprocess_data(df)

# Initialiser l’environnement de trading
env = TradingEnv(data_scaled)

# Définir et entraîner l’agent PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Sauvegarder le modèle
model.save("C:/Users/El Hammoumi/Desktop/Projet_trading_automatique/src/models/models/ppo_trading")
