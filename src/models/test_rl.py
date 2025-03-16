import gym
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv
from utils.data_preprocessing import load_data, preprocess_data

# Charger les données et l'environnement
df = load_data()
data_scaled, _ = preprocess_data(df)
env = TradingEnv(data_scaled)

# Charger le modèle entraîné
model = PPO.load("C:/Users/El Hammoumi/Desktop/Projet_trading_automatique/src/models/models/ppo_trading")

# Tester l'agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
