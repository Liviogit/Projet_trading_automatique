import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """Environnement de trading basé sur OpenAI Gym"""
    
    def __init__(self, data, ticker='AC.PA', initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Filtrer les données pour un ticker spécifique
        self.data = data[data['Ticker'] == ticker].reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Définition de l'espace d'action : 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Définition de l'espace d'observation : Open, High, Low, Close, Volume
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Variables du portefeuille
        self.balance = initial_balance
        self.shares_held = 0
        self.total_value = initial_balance
    
    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        return self._next_observation()
    
    def _next_observation(self):
        """Renvoie l'état actuel du marché"""
        obs = self.data.iloc[self.current_step][["Open", "High", "Low", "Close", "Volume"]].values
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Applique une action (Buy, Sell, Hold) et retourne le nouvel état"""
        prev_value = self.total_value
        current_price = self.data.iloc[self.current_step]["Close"]
        
        # Gestion des actions
        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price

        # Mettre à jour la valeur totale du portefeuille
        self.total_value = self.balance + (self.shares_held * current_price)
        
        # Calculer la récompense (profit ou perte)
        reward = self.total_value - prev_value
        
        # Vérifier si l'épisode est terminé
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        """Affiche l'état actuel du trading"""
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Total Value: {self.total_value:.2f}')
