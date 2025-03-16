import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    """Environnement de trading compatible avec OpenAI Gym."""
    
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Nombre d'actions détenues
        
        # Définition de l'espace d'observation et d'action
        self.observation_space = spaces.Box(low=-1, high=1, shape=(data.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def reset(self):
        """Réinitialise l'environnement au début d'un nouvel épisode."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self.data[self.current_step]

    def step(self, action):
        """Exécute une action et retourne l'état suivant, la récompense et si l'épisode est terminé."""
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Simule l'exécution de l'action
        if action == 1:  # Buy
            self.position += 1
        elif action == 2:  # Sell
            self.position -= 1

        # Calcul de la récompense (exemple simple)
        reward = self.position * (self.data[self.current_step][3] - self.data[self.current_step - 1][3])
        
        return self.data[self.current_step], reward, done, {}

