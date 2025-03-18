import gym
from gym import spaces
import numpy as np
import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # ✅ Évite la division par zéro
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(rsi.mean())  # Remplace les NaN initiaux


class TradingEnv(gym.Env):
    def __init__(self, data, ticker='AC.PA', initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Filtrer les données pour un ticker spécifique
        self.data = data[data['Ticker'] == ticker].reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        # Définition de l'espace d'action : 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Définition de l'espace d'observation : Open, High, Low, Close, Volume, RSI
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

        # ✅ Initialiser l'historique du portefeuille
        self.portfolio_history = [self.initial_balance]

        # Variables du portefeuille
        self.balance = initial_balance
        self.shares_held = 0
        self.total_value = initial_balance
        self.max_drawdown = initial_balance  # Ajout pour gérer le drawdown

    def reset(self, seed=None, **kwargs):
        """Réinitialise l’environnement pour un nouvel épisode"""
        super().reset(seed=seed)
        
        # ✅ Affichage final des résultats avant la réinitialisation
        if self.current_step > 0:
            print(f"\n✅ Fin de l'entraînement !")
            print(f"📊 Balance finale: {self.balance:.2f}")
            print(f"📈 Actions détenues: {self.shares_held}")
            print(f"💰 Valeur totale du portefeuille: {self.total_value:.2f}\n")

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        return self._next_observation(), {}
    
    def _next_observation(self):
        obs = self.data.iloc[self.current_step][["Open", "High", "Low", "Close", "Volume"]].values
        
        # Ajoute RSI
        rsi = compute_rsi(self.data["Close"], 14).iloc[self.current_step]

        # Construit l'observation
        observation = np.array([*obs, rsi], dtype=np.float32)

        return observation

    def compute_reward(self):
        """
        Calcule la récompense en fonction du profit, du drawdown et du Sharpe Ratio.
        """
        profit = self.total_value - self.initial_balance

        # Pénalisation du drawdown
        drawdown_penalty = -0.1 * max(0, self.max_drawdown - self.total_value)

        # Ajout du ratio de Sharpe
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) if len(returns) > 1 else 0

        reward = profit + drawdown_penalty + sharpe_ratio * 10

        return reward

    def step(self, action):
        """Applique une action (Buy, Sell, Hold) et retourne le nouvel état"""
        prev_value = self.total_value
        current_price = self.data.iloc[self.current_step]["Close"]

        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price

        # Mise à jour de la valeur totale du portefeuille
        self.total_value = self.balance + (self.shares_held * current_price)
        self.portfolio_history.append(self.total_value)

        # Mise à jour du drawdown max
        self.max_drawdown = max(self.max_drawdown, self.total_value)

        # Calcul de la récompense avec la nouvelle fonction
        reward = self.compute_reward()

        # ✅ Ajout d'une sécurité contre les NaN
        if np.isnan(reward):
            print(f"❌ NaN détecté dans reward ! prev_value={prev_value}, total_value={self.total_value}")
            reward = 0  

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  


        return self._next_observation(), reward, terminated, truncated, {}


    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Total Value: {self.total_value:.2f}')

    def seed(self, seed=None):
        np.random.seed(seed)
