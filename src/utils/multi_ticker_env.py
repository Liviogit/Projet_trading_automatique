import gym
import numpy as np
import pandas as pd
from gym import spaces

class MultiTickerTradingEnv(gym.Env):
    def __init__(self, df, tickers):
        super().__init__()
        self.df = df
        self.tickers = tickers
        self.num_assets = len(tickers)

        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets * 5 + 1,),
            dtype=np.float32
        )

        self.initial_balance = 10000
        self.df["Date"] = self.df["Datetime"].dt.normalize()
        self.unique_dates = sorted(self.df["Date"].unique())
        self.max_steps = len(self.unique_dates) - 1

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.total_value = self.balance
        self.current_step = 0
        self.current_date = self.unique_dates[self.current_step]

        self.portfolio_history = [self.total_value]
        self.portfolio_returns = []

        return self._next_observation()

    def _next_observation(self):
        obs = []

        for ticker in self.tickers:
            ticker_df = self.df[(self.df["Date"] == self.current_date) & (self.df["Ticker"] == ticker)]
            if not ticker_df.empty:
                row = ticker_df.iloc[0]
                obs.extend([row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
            else:
                obs.extend([0, 0, 0, 0, 0])

        obs.append(self.balance)
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        prev_value = self.total_value

        for i, action in enumerate(actions):
            ticker = self.tickers[i]
            price_row = self.df[(self.df["Date"] == self.current_date) & (self.df["Ticker"] == ticker)]
            if price_row.empty:
                continue

            close_price = price_row.iloc[0]["Close"]

            if action == 1:  # Buy
                quantity = self.balance // (close_price * self.num_assets)
                self.balance -= quantity * close_price
                self.positions[ticker] += quantity

            elif action == 2:  # Sell
                quantity = self.positions[ticker]
                self.balance += quantity * close_price
                self.positions[ticker] = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        self.current_date = self.unique_dates[self.current_step] if not done else self.current_date

        current_value = self.balance
        for ticker in self.tickers:
            row = self.df[(self.df["Date"] == self.current_date) & (self.df["Ticker"] == ticker)]
            if not row.empty:
                current_value += self.positions[ticker] * row.iloc[0]["Close"]

        # 💡 Calcul du reward (Sharpe - Drawdown penalty)
        step_return = (current_value - prev_value) / (prev_value + 1e-8)
        self.portfolio_returns.append(step_return)

        sharpe = np.mean(self.portfolio_returns[-10:]) / (np.std(self.portfolio_returns[-10:]) + 1e-8) if len(self.portfolio_returns) >= 2 else 0
        peak = max(self.portfolio_history)
        drawdown = (peak - current_value) / (peak + 1e-8)
        penalty = drawdown if drawdown > 0.05 else 0  # pénalité si drawdown > 5%
        reward = sharpe - penalty

        self.total_value = current_value
        self.portfolio_history.append(self.total_value)

        return self._next_observation(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Date: {self.current_date}")
        print(f"Balance: {self.balance}")
        print(f"Total Portfolio Value: {self.total_value}")
        print(f"Positions: {self.positions}")
