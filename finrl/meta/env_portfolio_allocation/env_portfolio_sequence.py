from __future__ import annotations

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockPortfolioSequenceEnv(gym.Env):
    """
    A sequence-aware portfolio allocation environment for temporal models (LSTM, CNN, Transformer, CNN-LSTM).
    
    Key Changes from Original:
    1. Maintains a rolling window of historical observations
    2. Observation space is 3D: (sequence_length, features, stocks) or flattened to 2D
    3. Supports both 2D and 1D observation formats for different models
    4. Backward compatible with existing models via flatten_observations parameter
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
        initial = True,
        previous_state = [],
        sequence_length=20,  # NEW: Length of historical sequence
        flatten_observations=False,  # NEW: Whether to flatten for MLP models
        include_returns=False,  # NEW: Include historical returns in observations
        include_volume=False,  # NEW: Include volume data,
        model_name="",
        mode="",
        iteration="",
        seed=""

    ):
        """
        Initialize the sequence-aware portfolio environment.
        
        State Structure (following stock trading env pattern):
        [portfolio_value, stock1_price, stock2_price, ..., weight1, weight2, ..., tech_indicators...]
        
        Args:
            sequence_length: Number of historical days to include in observations
            flatten_observations: If True, flatten to 1D for MLP models. If False, keep 2D for sequence models
            include_returns: Whether to include historical returns
            include_volume: Whether to include volume information
        """
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        self.initial = initial
        self.previous_state = previous_state
        
        # NEW: Sequence parameters
        self.sequence_length = sequence_length
        self.flatten_observations = flatten_observations
        self.include_returns = include_returns
        self.include_volume = include_volume

        # for muliple runs
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.seed = seed
        
        # Calculate state dimensions (1D like stock trading env)
        # State: [portfolio_value, prices, weights, tech_indicators, returns?, volume?]
        base_features_per_stock = len(self.tech_indicator_list)
        if self.include_returns:
            base_features_per_stock += 1
        if self.include_volume:
            base_features_per_stock += 1

        # Total state dimension:  prices + weights + tech_features * stocks + portfolio_value
        self.state_dim = self.stock_dim + self.stock_dim + (base_features_per_stock * self.stock_dim) + 1

        # Action space: portfolio weights (softmax normalized)
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))
        
        # NEW: Observation space design
        if self.flatten_observations:
            # For MLP models: flatten sequence to 1D
            obs_dim = self.sequence_length * self.state_dim
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,)
            )
        else:
            # For sequence models: (sequence_length, state_features)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sequence_length, self.state_dim)
            )
        
        # Initialize environment
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        
        # Initialize portfolio weights to equal weight
        if self.initial:
            self.current_weights = np.array([1.0 / self.stock_dim] * self.stock_dim)
        else: 
        # extract weights from the last step (sequence length) of the previous state, 
        # state is an array [sequence length, list of stock prices for len(daily_data.tic.unique()) 
        # followed by weights for len(daily_data.tic.unique()) stocks and then followed by some indicators]
            self.current_weights = np.array(self.previous_state[-1][self.stock_dim:2*self.stock_dim])
            self.initial_amount = self.previous_state[-1][-1]

        self.portfolio_value = self.initial_amount

        # Memory containers
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.current_weights.tolist()]
        self.date_memory = []
        
        # NEW: Historical observation buffer for sequences
        self.observation_buffer = []
        
        # Initialize with first observation
        self._initialize_observation_buffer()

    def _initialize_observation_buffer(self):
        """Initialize the observation buffer with historical data."""
        start_idx = max(0, self.day - self.sequence_length + 1)
        
        for i in range(start_idx, self.day + 1):
            if i < len(self.df.index.unique()):
                daily_data = self.df.loc[i, :]
                obs = self._build_daily_observation(daily_data, i)
                self.observation_buffer.append(obs)
        
        # Pad if we don't have enough historical data
        while len(self.observation_buffer) < self.sequence_length:
            # Duplicate first observation for padding
            self.observation_buffer.insert(0, self.observation_buffer[0].copy())
        
        # Keep only the last sequence_length observations
        self.observation_buffer = self.observation_buffer[-self.sequence_length:]
        
        # Set current data and date memory
        self.data = self.df.loc[self.day, :]
        self.date_memory = [self.data.date.unique()[0]]

    def _build_daily_observation(self, daily_data, day_idx):
        """
        Build observation for a single day following stock trading env pattern.
        
        IMPROVED State Structure: [stock_prices, portfolio_weights, tech_indicators, returns?, volume?]
        (Removed portfolio_value as it's not actionable)
        
        Returns:
            np.array: 1D state vector for one timestep
        """
        state = []
        
        # 1. Current stock prices (market state)
        if len(daily_data.tic.unique()) > 1:
            state.extend(daily_data.close.values.tolist())
        else:
            state.append(daily_data.close.iloc[0])
        
        # 2. Current portfolio weights (allocation state)
        state.extend(self.current_weights.tolist())
        
        # 3. Technical indicators for all stocks (market sentiment/momentum)
        for tech in self.tech_indicator_list:
            if len(daily_data.tic.unique()) > 1:
                state.extend(daily_data[tech].values.tolist())
            else:
                state.append(daily_data[tech].iloc[0])

        # 4. current portfolio worth
        state.append(self.portfolio_value)

        # 5. Returns (if enabled) - market performance
        if self.include_returns:
            if day_idx > 0:
                prev_data = self.df.loc[day_idx - 1, :]
                if len(daily_data.tic.unique()) > 1:
                    returns = (daily_data.close.values / prev_data.close.values) - 1
                    state.extend(returns.tolist())
                else:
                    returns = (daily_data.close.iloc[0] / prev_data.close.iloc[0]) - 1
                    state.append(returns)
            else:
                # First day: zero returns
                if len(daily_data.tic.unique()) > 1:
                    state.extend([0.0] * self.stock_dim)
                else:
                    state.append(0.0)

        # 6. Volume (if enabled and available) - market liquidity
        if self.include_volume:
            if 'volume' in daily_data.columns:
                if len(daily_data.tic.unique()) > 1:
                    volumes = daily_data.volume.values
                    # Normalize volume to prevent scale issues
                    normalized_volumes = np.log(volumes + 1)
                    state.extend(normalized_volumes.tolist())
                else:
                    volume = daily_data.volume.iloc[0]
                    normalized_volume = np.log(volume + 1)
                    state.append(normalized_volume)
            else:
                # If volume not available, use zeros
                if len(daily_data.tic.unique()) > 1:
                    state.extend([0.0] * self.stock_dim)
                else:
                    state.append(0.0)

        if len(state) != self.state_dim:
            print(f"ERROR: State dimension mismatch!")
        
        return np.array(state)
    
    def _get_observation(self):
        """
        Get the current observation (sequence of historical data).
        
        Returns:
            np.array: Observation in the format expected by the model
        """
        if self.flatten_observations:
            # Flatten everything to 1D for MLP models
            return np.concatenate(self.observation_buffer)
        else:
            # Stack as 2D array for sequence models: (sequence_length, features)
            return np.stack(self.observation_buffer, axis=0)

    def step(self, actions):
        """Step function with sequence-aware observations."""
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # Terminal state - save plots and print statistics
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    )
                )

                df_total_value = pd.DataFrame(self.asset_memory)
                df_total_value.columns = ["account_value"]
                df_total_value["date"] = self.date_memory
                df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    ),
                    index=False,
                )

            return self._get_observation(), self.reward, self.terminal, False, {}

        else:
            # Normalize actions to portfolio weights
            # weights = self.softmax_normalization(actions) - library solution
            total_weight = np.sum(actions)
            weights = actions / total_weight
            self.current_weights = weights  # Update current weights
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # Move to next day
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Calculate portfolio return
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            
            # Update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            gain = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            # NEW: Update observation buffer with new state
            new_obs = self._build_daily_observation(self.data, self.day)
            self.observation_buffer.append(new_obs)
            self.observation_buffer = self.observation_buffer[-self.sequence_length:]  # Keep only last N observations

            # Save to memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # Reward is the gain
             # self.reward = gain * self.reward_scaling # this produced inferior results - needs to be investigated
            self.reward = gain

        return self._get_observation(), self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset environment with sequence buffer initialization."""
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]

        if self.initial:
            self.current_weights = np.array([1.0 / self.stock_dim] * self.stock_dim)
        else: 
            self.current_weights = np.array(self.previous_state[-1][self.stock_dim:2*self.stock_dim])

        self.actions_memory = [self.current_weights.tolist()]
        
        self.date_memory = []
        self.observation_buffer = []
        
        # Initialize observation buffer
        self._initialize_observation_buffer()
        
        return self._get_observation(), {}

    def render(self, mode="human"):
        """Render current state."""
        return self._get_observation()

    def softmax_normalization(self, actions):
        """Softmax normalization for portfolio weights."""
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        """Save asset memory to DataFrame."""
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return, "account_value": self.asset_memory}
        )
        return df_account_value

    def save_action_memory(self):
        """Save action memory to DataFrame."""
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get Stable-Baselines3 compatible environment."""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
