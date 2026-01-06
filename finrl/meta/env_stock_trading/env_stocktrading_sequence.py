from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockTradingSequenceEnv(gym.Env):
    """
    A sequence-aware stock trading environment for temporal models (LSTM, CNN, Transformer, CNN-LSTM).
    
    Key Changes from Original StockTradingEnv:
    1. Maintains a rolling window of historical observations
    2. Observation space is 2D: (sequence_length, features) or flattened to 1D
    3. Supports both 2D and 1D observation formats for different models
    4. Backward compatible with existing models via flatten_observations parameter
    5. Maintains the same action space and trading logic as original environment
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        macro_df=None,  # NEW: Macro economic data
        sequence_length=20,  # NEW: Length of historical sequence
        flatten_observations=False,  # NEW: Whether to flatten for MLP models
        include_returns=False,  # NEW: Include historical returns in observations
        include_volume=False,  # NEW: Include volume data,
        sharpe_window=20, # NEW: Rolling window for Sharpe ratio calculation
        dsr_eta: float = 0.1, # NEW: Update rate for Differential Sharpe Ratio
        reward_type: str = "pnl",  # NEW: Reward function type: 'pnl' or 'sharpe' or dsr
        model_name="",
        mode="",
        iteration="",
        seed=""
    ):
        """
        Initialize the sequence-aware stock trading environment.
        
        State Structure (same as original):
        [cash, stock1_price, stock2_price, ..., stock1_shares, stock2_shares, ..., tech_indicators...]
        
        Args:
            macro_df: DataFrame with macro economic indicators (optional)
            sequence_length: Number of historical days to include in observations
            flatten_observations: If True, flatten to 1D for MLP models. If False, keep 2D for sequence models
            include_returns: Whether to include historical returns
            include_volume: Whether to include volume information
            The dsr_eta parameter is the update rate or smoothing factor for the Differential Sharpe Ratio (DSR) calculation.
            It controls how much influence the most recent return has on the running average of returns and volatility.
            A small eta (e.g., 0.01): Gives more weight to past performance. The running averages change slowly, resulting in a smoother but less responsive reward signal. 
            This is like calculating the Sharpe ratio over a long period. A large eta (e.g., 0.1): Gives more weight to the most recent return.
            The running averages change quickly, making the reward signal very responsive but potentially noisy.
            This is like calculating the Sharpe ratio over a very short period.

        """
        self.day = day
        self.df = df
        self.macro_df = macro_df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.seed = seed
        
        # NEW: Sequence parameters
        self.sequence_length = sequence_length
        self.flatten_observations = flatten_observations
        self.include_returns = include_returns
        self.include_volume = include_volume
        self.sharpe_window = sharpe_window # NEW: Rolling window for Sharpe ratio calculation
        self.reward_type = reward_type
        self.dsr_eta = dsr_eta # NEW

        # NEW: Initialize state for Differential Sharpe Ratio
        self.dsr_a = 0.0  # Running mean of returns
        self.dsr_b = 0.0  # Running second moment of returns
        self.last_sharpe = 0.0

        # Calculate enhanced state dimensions
        base_features_per_stock = len(self.tech_indicator_list)
        if self.include_returns:
            base_features_per_stock += 1
        if self.include_volume:
            base_features_per_stock += 1

        # Macro features
        if self.macro_df is not None:
            macro_features = self.macro_df.shape[1] - 1  # subtract date column
        else:
            macro_features = 0
            
        # Enhanced state dimension: cash + prices + shares + enhanced_tech_features + macro_features
        self.enhanced_state_dim = 1 + self.stock_dim + self.stock_dim + (base_features_per_stock * self.stock_dim) + macro_features
        
        # Action space: same as original (buy/sell actions)
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_space,))
        
        # NEW: Observation space design
        if self.flatten_observations:
            # For MLP models: flatten sequence to 1D
            obs_dim = self.sequence_length * self.enhanced_state_dim
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,)
            )
        else:
            # For sequence models: (sequence_length, features)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sequence_length, self.enhanced_state_dim)
            )
        
        # Initialize environment state
        self.data = self.df.loc[self.day, :]
        self.terminal = False

        # Handle different previous_state formats
        if not initial and previous_state is not None and len(previous_state) > 0:
            # Check if previous_state is 2D (from sequence environment)
            if isinstance(previous_state, np.ndarray) and len(previous_state.shape) == 2:
                # Take the last observation (most recent state) to get 1D state
                self.previous_state = previous_state[-1].tolist()
                print(f"Converted 2D previous_state {previous_state.shape} to 1D state with {len(self.previous_state)} elements")
            elif isinstance(previous_state, list) and len(previous_state) > 0:
                # Check if it's a list of arrays/sequences
                if isinstance(previous_state[0], (np.ndarray, list)) and hasattr(previous_state[0], '__len__'):
                    # It's a sequence of states, take the last one
                    if isinstance(previous_state[-1], np.ndarray):
                        self.previous_state = previous_state[-1].tolist()
                    else:
                        self.previous_state = previous_state[-1]
                    print(f"Extracted last state from sequence of {len(previous_state)} states")
                else:
                    # It's already a 1D list
                    self.previous_state = previous_state
            else:
                # Fallback: use as-is
                self.previous_state = previous_state
        else:
            self.previous_state = previous_state
        
        # Initialize state (1D array like original)
        self.state = self._initiate_state()
        
        # Initialize metrics
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # Memory containers
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.cash_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        
        # NEW: Historical observation buffer for sequences
        self.observation_buffer = []
        
        # Initialize with first observation
        self._initialize_observation_buffer()
        
        if isinstance(self.seed, int):
            self._seed(self.seed)

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

    def _build_daily_observation(self, daily_data, day_idx):
        """
        Build enhanced observation for a single day.
        
        Enhanced State Structure: 
        [cash, stock_prices, stock_shares, tech_indicators, returns?, volume?, macro_features?]
        
        Returns:
            np.array: 1D state vector for one timestep
        """
        state = []
        
        # 1. Cash (portfolio state)
        state.append(self.state[0] if hasattr(self, 'state') else self.initial_amount)
        
        # 2. Stock prices (market state)
        if len(daily_data.tic.unique()) > 1:
            state.extend(daily_data.close.values.tolist())
        else:
            state.append(daily_data.close.iloc[0])
        
        # 3. Stock shares (position state)
        if hasattr(self, 'state') and len(self.state) > self.stock_dim + 1:
            # Use current holdings
            state.extend(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        else:
            # Use initial holdings
            state.extend(self.num_stock_shares)
        
        # 4. Technical indicators (market sentiment/momentum)
        for tech in self.tech_indicator_list:
            if len(daily_data.tic.unique()) > 1:
                state.extend(daily_data[tech].values.tolist())
            else:
                state.append(daily_data[tech].iloc[0])

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

        # 7. Macro features (if available) - economic context
        if self.macro_df is not None:
            # Get the actual date from the current daily_data
            if len(daily_data.tic.unique()) > 1:
                current_date = daily_data.date.unique()[0]
            else:
                current_date = daily_data.date.iloc[0]
            
            # Find matching row in macro_df by date
            macro_row = self.macro_df[self.macro_df['date'] == current_date]
            
            if len(macro_row) > 0:
                # Found exact date match
                macro_features = macro_row.iloc[0, 1:].values.flatten()  # skip date column
                state.extend(macro_features.tolist())
            else:
                # Fallback: find the most recent macro data before or on current_date
                available_macro_dates = self.macro_df[self.macro_df['date'] <= current_date]
                if len(available_macro_dates) > 0:
                    # Use the most recent available macro data
                    latest_macro_row = available_macro_dates.iloc[-1]
                    macro_features = latest_macro_row.iloc[1:].values.flatten()  # skip date column
                    state.extend(macro_features.tolist())
                else:
                    # No historical macro data available - use zeros
                    macro_feature_count = self.macro_df.shape[1] - 1  # subtract date column
                    state.extend([0.0] * macro_feature_count)

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

    def _sell_stock(self, index, action):
        """Same sell logic as original environment."""
        def _do_sell_normal():
            if (
                len(self.state) > index + 2 * self.stock_dim + 1 and
                self.state[index + 2 * self.stock_dim + 1] != True
            ):
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """Same buy logic as original environment."""
        def _do_buy():
            if (
                len(self.state) <= index + 2 * self.stock_dim + 1 or
                self.state[index + 2 * self.stock_dim + 1] != True
            ):
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares

    def _make_plot(self):
        """Same plotting logic as original environment."""
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        """Step function with sequence-aware observations but same trading logic."""
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            # Terminal state - same as original
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            df_total_value["cash"] = self.cash_memory
            df_total_value['cash_share'] = df_total_value['cash'] / df_total_value['account_value']
            
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration, self.seed
                    )
                )
                plt.close()

            return self._get_observation(), self.reward, self.terminal, False, {}

        else:
            # Same trading logic as original
            actions = actions * self.hmax
            actions = actions.astype(int)
            
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # Move to next day
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            
            self.state = self._update_state()

            # NEW: Update observation buffer with new state
            new_obs = self._build_daily_observation(self.data, self.day)
            self.observation_buffer.append(new_obs)
            self.observation_buffer = self.observation_buffer[-self.sequence_length:]

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            # NEW: Selectable reward function
            if self.reward_type == "sharpe":
                # We need at least 3 asset values to get 2 returns for a std dev
                if len(self.asset_memory) > 2:
                    # Use a window of up to `sharpe_window` past returns
                    window_size = min(len(self.asset_memory), self.sharpe_window + 1)
                    window_asset_values = np.array(self.asset_memory[-window_size:])
                    
                    # Calculate daily returns within the window
                    daily_returns = (window_asset_values[1:] / window_asset_values[:-1]) - 1
                    
                    # Calculate Sharpe Ratio for the window
                    mean_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    
                    if std_return != 0:
                        # Annualize for better scaling (assuming 252 trading days)
                        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                        self.reward = sharpe_ratio
                    else:
                        # If no volatility, reward is 0 (neutral)
                        self.reward = 0.0
                else:
                    # Not enough data for a meaningful Sharpe Ratio, reward is 0
                    self.reward = 0.0
            elif self.reward_type == "dsr": # NEW: Differential Sharpe Ratio
                # We need at least 2 asset values to calculate one return
                if len(self.asset_memory) > 1:
                    current_return = (self.asset_memory[-1] / self.asset_memory[-2]) - 1
                    
                    # Update running moments (exponential moving average)
                    self.dsr_a = (1 - self.dsr_eta) * self.dsr_a + self.dsr_eta * current_return
                    self.dsr_b = (1 - self.dsr_eta) * self.dsr_b + self.dsr_eta * (current_return ** 2)
                    
                    # Calculate current Sharpe and the differential reward
                    current_std = (self.dsr_b - self.dsr_a**2)**0.5
                    if current_std != 0:
                        current_sharpe = self.dsr_a / current_std
                        self.reward = current_sharpe - self.last_sharpe
                        self.last_sharpe = current_sharpe
                    else:
                        self.reward = 0.0
                else:
                    self.reward = 0.0
            else:  # Default to 'pnl' (profit and loss)
                self.reward = end_total_asset - begin_total_asset

            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.cash_memory.append(self.state[0])
            self.state_memory.append(self.state)

            # Validate observation
            observation = self._get_observation()
            assert not np.any(np.isnan(observation)), "Observation contains NaN values"
            assert not np.any(np.isinf(observation)), "Observation contains Inf values"

        return observation, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset environment with sequence buffer initialization."""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.cash_memory = [self.initial_amount]

        # Reset DSR states
        self.dsr_a = 0.0
        self.dsr_b = 0.0
        self.last_sharpe = 0.0
        
        # NEW: Reset observation buffer
        self.observation_buffer = []
        self._initialize_observation_buffer()

        self.episode += 1

        return self._get_observation(), {}

    def render(self, mode="human", close=False):
        """Render current state."""
        return self._get_observation()

    def _initiate_state(self):
        """Same state initiation logic as original environment."""
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        """Same state update logic as original environment."""
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )
        else:
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )
        return state

    def _get_date(self):
        """Same date extraction logic as original environment."""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_state_memory(self):
        """Same state memory saving logic as original environment."""
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def save_asset_memory(self):
        """Same asset memory saving logic as original environment."""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """Same action memory saving logic as original environment."""
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        """Same seeding logic as original environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Same Stable-Baselines3 compatibility as original environment."""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
