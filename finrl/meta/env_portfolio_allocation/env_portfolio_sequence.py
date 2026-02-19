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
        macro_df,
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
        dsr_eta: float = 0.1, # NEW: Update rate for Differential Sharpe Ratio
        reward_type: str = "pnl",  # NEW: Reward function type: 'pnl', 'sharpe', 'dsr', 'log_return', or 'active_return'
        log_return_window: int = 20,  # NEW: Window for averaging log returns
        random_start: bool = False,  # NEW: Random episode start for training diversity
        normalization_stats: dict = None,  # Pre-computed stats from training env (prevents data leakage)
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
        self.macro_df = macro_df
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
        self.reward_type = reward_type
        self.dsr_eta = dsr_eta
        self.log_return_window = log_return_window
        self.random_start = random_start
        self._normalization_stats = normalization_stats  # externally provided stats

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

        # Macro features
        if self.macro_df is not None:
            macro_features = self.macro_df.shape[1] - 1 # subtract date column
            self.validate_macro_alignment()
        else:
            macro_features = 0

        # Total state dimension:  log_returns + weights + tech_features * stocks + macro_features
        self.state_dim = self.stock_dim + self.stock_dim + (base_features_per_stock * self.stock_dim) + macro_features

        # Action space: [0, 1] per asset, normalized to portfolio weights in step()
        # SB3 clips actions to Box bounds before env.step(), guaranteeing non-negative values
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(action_space,), dtype=np.float32)
        
        # Load or compute normalization statistics for observations (RC1)
        # If normalization_stats provided (from training env), use them to prevent data leakage
        if self._normalization_stats is not None:
            self._macro_means = self._normalization_stats.get('macro_means')
            self._macro_stds = self._normalization_stats.get('macro_stds')
            self._tech_means = self._normalization_stats.get('tech_means', {})
            self._tech_stds = self._normalization_stats.get('tech_stds', {})
        else:
            self._precompute_normalization_stats()
        
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
        
        # Initialize portfolio weights - START WITH 100% CASH (0% allocation to all assets)
        if self.initial:
            self.current_weights = np.array([0.0] * self.stock_dim)
        else:
            # Restore state from previous walk-forward block
            if isinstance(self.previous_state, dict):
                # New format from get_terminal_state() — explicit and safe
                self.current_weights = np.array(self.previous_state["current_weights"])
                self.initial_amount = self.previous_state["portfolio_value"]
            else:
                # Legacy format: raw observation array
                # Weights are at positions [stock_dim : 2*stock_dim] in each observation row
                self.current_weights = np.array(self.previous_state[-1][self.stock_dim:2*self.stock_dim])
                # NOTE: Do NOT extract initial_amount from observation[-1][-1] — that field
                # is now relative_performance, not raw portfolio_value. Keep constructor default.

        self.portfolio_value = self.initial_amount

        # Memory containers
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.current_weights.tolist()]
        self.date_memory = []
        
        # NEW: Track transaction costs and turnover
        self.transaction_cost_memory = []
        self.turnover_memory = []
        self.cost_memory = [0]  # Total cumulative costs
        
        # NEW: Historical observation buffer for sequences
        self.observation_buffer = []
        
        # Initialize with first observation
        self._initialize_observation_buffer()

        # NEW: Initialize state for Differential Sharpe Ratio
        self.dsr_a = 0.0
        self.dsr_b = 0.0
        self.last_sharpe = 0.0
        
        # NEW: Initialize log return memory for log_return reward
        self.log_return_memory = []

    def _precompute_normalization_stats(self):
        """Precompute normalization statistics from the training data for stable z-scoring."""
        # Macro feature statistics
        if self.macro_df is not None:
            macro_vals = self.macro_df.iloc[:, 1:].values.astype(float)
            self._macro_means = np.nanmean(macro_vals, axis=0)
            self._macro_stds = np.nanstd(macro_vals, axis=0) + 1e-8
        else:
            self._macro_means = None
            self._macro_stds = None
        
        # Technical indicator statistics (per indicator, across all stocks and dates)
        self._tech_means = {}
        self._tech_stds = {}
        for tech in self.tech_indicator_list:
            if tech in self.df.columns:
                vals = self.df[tech].values.astype(float)
                self._tech_means[tech] = np.nanmean(vals)
                self._tech_stds[tech] = np.nanstd(vals) + 1e-8
            else:
                self._tech_means[tech] = 0.0
                self._tech_stds[tech] = 1.0

    def get_normalization_stats(self) -> dict:
        """Export normalization stats so they can be passed to val/trade environments.
        
        Call this on the training environment, then pass the result as
        normalization_stats= to validation and trade environments to prevent data leakage.
        """
        return {
            'macro_means': self._macro_means,
            'macro_stds': self._macro_stds,
            'tech_means': dict(self._tech_means),
            'tech_stds': dict(self._tech_stds),
        }

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
        Build NORMALIZED observation for a single day (RC1: stationary, zero-mean features).
        
        State Structure: [log_returns, portfolio_weights, z-scored_tech_indicators,
                          z-scored_macro_features]
        
        Key changes from original:
        1. Log-returns instead of raw prices (stationary)
        2. Z-score normalized technical indicators
        3. Z-score normalized macro features
        4. Relative portfolio performance instead of raw portfolio value
        
        Returns:
            np.array: 1D state vector for one timestep
        """
        state = []
        
        # 1. LOG RETURNS instead of raw prices (stationary, ~zero mean)
        if day_idx > 0:
            prev_data = self.df.loc[day_idx - 1, :]
            if len(daily_data.tic.unique()) > 1:
                log_returns = np.log(daily_data.close.values / prev_data.close.values)
                log_returns = np.clip(log_returns, -0.15, 0.15)
                state.extend(log_returns.tolist())
            else:
                lr = np.log(daily_data.close.iloc[0] / prev_data.close.iloc[0])
                state.append(float(np.clip(lr, -0.15, 0.15)))
        else:
            state.extend([0.0] * self.stock_dim)
        
        # 2. Current portfolio weights (already 0-1, no normalization needed)
        state.extend(self.current_weights.tolist())
        
        # 3. Technical indicators — z-score normalized using precomputed stats
        for tech in self.tech_indicator_list:
            if len(daily_data.tic.unique()) > 1:
                vals = daily_data[tech].values.astype(float)
            else:
                vals = np.array([float(daily_data[tech].iloc[0])])
            
            normalized = np.clip(
                (vals - self._tech_means[tech]) / self._tech_stds[tech], -3.0, 3.0
            )
            state.extend(normalized.tolist())

        # 4. Returns (if enabled) - clipped for stability
        if self.include_returns:
            if day_idx > 0:
                prev_data = self.df.loc[day_idx - 1, :]
                if len(daily_data.tic.unique()) > 1:
                    returns = (daily_data.close.values / prev_data.close.values) - 1
                    state.extend(np.clip(returns, -0.15, 0.15).tolist())
                else:
                    r = (daily_data.close.iloc[0] / prev_data.close.iloc[0]) - 1
                    state.append(float(np.clip(r, -0.15, 0.15)))
            else:
                state.extend([0.0] * self.stock_dim)

        # 5. Volume (if enabled) - log-normalized and z-scored
        if self.include_volume:
            if 'volume' in daily_data.columns:
                if len(daily_data.tic.unique()) > 1:
                    volumes = daily_data.volume.values.astype(float)
                    log_volumes = np.log(volumes + 1)
                    mean_v = np.mean(log_volumes)
                    std_v = np.std(log_volumes) + 1e-8
                    normalized_volumes = np.clip((log_volumes - mean_v) / std_v, -3.0, 3.0)
                    state.extend(normalized_volumes.tolist())
                else:
                    state.append(0.0)  # single stock, no cross-sectional normalization
            else:
                state.extend([0.0] * self.stock_dim if len(daily_data.tic.unique()) > 1 else [0.0])

        # 6. Macro features — z-score normalized using precomputed stats
        if self.macro_df is not None:
            if len(daily_data.tic.unique()) > 1:
                current_date = daily_data.date.unique()[0]
            else:
                current_date = daily_data.date.iloc[0]
            
            macro_row = self.macro_df[self.macro_df['date'] == current_date]
            
            if len(macro_row) > 0:
                macro_features = macro_row.iloc[0, 1:].values.flatten().astype(float)
            else:
                available = self.macro_df[self.macro_df['date'] <= current_date]
                if len(available) > 0:
                    macro_features = available.iloc[-1, 1:].values.flatten().astype(float)
                else:
                    macro_features = np.zeros(self.macro_df.shape[1] - 1)
            
            # Z-score with precomputed statistics
            normalized_macro = np.clip(
                (macro_features - self._macro_means) / self._macro_stds, -3.0, 3.0
            )
            state.extend(normalized_macro.tolist())

        if len(state) != self.state_dim:
            print(f"ERROR: State dimension mismatch! Expected {self.state_dim}, got {len(state)}")
        
        return np.array(state, dtype=np.float32)
    
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
        """
        Step function with sequence-aware observations and proper transaction cost accounting.
        
        Transaction Cost Implementation:
        ================================
        
        Financial Model:
        ---------------
        When rebalancing from weights w_old to w_new:
        
        1. At end of day T:
           - Portfolio value = V_T
           - Current weights = w_old (from previous rebalancing)
           
        2. Decide to rebalance to w_new
           - Turnover = sum(|w_new[i] - w_old[i]|)
           - This is the fraction of portfolio that must be traded
           - Example: [0.5, 0.5] -> [0.3, 0.7] has turnover = |0.3-0.5| + |0.7-0.5| = 0.4
           
        3. Pay transaction costs
           - Cost = Turnover × transaction_cost_pct × V_T
           - V_T_after_cost = V_T - Cost
           
        4. Portfolio is now allocated to w_new with value V_T_after_cost
        
        5. Market moves on day T+1
           - Each position earns its return: r[i] = (Price_T+1[i] / Price_T[i]) - 1
           - Portfolio return = sum(w_new[i] × r[i])
           - V_T+1 = V_T_after_cost × (1 + portfolio_return)
           
        Key Insight:
        -----------
        Transaction costs are paid BEFORE market returns. This is the correct
        temporal ordering and prevents "free rebalancing" where the agent can
        change allocations costlessly.
        
        The agent must learn that rebalancing has a cost, creating an optimal
        turnover rate: too little and you miss opportunities, too much and you
        pay excessive costs.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if np.all(actions == 0):
            print("Warning: Actions are all zeros, assigning equal weights.")
            actions = np.array([1.0 / len(actions)] * len(actions))

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
            
            # NEW: Report transaction costs and turnover
            total_transaction_costs = sum(self.transaction_cost_memory)
            avg_turnover = np.mean(self.turnover_memory) if len(self.turnover_memory) > 0 else 0
            print(f"total_transaction_costs: {total_transaction_costs:.2f}")
            print(f"avg_daily_turnover: {avg_turnover:.4f}")
            print(f"num_rebalances: {len(self.turnover_memory)}")

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
            # ================================================================
            # STEP 1: Convert unbounded logits to portfolio weights via softmax (RC2)
            # ================================================================
            new_weights = self._softmax(actions)
            
            # Store old weights before updating
            old_weights = self.current_weights.copy()
            
            # ================================================================
            # STEP 2: Calculate transaction costs for rebalancing
            # ================================================================
            # Turnover is the sum of absolute weight changes
            # This represents the fraction of portfolio that needs to be traded
            turnover = np.sum(np.abs(new_weights - old_weights))
            
            # Transaction cost as a percentage of the traded amount
            # If turnover = 0.5 (50% of portfolio traded) and cost_pct = 0.001 (0.1%)
            # Then total cost = 0.5 * 0.001 * portfolio_value = 0.0005 * portfolio_value
            transaction_cost = turnover * self.transaction_cost_pct * self.portfolio_value
            
            # Apply transaction cost BEFORE calculating returns
            # This is the correct sequence: pay to rebalance, then earn returns
            self.portfolio_value -= transaction_cost
            
            # Track costs and turnover for analysis
            self.transaction_cost_memory.append(transaction_cost)
            self.turnover_memory.append(turnover)
            
            # Update current weights AFTER paying transaction costs
            self.current_weights = new_weights
            self.actions_memory.append(new_weights)
            
            # ================================================================
            # STEP 3: Move to next day and calculate market returns
            # ================================================================
            last_day_memory = self.data
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Calculate portfolio return based on NEW weights
            # The portfolio is now allocated according to new_weights
            # and we observe how it performs with today's price changes
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * new_weights
            )

            assert not np.isnan(portfolio_return), "Portfolio return contains NaN values"
            assert not np.isinf(portfolio_return), "Portfolio return contains Inf values"
            
            # ================================================================
            # STEP 4: Update portfolio value with market returns
            # ================================================================
            # Note: portfolio_value was already reduced by transaction_cost above
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

            # ================================================================
            # STEP 5: Calculate reward based on reward_type
            # ================================================================
            if self.reward_type == "log_return":
                # Logarithmic return reward - Kelly optimal for long-term growth
                if len(self.asset_memory) > 1:
                    # Current log return
                    current_log_return = np.log(self.asset_memory[-1] / self.asset_memory[-2])
                    
                    # Store for averaging
                    self.log_return_memory.append(current_log_return)
                    
                    # Average over window (smooths noise, encourages consistent growth)
                    if len(self.log_return_memory) >= self.log_return_window:
                        # Use exponential moving average for recent bias
                        weights = np.exp(np.linspace(-1, 0, len(self.log_return_memory[-self.log_return_window:])))
                        weights = weights / weights.sum()
                        avg_log_return = np.average(
                            self.log_return_memory[-self.log_return_window:],
                            weights=weights
                        )
                        self.reward = avg_log_return
                    else:
                        # Not enough history yet - use simple average
                        self.reward = np.mean(self.log_return_memory)
                    
                    # Scale up for better gradient magnitude
                    self.reward *= 100  # Convert to percentage-like scale
                else:
                    self.reward = 0.0
            
            elif self.reward_type == "dsr":
                # Differential Sharpe Ratio - rewards improvements in Sharpe ratio
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
            
            elif self.reward_type == "sharpe":
                # Rolling Sharpe ratio
                if len(self.portfolio_return_memory) >= 20:
                    recent_returns = self.portfolio_return_memory[-20:]
                    mean_return = np.mean(recent_returns)
                    std_return = np.std(recent_returns)
                    if std_return != 0:
                        self.reward = (mean_return / std_return) * np.sqrt(252)
                    else:
                        self.reward = 0.0
                else:
                    self.reward = portfolio_return
            
            else:  # "pnl" or default
                # Simple profit and loss
                self.reward = gain
            
            # ================================================================
            # Active return reward (RC3): clear credit assignment vs 1/N
            # ================================================================
            if self.reward_type == "active_return":
                # Equal-weight benchmark return
                price_returns = (self.data.close.values / last_day_memory.close.values) - 1
                benchmark_return = np.mean(price_returns)  # 1/N portfolio
                
                # Active return (excess over benchmark)
                active_return = portfolio_return - benchmark_return
                
                # Turnover penalty: only for excessive rebalancing
                turnover_penalty = 0.0
                if turnover > 0.2:  # Raised from 0.1 — allow moderate rebalancing
                    turnover_penalty = 0.001 * (turnover - 0.2)
                
                # No entropy bonus — PPO's ent_coef handles exploration.
                # Entropy in reward causes convergence to 1/N (equal weights).
                self.reward = (active_return * 1000) - turnover_penalty
            
            assert not np.isnan(self.reward), "Reward contains NaN values"
            assert not np.isinf(self.reward), "Reward contains Inf values"

            # Validate observation
            observation = self._get_observation()
            assert not np.any(np.isnan(observation)), "Observation contains NaN values"
            assert not np.any(np.isinf(observation)), "Observation contains Inf values"

        return observation, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset environment with sequence buffer initialization and optional random start (RC6)."""
        self.asset_memory = [self.initial_amount]
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]

        if self.initial:
            self.current_weights = np.array([0.0] * self.stock_dim)
        else:
            # Restore weights from previous walk-forward block
            if isinstance(self.previous_state, dict):
                self.current_weights = np.array(self.previous_state["current_weights"])
            else:
                self.current_weights = np.array(self.previous_state[-1][self.stock_dim:2*self.stock_dim])

        self.actions_memory = [self.current_weights.tolist()]
        
        self.date_memory = []
        self.observation_buffer = []
        
        # Reset cost and turnover tracking
        self.transaction_cost_memory = []
        self.turnover_memory = []
        self.cost_memory = [0]

        # Reset DSR states
        self.dsr_a = 0.0
        self.dsr_b = 0.0
        self.last_sharpe = 0.0
        
        # Reset log return memory
        self.log_return_memory = []
        
        # RC6: Random start position for training diversity
        n_days = len(self.df.index.unique())
        if self.random_start and n_days > self.sequence_length + 10:
            # Start at a random point after enough history for the sequence buffer
            max_start = n_days - self.sequence_length - 1
            self.day = np.random.randint(self.sequence_length, max(self.sequence_length + 1, max_start))
        else:
            self.day = 0
        
        # Initialize observation buffer
        self._initialize_observation_buffer()
        
        return self._get_observation(), {}

    def render(self, mode="human"):
        """Render current state (returns observation; for walk-forward handoff use get_terminal_state())."""
        return self._get_observation()

    def get_terminal_state(self) -> dict:
        """Return terminal portfolio state for walk-forward handoff.
        
        This provides explicit portfolio_value and weights so the next
        walk-forward block can initialize correctly without depending on
        observation layout (which changed after normalization in RC1).
        """
        return {
            "portfolio_value": self.portfolio_value,
            "current_weights": self.current_weights.copy(),
        }

    def _softmax(self, actions):
        """Normalize non-negative actions to portfolio weights that sum to 1.
        
        With action_space Box(0, 1), SB3 clips actions to [0, 1] before
        env.step(). This gives a linear mapping from policy outputs to
        portfolio weights — no exponential compression, no hyperparameters.
        """
        actions = np.asarray(actions, dtype=np.float64)
        # Safety clip (actions should already be in [0,1] from SB3 clipping)
        actions = np.clip(actions, 0.0, None)
        total = actions.sum()
        if total < 1e-8:
            # All actions near zero → equal weight fallback
            return np.ones_like(actions) / len(actions)
        return actions / total

    def softmax_normalization(self, actions):
        """Normalize portfolio weights (backward compat)."""
        return self._softmax(actions)

    def save_asset_memory(self):
        """Save asset memory to DataFrame with transaction costs and turnover."""
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        
        # Pad transaction costs to match dates (first entry has no rebalancing)
        transaction_costs = [0] + self.transaction_cost_memory
        turnover = [0] + self.turnover_memory
        
        df_account_value = pd.DataFrame({
            "date": date_list, 
            "daily_return": portfolio_return, 
            "account_value": self.asset_memory,
            "transaction_cost": transaction_costs[:len(date_list)],
            "turnover": turnover[:len(date_list)]
        })
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

    def get_sb_env(self, n_envs=1):
        """Get Stable-Baselines3 compatible environment.
        
        Args:
            n_envs: Number of parallel environments. If > 1, uses SubprocVecEnv
                    with random starts for training diversity (RC7).
        """
        if n_envs <= 1:
            e = DummyVecEnv([lambda: self])
            obs = e.reset()
            return e, obs
        else:
            import copy
            from stable_baselines3.common.vec_env import SubprocVecEnv
            
            def make_env_fn(seed_offset):
                def _init():
                    env = copy.deepcopy(self)
                    env.random_start = True
                    env._seed(seed_offset)
                    return env
                return _init
            
            e = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
            obs = e.reset()
            return e, obs

    def validate_macro_alignment(self):
        """Validate that macro_df has perfect alignment with main df."""
        if self.macro_df is None:
            return True
        
        # Check date column exists
        if 'date' not in self.macro_df.columns:
            raise ValueError("macro_df must have a 'date' column")
        
        # Get date ranges and normalize
        main_dates = set(pd.to_datetime(self.df['date'].unique()).date)
        macro_dates = set(pd.to_datetime(self.macro_df['date'].unique()).date)
        
        # Check for perfect coverage
        missing_dates = main_dates - macro_dates
        extra_dates = macro_dates - main_dates
        
        coverage = len(main_dates & macro_dates) / len(main_dates)
        
        print(f"Date coverage: {coverage:.1%} ({len(main_dates & macro_dates)}/{len(main_dates)} days)")
        
        if coverage < 1.0:
            print(f"❌ CRITICAL: Incomplete macro data coverage!")
            if missing_dates:
                print(f"Missing macro dates: {sorted(list(missing_dates))[:10]}{'...' if len(missing_dates) > 10 else ''}")
            raise ValueError(f"Macro data must cover ALL trading dates. Missing {len(missing_dates)} dates.")
        
        if extra_dates:
            print(f"ℹ️  Info: Macro data has {len(extra_dates)} extra dates (will be ignored)")
        
        print("✅ Perfect macro data alignment confirmed")
        return True