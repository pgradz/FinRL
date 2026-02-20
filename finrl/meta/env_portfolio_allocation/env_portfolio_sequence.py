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

# Reward-type-specific initial variance defaults for EWMA normalization
# These values are chosen to match typical empirical variance for each reward type:
# - pnl/log_return: daily returns ~ O(10⁻³), variance ~ O(10⁻⁴) to O(10⁻⁶)
# - active_return: excess returns ~ O(10⁻⁴), variance ~ O(10⁻⁵) to O(10⁻⁶)
# - dsr: dimensionless ~ O(0.1-1), variance ~ O(0.01) to O(0.1)
# - sharpe: dimensionless ~ O(0.1-2), variance ~ O(0.1) to O(1)
REWARD_VAR_DEFAULTS = {
    'pnl': 1e-4,
    'log_return': 1e-4,
    'active_return': 1e-5,
    'dsr': 0.01,
    'sharpe': 0.1
}


class StockPortfolioSequenceEnv(gym.Env):
    """
    A sequence-aware portfolio allocation environment for temporal models (LSTM, CNN, Transformer, CNN-LSTM).
    
    Key Changes from Original:
    1. Maintains a rolling window of historical observations
    2. Observation space is 3D: (sequence_length, features, stocks) or flattened to 2D
    3. Supports both 2D and 1D observation formats for different models
    4. Backward compatible with existing models via flatten_observations parameter    
    Reward Functions:
    - pnl: Day-over-day portfolio percentage return (scale-invariant, includes TC)
    - log_return: Per-step portfolio log return (adapted from Jiang et al., 2017 - arXiv:1706.10059)
    - dsr: Differential Sharpe Ratio (Moody & Saffell, 2001)
    - sharpe: Rolling Sharpe ratio (annualized)
    - active_return: Excess return over equal-weight benchmark    """

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
        dsr_eta: float = 2.0 / 21,  # Moody & Saffell (2001) decay = 2/(T+1), T=20
        reward_type: str = "pnl",  # Reward function type: 'pnl', 'sharpe', 'dsr', 'log_return', or 'active_return'
        log_return_window: int = 20,  # Retained for memory tracking; no longer used for windowed averaging
        sharpe_window: int = 21,  # Rolling window for Sharpe ratio (days). Keep ≈ state horizon for credit assignment
        reward_transform: str = "ewma_zscore",  # Reward normalization: 'none' | 'ewma_zscore'
        reward_beta: float = 0.05,  # EWMA update rate (0.05 ≈ 20-day halflife, balances responsiveness vs stability)
        reward_clip: float = 3.0,  # Clip normalized rewards to ±3 std (keeps 99.7% of normal distribution)
        reward_stats: dict = None,  # Optional precomputed reward stats from training env
        update_reward_stats: bool = True,  # If False, use fixed reward_stats without updating
        random_start: bool = False,  # Random episode start for training diversity
        normalization_stats: dict = None,  # Pre-computed stats from training env (prevents data leakage)
        rebalancing_threshold: float = 0.0,  # Minimum turnover required to execute rebalancing (default: 0.0 = always rebalance)
        turnover_penalty_threshold: float = 0.20,  # Penalty-free daily turnover (~10% reallocation)
        turnover_penalty_coeff: float = 0.0,  # Set to 0.0 to disable (default); use >0 to penalize excessive turnover
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
            dsr_eta: Learning rate for Differential Sharpe Ratio (Moody & Saffell, 2001)
            reward_type: Type of reward function ('pnl', 'dsr', 'sharpe', 'log_return', 'active_return')
            log_return_window: Window for averaging log returns
            reward_transform: Normalization strategy ('none' or 'ewma_zscore')
            reward_beta: EWMA update rate for reward normalization
            reward_clip: Clipping range for normalized rewards (in std devs)
            reward_stats: Pre-computed reward stats from training env (prevents data leakage)
            update_reward_stats: Whether to update reward stats online
            random_start: Random episode start for training diversity
            normalization_stats: Pre-computed normalization stats from training
            rebalancing_threshold: Minimum turnover required to execute rebalancing (default: 0.0 = always rebalance)
            turnover_penalty_threshold: Turnover threshold before penalty applies (default: 0.20)
            turnover_penalty_coeff: Penalty coefficient (default: 0.0 = disabled)
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
        self.sharpe_window = sharpe_window
        self.reward_transform = reward_transform
        self.reward_beta = reward_beta
        self.reward_clip = reward_clip
        self.update_reward_stats = update_reward_stats
        self.random_start = random_start
        self.rebalancing_threshold = rebalancing_threshold
        self.turnover_penalty_threshold = turnover_penalty_threshold
        self.turnover_penalty_coeff = turnover_penalty_coeff
        self._normalization_stats = normalization_stats  # externally provided stats

        # Reward normalization state (shared transform layer across reward types)
        # Initial variance is reward-type-specific to avoid scale mismatches:
        # - pnl/log_return use 1e-4 (typical daily return variance)
        # - dsr/sharpe use larger values (0.01-0.1) to match their O(0.1-1) scale
        if reward_stats is None:
            self._reward_mean = 0.0
            self._reward_var = REWARD_VAR_DEFAULTS.get(reward_type, 1e-4)
        else:
            self._reward_mean = float(reward_stats.get("mean", 0.0))
            self._reward_var = float(reward_stats.get("var", 1e-4))

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
        self.raw_reward_memory = []
        self.scaled_reward_memory = []
        
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
        
        # Initialize log return memory for log_return reward
        self.log_return_memory = []

        # Initialize reward value
        self.reward = 0.0

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

    def get_reward_stats(self) -> dict:
        """Export reward normalization stats for val/trade environments."""
        return {
            "mean": float(self._reward_mean),
            "var": float(self._reward_var),
        }

    def _transform_reward(self, raw_reward: float) -> float:
        """Apply optional reward normalization and global scaling.
        
        Purpose: Bring different reward types (return, Sharpe, DSR, log returns) to
        similar scale for stable policy gradient learning.
        
        Pipeline:
            raw_reward -> (optional EWMA z-score + clipping) -> reward_scaling
        
        Hyperparameter Guidance:
        ------------------------
        reward_beta (EWMA update rate):
            - 0.01 = ~100 day halflife (very smooth, slow adaptation)
            - 0.05 = ~20 day halflife (balanced, RECOMMENDED)
            - 0.10 = ~10 day halflife (responsive, may overfit to recent data)
        
        reward_clip (std deviations):
            - 2.0 = keeps 95% of normal distribution (aggressive clipping)
            - 3.0 = keeps 99.7% of normal distribution (RECOMMENDED)
            - 5.0 = keeps 99.9994% (minimal clipping)
        
        For minimal ablation: test {beta=0.05, clip=3.0} vs {transform='none'}
        """
        raw_reward = float(raw_reward)

        if self.reward_transform == "none":
            return raw_reward * self.reward_scaling

        if self.reward_transform != "ewma_zscore":
            raise ValueError(f"Unknown reward_transform: {self.reward_transform}")

        # Capture mean BEFORE update for consistent z-scoring.
        # δ is computed against the old mean, so the z-score must also
        # centre on the old mean; using the *updated* mean would shrink
        # every z-score by a factor of (1-β) ≈ 0.95.
        centering_mean = self._reward_mean

        if self.update_reward_stats:
            beta = self.reward_beta
            delta = raw_reward - self._reward_mean
            self._reward_mean = (1 - beta) * self._reward_mean + beta * raw_reward
            # Standard EWMA variance update:
            #   var_t = (1-β) * var_{t-1} + β * δ²
            # where δ = x_t - μ_{t-1} (deviation from OLD mean).
            # This is the conventional exponentially weighted moving average of
            # squared deviations, ensuring numerical stability and consistency
            # with the mean update.
            self._reward_var = (1 - beta) * self._reward_var + beta * (delta ** 2)

        reward_std = np.sqrt(max(self._reward_var, 1e-8))
        normalized_reward = (raw_reward - centering_mean) / reward_std
        normalized_reward = float(np.clip(normalized_reward, -self.reward_clip, self.reward_clip))
        return normalized_reward * self.reward_scaling

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
            # STEP 2: Rebalancing Threshold Check (Execution Layer)
            # ================================================================
            # Calculate turnover BEFORE deciding whether to rebalance
            # This represents the fraction of portfolio that needs to be traded
            turnover = np.sum(np.abs(new_weights - old_weights))
            
            # Check if turnover exceeds threshold - if not, skip rebalancing entirely
            if turnover < self.rebalancing_threshold:
                # Skip rebalancing: keep old weights, pay no transaction costs
                new_weights = old_weights.copy()
                turnover = 0.0
                transaction_cost = 0.0
            else:
                # Execute rebalancing: calculate and pay transaction costs
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
            
            # Update current weights
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
            # STEP 5: Calculate raw reward metric based on reward_type
            # ================================================================
            # Reward scaling is applied through a shared transform layer
            # (EWMA z-score + clipping + reward_scaling) for all reward types.
            # ================================================================
            
            # Optional turnover penalty (raw-reward space)
            # Keep in the same unit space as raw_reward, then normalize once.
            turnover_penalty = 0.0
            if turnover > self.turnover_penalty_threshold:
                turnover_penalty = self.turnover_penalty_coeff * (turnover - self.turnover_penalty_threshold)
            
            if self.reward_type == "log_return":
                # Per-step portfolio log return adapted from Jiang et al. (2017,
                # arXiv:1706.10059, Eq. 11).
                #
                # Original (batch):  R = (1/T) Σ_t ln(μ_t · y_t · w_{t-1})
                #   y_t  = price relative vector  (close_t / close_{t-1})
                #   w    = portfolio weight vector chosen at end of t-1
                #   μ_t  = transaction remainder factor ≈ 1 - c_p · turnover
                #
                # Adaptation for online (step-by-step) RL:
                #   - Emit the per-step log return  ln(μ_t · y_t · w_{t-1})
                #     as the immediate reward.  Maximising the discounted sum
                #     of per-step log returns is equivalent to maximising the
                #     log of terminal wealth (Kelly/growth-optimal criterion).
                #   - Variance reduction is delegated to _transform_reward
                #     (EWMA z-score) and PPO's GAE, rather than averaging
                #     over a rolling window which dilutes credit assignment.
                #
                # Caveat vs. original paper:
                #   Jiang et al. optimise the *episode-level* sum with a
                #   deterministic policy gradient (EIIE).  Here we emit
                #   per-step rewards for compatibility with SB3's online
                #   algorithms (PPO, DDPG, etc.).  The objective is
                #   mathematically equivalent when γ = 1, but in practice
                #   γ < 1 down-weights distant future returns.
                if len(self.asset_memory) > 1:
                    # Per-step log return: ln(V_t / V_{t-1})
                    # Computed directly from portfolio value ratio, which
                    # already incorporates TC (subtracted in Step 2).
                    # This is numerically identical to the decomposed form
                    #   ln(μ_t · dot(y_t, w))  where μ_t = 1 - c·turnover
                    # but avoids any floating-point divergence between paths
                    # and eliminates the appearance of double-counting TC.
                    ratio = self.asset_memory[-1] / (self.asset_memory[-2] + 1e-10)
                    raw_reward = np.log(max(ratio, 1e-10))  # Guard against log(0) or log(negative)
                    
                    self.log_return_memory.append(raw_reward)
                else:
                    raw_reward = 0.0
            
            elif self.reward_type == "dsr":
                # Differential Sharpe Ratio — Moody & Saffell (2001), Eq. 9:
                #   DSR_t = (B_{t-1} · ΔA_t  -  0.5 · A_{t-1} · ΔB_t)
                #           / (B_{t-1} - A_{t-1}²)^{3/2}
                #
                # A_t = EMA of returns,  B_t = EMA of squared returns.
                # The formula uses the PREVIOUS moments (t-1) because DSR is
                # the derivative of the Sharpe ratio w.r.t. the new return,
                # evaluated at the state *before* observing R_t.
                if len(self.asset_memory) > 1:
                    current_return = (self.asset_memory[-1] / (self.asset_memory[-2] + 1e-10)) - 1
                    
                    prev_a = self.dsr_a
                    prev_b = self.dsr_b
                    
                    # Update exponential moving averages
                    self.dsr_a = (1 - self.dsr_eta) * prev_a + self.dsr_eta * current_return
                    self.dsr_b = (1 - self.dsr_eta) * prev_b + self.dsr_eta * (current_return ** 2)
                    
                    # Calculate deltas
                    delta_a = self.dsr_a - prev_a  # = eta * (R_t - prev_a)
                    delta_b = self.dsr_b - prev_b  # = eta * (R_t² - prev_b)
                    
                    # Variance from PREVIOUS moments (Moody & Saffell 2001, Eq. 9)
                    variance = prev_b - prev_a ** 2
                    
                    if variance > 1e-12:
                        dsr = (prev_b * delta_a - 0.5 * prev_a * delta_b) / (variance ** 1.5)
                        raw_reward = dsr
                    else:
                        # Warmup: variance not yet reliable. Emit 0 to avoid scale
                        # discontinuity (DSR ~ O(0.1-1) vs current_return ~ O(10⁻³)).
                        # The EWMA normalization will handle the warmup period gracefully.
                        raw_reward = 0.0
                else:
                    raw_reward = 0.0
            
            elif self.reward_type == "active_return":
                # Excess return over equal-weight benchmark
                # 
                # Benchmark Strategy: 1/N equal-weight allocation
                # Default: Benchmark is GROSS (no TC), Agent is NET (includes TC)
                # Rationale: Equal-weight indices typically rebalance infrequently
                # (monthly/quarterly), so daily TC is negligible. This follows
                # industry standard for evaluating active management.
                # 
                # For symmetric comparison, benchmark TC can be calculated as:
                # Equal-weight drift due to price changes causes weight deviations,
                # rebalancing back to 1/N incurs turnover costs.
                price_returns = (self.data.close.values / np.maximum(last_day_memory.close.values, 1e-10)) - 1
                benchmark_gross_return = np.mean(price_returns)
                
                # Optional: Calculate benchmark transaction costs if rebalancing to equal weights
                # Uncomment below to include benchmark TC for symmetric comparison:
                # if len(self.asset_memory) > 1:
                #     # After price changes, weights drift from equal-weight
                #     benchmark_weights_after_drift = np.ones(self.stock_dim) / self.stock_dim
                #     benchmark_weights_after_drift *= (1 + price_returns)
                #     benchmark_weights_after_drift /= benchmark_weights_after_drift.sum()
                #     # Turnover to rebalance back to equal weights
                #     target_weights = np.ones(self.stock_dim) / self.stock_dim
                #     benchmark_turnover = np.sum(np.abs(target_weights - benchmark_weights_after_drift))
                #     benchmark_tc = benchmark_turnover * self.transaction_cost_pct
                #     benchmark_net_return = benchmark_gross_return - benchmark_tc
                # else:
                #     benchmark_net_return = benchmark_gross_return
                
                # Use agent NET return (TC already applied to portfolio_value)
                agent_return = (self.asset_memory[-1] / (self.asset_memory[-2] + 1e-10)) - 1
                
                # Active return: Agent NET vs Benchmark GROSS (default, conservative)
                active_return = agent_return - benchmark_gross_return
                # Alternatively, for symmetric comparison: agent_return - benchmark_net_return
                
                raw_reward = active_return
            
            elif self.reward_type == "sharpe":
                # Rolling Sharpe ratio using NET returns (TC-adjusted)
                #
                # Window size (sharpe_window, default 21 ≈ 1 month):
                #   - Balances statistical reliability vs. credit assignment
                #   - Current action contributes ~1/window to the signal
                #   - Consider DSR for sharper per-step credit assignment
                #
                # Design choices:
                #   - ddof=1 for unbiased std estimate on small windows
                #   - No √252 annualization: _transform_reward handles scaling;
                #     annualisation inflates magnitude without adding info
                #   - Warmup emits 0 to avoid scale discontinuity (warmup
                #     net-return ≈ O(0.001) vs Sharpe ≈ O(0.5–2) is a ~1000×
                #     jump that destabilises the EWMA normaliser)
                if len(self.asset_memory) >= self.sharpe_window:
                    recent_values = self.asset_memory[-self.sharpe_window:]
                    recent_net_returns = [
                        recent_values[i] / (recent_values[i - 1] + 1e-10) - 1
                        for i in range(1, len(recent_values))
                    ]
                    mean_return = np.mean(recent_net_returns)
                    std_return = np.std(recent_net_returns, ddof=1)
                    if std_return > 1e-12:
                        raw_reward = mean_return / std_return
                    else:
                        raw_reward = 0.0
                else:
                    # Warmup: not enough history for reliable Sharpe estimate.
                    # Use None to signal that normalization should be skipped
                    # (avoids corrupting EWMA stats with zeros during warmup).
                    raw_reward = None
            
            elif self.reward_type == "pnl":
                # Day-over-day portfolio percentage return (scale-invariant,
                # includes TC drag via portfolio_value adjustment in step 2).
                # NOTE: Named 'pnl' for backward compatibility; the actual
                # quantity is (V_t / V_{t-1}) - 1, not dollar PnL.
                if len(self.asset_memory) > 1:
                    raw_reward = (self.asset_memory[-1] / (self.asset_memory[-2] + 1e-10)) - 1
                else:
                    raw_reward = 0.0
            
            else:
                raise ValueError(f"Unknown reward_type: {self.reward_type}")

            # Handle turnover penalty and reward transformation
            # Special case: Sharpe warmup returns None to skip normalization
            if raw_reward is None:
                # Warmup period for Sharpe — not enough history for reliable estimate.
                # Emit 0 reward: avoids corrupting EWMA stats (which would adapt to
                # return-scale ~10⁻³, then face ~1000× scale jump when real Sharpe
                # values ~O(1) begin flowing). Cost: agent learns nothing for ~21
                # steps per episode (negligible for multi-year episodes).
                self.reward = 0.0
                self.raw_reward_memory.append(0.0)
                self.scaled_reward_memory.append(0.0)
            else:
                raw_reward = raw_reward - turnover_penalty
                transformed_reward = self._transform_reward(raw_reward)
                self.reward = transformed_reward
                self.raw_reward_memory.append(float(raw_reward))
                self.scaled_reward_memory.append(float(self.reward))
            
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
        self.raw_reward_memory = []
        self.scaled_reward_memory = []
        
        self.date_memory = []
        self.observation_buffer = []
        
        # Reset cost and turnover tracking
        self.transaction_cost_memory = []
        self.turnover_memory = []
        self.cost_memory = [0]

        # Reset DSR states
        self.dsr_a = 0.0
        self.dsr_b = 0.0
        
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