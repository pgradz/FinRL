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
        post_norm_tc_coeff: float = 0.0,  # Post-normalization TC penalty: applied AFTER EWMA z-score, before reward_scaling. 0.0 = disabled.
        action_mode: str = "absolute",  # "absolute" = target weights, "residual" = weight deltas (zero action = hold)
        decision_interval: int = 1,  # Act every N trading days (1=daily, 5=weekly). Hold days advance prices but skip action/TC.
        randomize_interval_offset: bool = False,  # Training: randomize phase so agent doesn't overfit to a fixed weekday
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
            post_norm_tc_coeff: Unified TC penalty applied in normalized reward space (after EWMA z-score).
                Set >0 (e.g. 1.0) to penalize TC uniformly across all reward types.
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
        self.post_norm_tc_coeff = post_norm_tc_coeff
        self.action_mode = action_mode
        self.decision_interval = max(1, int(decision_interval))
        self.randomize_interval_offset = randomize_interval_offset
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

        # Action space definition depends on action_mode:
        # - "absolute": Box(0, 1) — actions are target portfolio weights (SB3 clips to non-negative)
        # - "residual": Box(-1, 1) — actions are weight DELTAS from current allocation
        #   Zero action = hold current portfolio = zero turnover = zero TC
        #   This aligns NN zero-initialization with the optimal default (hold),
        #   so the agent must actively push outputs away from zero to trade.
        if self.action_mode == "residual":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_space,), dtype=np.float32)
        else:
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
        
        # Decision interval state
        self._steps_since_decision = 0
        self._interval_offset = 0  # Phase offset (randomized in training)
        
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

    def _transform_reward(self, raw_reward: float, tc_fraction: float = 0.0) -> float:
        """Apply optional reward normalization and global scaling.
        
        Purpose: Bring different reward types (return, Sharpe, DSR, log returns) to
        similar scale for stable policy gradient learning.
        
        Pipeline:
            raw_reward -> (optional EWMA z-score + clipping) -> post-norm TC penalty -> reward_scaling
        
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
            # Even with no normalization, apply post-norm TC penalty if enabled
            reward = raw_reward
            if self.post_norm_tc_coeff > 0:
                reward -= self.post_norm_tc_coeff * tc_fraction
            return reward * self.reward_scaling

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

        # Post-normalization TC penalty: applied in the normalized O(1) reward
        # space so that the same coeff works identically across all reward types.
        # tc_fraction is dimensionless (total_TC / initial_amount, ~0.001 per rebalance).
        if self.post_norm_tc_coeff > 0:
            normalized_reward -= self.post_norm_tc_coeff * tc_fraction

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

        if self.action_mode != "residual" and np.all(actions == 0):
            # In absolute mode, all-zero actions are degenerate (no allocation preference).
            # In residual mode, all-zero actions mean "hold" — the desired default.
            print("Warning: Actions are all zeros, assigning equal weights.")
            actions = np.array([1.0 / len(actions)] * len(actions))

        if self.terminal:
            # Terminal state - save plots and print statistics
            self._save_terminal_stats()
            return self._get_observation(), self.reward, self.terminal, False, {}

        else:
            # ================================================================
            # DECISION INTERVAL: Each call to step() processes one decision
            # followed by (decision_interval - 1) hold days. On hold days
            # the portfolio drifts with market returns but no action/TC.
            # This lets the agent train on daily price data while trading
            # at a lower frequency (e.g., weekly with decision_interval=5).
            # Backward compatible: decision_interval=1 → original daily behavior.
            # ================================================================
            
            # Determine how many days this step covers.
            # First step of an episode may use a shorter interval (phase offset).
            if self._steps_since_decision == 0 and self._interval_offset > 0:
                # First interval after reset: shortened by offset
                interval_length = max(1, self._interval_offset)
            else:
                interval_length = self.decision_interval
            
            # ================================================================
            # STEP 1: Convert actions to portfolio weights (DECISION DAY)
            # ================================================================
            if self.action_mode == "residual":
                proposed_weights = self.current_weights + actions
                new_weights = self._softmax(proposed_weights)
            else:
                new_weights = self._softmax(actions)
            
            old_weights = self.current_weights.copy()
            
            # ================================================================
            # STEP 2: Rebalancing Threshold Check (Execution Layer)
            # ================================================================
            # Capture pre-TC portfolio value for reward computation.
            # Using the pre-TC value makes interval_return TC-inclusive:
            #   return = V_after_market / V_before_TC - 1
            # This preserves the original reward semantics where TC drag is
            # implicitly embedded in the return signal, alongside the explicit
            # TC penalty.  Without this, reward types with low tc_scale (pnl,
            # log_return, active_return) would lose most of their TC pressure.
            interval_start_value = self.portfolio_value
            
            turnover = np.sum(np.abs(new_weights - old_weights))
            
            if turnover < self.rebalancing_threshold:
                new_weights = old_weights.copy()
                turnover = 0.0
                transaction_cost = 0.0
            else:
                transaction_cost = turnover * self.transaction_cost_pct * self.portfolio_value
                self.portfolio_value -= transaction_cost
            
            self.transaction_cost_memory.append(transaction_cost)
            self.turnover_memory.append(turnover)
            
            self.current_weights = new_weights
            self.actions_memory.append(new_weights)
            
            # ================================================================
            # STEP 3–4: Advance through interval days (market returns)
            # ================================================================
            total_transaction_cost = transaction_cost  # accumulate TC across interval
            
            # DSR: accumulate per-day differential Sharpe increments across the interval.
            # This keeps Moody & Saffell (2001) semantics even when decisions are weekly.
            dsr_increment_sum = 0.0
            dsr_increment_count = 0
            
            # Track daily price returns for benchmark computation
            interval_price_returns = []
            
            for day_in_interval in range(interval_length):
                # Advance to next calendar day
                last_day_memory = self.data
                self.day += 1
                
                # Safety: don't advance past available data
                n_unique = len(self.df.index.unique())
                if self.day >= n_unique:
                    self.day = n_unique - 1
                    self.terminal = True
                    break
                
                self.data = self.df.loc[self.day, :]
                
                # Calculate market returns with current weights
                price_returns = (self.data.close.values / last_day_memory.close.values) - 1
                portfolio_return = sum(price_returns * self.current_weights)
                
                # Store price returns for benchmark computation
                interval_price_returns.append(price_returns)
                
                assert not np.isnan(portfolio_return), "Portfolio return contains NaN values"
                assert not np.isinf(portfolio_return), "Portfolio return contains Inf values"
                
                new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
                self.portfolio_value = new_portfolio_value
                
                # Update observation buffer
                new_obs = self._build_daily_observation(self.data, self.day)
                self.observation_buffer.append(new_obs)
                self.observation_buffer = self.observation_buffer[-self.sequence_length:]
                
                # Save daily records to memory
                self.portfolio_return_memory.append(portfolio_return)
                self.date_memory.append(self.data.date.unique()[0])
                self.asset_memory.append(new_portfolio_value)
                
                # Update DSR statistics DAILY (not just on decision days)
                # This maintains proper 20-day memory for risk estimation
                # even when making decisions weekly (decision_interval=5)
                if self.reward_type == "dsr" and len(self.asset_memory) > 1:
                    prev_a = self.dsr_a
                    prev_b = self.dsr_b
                    prev_var = prev_b - prev_a ** 2

                    # Use TC-inclusive net return on the decision day so DSR
                    # naturally penalises excessive turnover without an arbitrary
                    # tc_scale hyperparameter.  On hold days (no TC), this equals
                    # the gross market return.
                    #   net_ret = V_after_market / V_before_TC - 1
                    #           = (post_TC * (1+r)) / pre_TC - 1
                    if day_in_interval == 0 and interval_start_value > 0:
                        dsr_return = (new_portfolio_value / interval_start_value) - 1
                    else:
                        dsr_return = portfolio_return

                    # --- ORIGINAL (gross market return, TC-blind): ---
                    # dsr_return = portfolio_return

                    # EWMA moment updates
                    self.dsr_a = (1 - self.dsr_eta) * prev_a + self.dsr_eta * dsr_return
                    self.dsr_b = (1 - self.dsr_eta) * prev_b + self.dsr_eta * (dsr_return ** 2)

                    # Differential Sharpe increment (Moody & Saffell 2001)
                    if prev_var > 1e-12:
                        delta_a = self.dsr_a - prev_a
                        delta_b = self.dsr_b - prev_b
                        dsr_inc = (prev_b * delta_a - 0.5 * prev_a * delta_b) / (prev_var ** 1.5)
                        dsr_increment_sum += float(dsr_inc)
                        dsr_increment_count += 1
                
                # Drift weights after market returns — applies to ALL days in interval.
                # After price changes, portfolio composition shifts naturally:
                #   w_i' = w_i * (1 + r_i) / sum(w_j * (1 + r_j))
                # This must happen on the decision day too, so that hold day 1
                # starts with correctly drifted weights.
                drifted = self.current_weights * (1 + price_returns)
                weight_sum = drifted.sum()
                if weight_sum > 1e-10:
                    self.current_weights = drifted / weight_sum
                
                # Record hold-day memory only (decision day already recorded above)
                if day_in_interval > 0:
                    self.transaction_cost_memory.append(0.0)
                    self.turnover_memory.append(0.0)
                    self.actions_memory.append(self.current_weights.tolist())
                
                # Check terminal AFTER computing this day's return
                self.terminal = self.day >= n_unique - 1
                if self.terminal:
                    break
            
            self._steps_since_decision += 1
            
            # ================================================================
            # STEP 5: Calculate reward for the full interval
            # ================================================================
            # Use interval-level return for reward computation.
            # This gives the agent one reward signal per decision, reflecting
            # the cumulative consequence of that decision over the holding period.
            
            # Transaction costs are embedded in portfolio_value for ALL reward
            # types: interval_return and asset_memory naturally reflect TC drag.
            # DSR additionally uses TC-inclusive net returns in its EWMA moments
            # (see decision-day branch above), so no explicit penalty is needed.
            explicit_tc_penalty = 0.0

            # --- ORIGINAL (arbitrary tc_scale explicit penalty): ---
            # TC_PENALTY_SCALES = {
            #     'dsr': 15.0,
            #     'sharpe': 100.0,
            # }
            # tc_scale = TC_PENALTY_SCALES.get(self.reward_type, 0.0)
            # if self.reward_type in ("dsr", "sharpe"):
            #     explicit_tc_penalty = (total_transaction_cost / self.initial_amount) * tc_scale
            # else:
            #     explicit_tc_penalty = 0.0
            
            turnover_penalty = 0.0
            if turnover > self.turnover_penalty_threshold:
                turnover_penalty = self.turnover_penalty_coeff * (turnover - self.turnover_penalty_threshold)
            
            # Interval-level return (includes TC from decision day)
            interval_return = (self.portfolio_value / (interval_start_value + 1e-10)) - 1
            
            if self.reward_type == "log_return":
                if len(self.asset_memory) > 1:
                    raw_reward = np.log(max(self.portfolio_value / (interval_start_value + 1e-10), 1e-10))
                    self.log_return_memory.append(raw_reward)
                else:
                    raw_reward = 0.0
            
            elif self.reward_type == "dsr":
                # Differential Sharpe Ratio (Moody & Saffell, 2001)
                # Accumulated daily DSR increments across the interval.
                if dsr_increment_count > 0:
                    # Average to keep reward magnitude stable when interval_length changes.
                    raw_reward = dsr_increment_sum / dsr_increment_count
                else:
                    raw_reward = 0.0
            
            elif self.reward_type == "active_return":
                # Active return = agent return - equal-weight benchmark return
                # Benchmark: equal-weight portfolio of all assets
                if len(self.asset_memory) > 1 and len(interval_price_returns) > 0:
                    # Compute equal-weight benchmark return over the interval
                    # Option 1: Compound daily returns (more accurate)
                    benchmark_value = 1.0
                    equal_weights = np.ones(self.stock_dim) / self.stock_dim
                    for daily_returns in interval_price_returns:
                        benchmark_daily_return = np.sum(daily_returns * equal_weights)
                        benchmark_value *= (1 + benchmark_daily_return)
                    benchmark_return = benchmark_value - 1.0
                    
                    # Active return = agent interval return - benchmark interval return
                    raw_reward = interval_return - benchmark_return
                else:
                    raw_reward = 0.0
            
            elif self.reward_type == "sharpe":
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
                    raw_reward = None
            
            elif self.reward_type == "pnl":
                if len(self.asset_memory) > 1:
                    raw_reward = interval_return
                else:
                    raw_reward = 0.0
            
            else:
                raise ValueError(f"Unknown reward_type: {self.reward_type}")

            # Handle penalties and reward transformation
            if raw_reward is None:
                self.reward = 0.0
                self.raw_reward_memory.append(0.0)
                self.scaled_reward_memory.append(0.0)
            else:
                raw_reward = raw_reward - explicit_tc_penalty - turnover_penalty
                # Pass TC fraction so _transform_reward can apply post-norm penalty
                tc_fraction = total_transaction_cost / (self.initial_amount + 1e-10)
                transformed_reward = self._transform_reward(raw_reward, tc_fraction=tc_fraction)
                self.reward = transformed_reward
                self.raw_reward_memory.append(float(raw_reward))
                self.scaled_reward_memory.append(float(self.reward))
            
            assert not np.isnan(self.reward), "Reward contains NaN values"
            assert not np.isinf(self.reward), "Reward contains Inf values"

            # If terminal was detected during interval loop, save stats
            if self.terminal:
                self._save_terminal_stats()

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
        
        # Decision interval: randomize phase offset during training
        # so agent doesn't overfit to a fixed weekday pattern.
        # First step is always a decision day (deploy from cash / evaluate after handoff).
        # Offset shifts SUBSEQUENT decisions: next decision at offset, then every interval after.
        if self.randomize_interval_offset and self.decision_interval > 1:
            self._interval_offset = np.random.randint(0, self.decision_interval)
        else:
            self._interval_offset = 0
        self._steps_since_decision = 0
        
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

    def _save_terminal_stats(self):
        """Save terminal statistics, plots, and CSVs.
        
        Called when the episode ends — either from the terminal branch at the
        top of step() or when terminal is detected inside the decision interval loop.
        """
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

    def _softmax(self, actions):
        """Project a vector onto the probability simplex (non-negative, sum=1).
        
        In absolute mode: input is raw actions from Box(0,1).
        In residual mode: input is (current_weights + deltas), which may
        contain negative entries — these are clipped to 0 before normalizing.
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