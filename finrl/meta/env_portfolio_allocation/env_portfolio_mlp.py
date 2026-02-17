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


class StockPortfolioMLPEnv(gym.Env):
    """
    A single-timestep portfolio allocation environment for MLP models.
    
    This environment is aligned with StockPortfolioSequenceEnv but WITHOUT temporal sequences.
    Perfect for benchmarking MLP vs sequence models (LSTM, CNN, Transformer, CNN-LSTM).
    
    Key Features:
    1. Single-timestep observations (no rolling window)
    2. Same state structure as sequence environment
    3. Support for macro features
    4. Multiple reward types (PnL, DSR, Sharpe)
    5. Proper transaction cost accounting
    6. NO covariance matrix (unlike original env_portfolio.py)
    
    This creates a fair comparison: MLP vs sequence models with identical features.
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
        initial=True,
        previous_state=[],
        include_returns=False,  # Include previous day returns in state
        include_volume=False,  # Include volume data
        dsr_eta: float = 0.1,  # Update rate for Differential Sharpe Ratio
        reward_type: str = "pnl",  # Reward function type: 'pnl', 'sharpe', 'dsr', or 'log_return'
        log_return_window: int = 20,  # Window for averaging log returns
        model_name="",
        mode="",
        iteration="",
        seed=""
    ):
        """
        Initialize the MLP portfolio environment.
        
        State Structure (aligned with sequence env):
        [stock_prices, portfolio_weights, tech_indicators, returns?, volume?, macro_features, portfolio_value]
        
        Args:
            df: DataFrame with stock data
            macro_df: DataFrame with macro-economic indicators (optional)
            stock_dim: Number of stocks
            hmax: Maximum shares to trade (legacy, not used in portfolio allocation)
            initial_amount: Starting portfolio value
            transaction_cost_pct: Transaction cost as percentage (e.g., 0.001 = 0.1%)
            reward_scaling: Scaling factor for rewards
            action_space: Dimension of action space (should equal stock_dim)
            tech_indicator_list: List of technical indicator names
            turbulence_threshold: Threshold for risk aversion (optional)
            lookback: Lookback period for calculations
            day: Starting day index
            initial: Whether this is initial state
            previous_state: State from previous episode (for continuation)
            include_returns: Whether to include previous day returns
            include_volume: Whether to include volume information
            dsr_eta: Learning rate for Differential Sharpe Ratio
            reward_type: Type of reward function ('pnl', 'dsr', 'sharpe')
            model_name: Name of model (for logging)
            mode: Mode of operation (for logging)
            iteration: Iteration number (for logging)
            seed: Random seed (for logging)
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
        
        # Feature flags (aligned with sequence env)
        self.include_returns = include_returns
        self.include_volume = include_volume
        self.reward_type = reward_type
        self.dsr_eta = dsr_eta
        self.log_return_window = log_return_window

        # For multiple runs
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.seed = seed
        
        # Calculate state dimensions (same as sequence env but without sequence dimension)
        # State: [prices, weights, tech_indicators, returns?, volume?, macro_features, portfolio_value]
        base_features_per_stock = len(self.tech_indicator_list)
        if self.include_returns:
            base_features_per_stock += 1
        if self.include_volume:
            base_features_per_stock += 1

        # Macro features
        if self.macro_df is not None:
            macro_features = self.macro_df.shape[1] - 1  # subtract date column
            self.validate_macro_alignment()
        else:
            macro_features = 0

        # Total state dimension: prices + weights + tech_features * stocks + portfolio_value + macro_features
        self.state_dim = self.stock_dim + self.stock_dim + (base_features_per_stock * self.stock_dim) + 1 + macro_features

        # Action space: portfolio weights (softmax normalized)
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))
        
        # Observation space: single timestep (1D vector)
        # This is the key difference from sequence env - no sequence dimension!
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,)
        )
        
        # Initialize environment
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        
        # Initialize portfolio weights - START WITH 100% CASH (0% allocation to all assets)
        if self.initial:
            self.current_weights = np.array([0.0] * self.stock_dim)
        else:
            # Extract weights from previous state
            # State structure: [prices, weights, tech_indicators, ...]
            self.current_weights = np.array(self.previous_state[self.stock_dim:2*self.stock_dim])
            self.initial_amount = self.previous_state[-1]

        self.portfolio_value = self.initial_amount

        # Memory containers
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.current_weights.tolist()]
        self.date_memory = []
        
        # Track transaction costs and turnover
        self.transaction_cost_memory = []
        self.turnover_memory = []
        self.cost_memory = [0]  # Total cumulative costs
        
        # Initialize with first observation
        self.data = self.df.loc[self.day, :]
        self.date_memory = [self.data.date.unique()[0]]

        # Initialize state for Differential Sharpe Ratio
        self.dsr_a = 0.0
        self.dsr_b = 0.0
        self.last_sharpe = 0.0
        
        # Initialize log return memory for log_return reward
        self.log_return_memory = []

    def _build_observation(self):
        """
        Build observation for current day (aligned with sequence env).
        
        State Structure: [stock_prices, portfolio_weights, tech_indicators, returns?, volume?, macro_features, portfolio_value]
        
        Returns:
            np.array: 1D state vector for current timestep
        """
        state = []
        
        # 1. Current stock prices (market state)
        if len(self.data.tic.unique()) > 1:
            state.extend(self.data.close.values.tolist())
        else:
            state.append(self.data.close.iloc[0])
        
        # 2. Current portfolio weights (allocation state)
        state.extend(self.current_weights.tolist())
        
        # 3. Technical indicators for all stocks (market sentiment/momentum)
        for tech in self.tech_indicator_list:
            if len(self.data.tic.unique()) > 1:
                state.extend(self.data[tech].values.tolist())
            else:
                state.append(self.data[tech].iloc[0])

        # 4. Returns (if enabled) - market performance
        if self.include_returns:
            if self.day > 0:
                prev_data = self.df.loc[self.day - 1, :]
                if len(self.data.tic.unique()) > 1:
                    returns = (self.data.close.values / prev_data.close.values) - 1
                    state.extend(returns.tolist())
                else:
                    returns = (self.data.close.iloc[0] / prev_data.close.iloc[0]) - 1
                    state.append(returns)
            else:
                # First day: zero returns
                if len(self.data.tic.unique()) > 1:
                    state.extend([0.0] * self.stock_dim)
                else:
                    state.append(0.0)

        # 5. Volume (if enabled and available) - market liquidity
        if self.include_volume:
            if 'volume' in self.data.columns:
                if len(self.data.tic.unique()) > 1:
                    volumes = self.data.volume.values
                    # Normalize volume to prevent scale issues
                    normalized_volumes = np.log(volumes + 1)
                    state.extend(normalized_volumes.tolist())
                else:
                    volume = self.data.volume.iloc[0]
                    normalized_volume = np.log(volume + 1)
                    state.append(normalized_volume)
            else:
                # If volume not available, use zeros
                if len(self.data.tic.unique()) > 1:
                    state.extend([0.0] * self.stock_dim)
                else:
                    state.append(0.0)

        # 6. Macro features (if available) - economic context
        if self.macro_df is not None:
            # Get the actual date from the current data
            if len(self.data.tic.unique()) > 1:
                current_date = self.data.date.unique()[0]
            else:
                current_date = self.data.date.iloc[0]
            
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

        # 7. Current portfolio value - has to be on the last place
        state.append(self.portfolio_value)

        if len(state) != self.state_dim:
            print(f"ERROR: State dimension mismatch! Expected {self.state_dim}, got {len(state)}")
        
        return np.array(state)

    def step(self, actions):
        """
        Step function with proper transaction cost accounting (aligned with sequence env).
        
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
            
            # Report transaction costs and turnover
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

            return self._build_observation(), self.reward, self.terminal, False, {}

        else:
            # ================================================================
            # STEP 1: Normalize actions to get desired new portfolio weights
            # ================================================================
            total_weight = np.sum(actions)
            new_weights = actions / total_weight
            
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
            
            assert not np.isnan(self.reward), "Reward contains NaN values"
            assert not np.isinf(self.reward), "Reward contains Inf values"

            # Build current observation
            observation = self._build_observation()
            assert not np.any(np.isnan(observation)), "Observation contains NaN values"
            assert not np.any(np.isinf(observation)), "Observation contains Inf values"

        return observation, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]

        if self.initial:
            self.current_weights = np.array([0.0] * self.stock_dim)
        else:
            # Extract weights from previous state
            self.current_weights = np.array(self.previous_state[self.stock_dim:2*self.stock_dim])

        self.actions_memory = [self.current_weights.tolist()]
        
        self.date_memory = []
        
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
        
        # Initialize current data
        self.data = self.df.loc[self.day, :]
        self.date_memory = [self.data.date.unique()[0]]
        
        return self._build_observation(), {}

    def render(self, mode="human"):
        """Render current state."""
        return self._build_observation()

    def softmax_normalization(self, actions):
        """Softmax normalization for portfolio weights."""
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

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

    def get_sb_env(self):
        """Get Stable-Baselines3 compatible environment."""
        e = DummyVecEnv([lambda: self])
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
