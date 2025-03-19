# DRL models from Stable Baselines 3
from __future__ import annotations

import time
import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=0,
        tensorboard_log=None,
    ):
        set_random_seed(seed)

        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model, 
        tb_log_name, 
        total_timesteps=5000,
        eval_env=None, 
        eval_freq=10000,
        best_model_save_path="./best_model",
        model_name=None, 
    ):
        """
        Train the given model and optionally use an evaluation callback to 
        select and save the best model during training.
        
        Parameters
        ----------
        model : RL model instance (e.g., PPO, A2C, etc.)
        tb_log_name : str
            Name for TensorBoard logs.
        total_timesteps : int
            Number of timesteps to train for.
        eval_env : gym.Env or VecEnv, optional
            Separate environment to evaluate the agent periodically.
        eval_freq : int, optional
            Evaluate the agent every `eval_freq` timesteps.
        best_model_save_path : str, optional
            Base folder where we store the best model subfolder.
        model_name : str, optional
            Name of the RL algorithm (e.g., "ppo", "a2c"). If provided,
            we create a subfolder with this name to avoid overwrites.
        """
        
        # Build the full path to ensure each model has its own subfolder
        if model_name is not None:
            best_model_save_path = os.path.join(best_model_save_path, f"{model_name}_best")
        # Now "best_model.zip" will be saved inside e.g. "./best_model/ppo_best/best_model.zip"

        if eval_env is not None:
            # Ensure directory exists
            os.makedirs(best_model_save_path, exist_ok=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=best_model_save_path,
                log_path=best_model_save_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            model = model.learn(
                total_timesteps=total_timesteps,
                tb_log_name=tb_log_name,
                callback=[TensorboardCallback(), eval_callback],
            )
        else:
            model = model.learn(
                total_timesteps=total_timesteps,
                tb_log_name=tb_log_name,
                callback=TensorboardCallback(),
            )
        return model

    
    @staticmethod
    def env_constructor(the_df, **kwargs):
        return StockTradingEnv(df=the_df, **kwargs)
    
    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """
        Read the environment-saved CSV from disk and compute Sharpe ratio.
        Matches library approach of reading:
        results/account_value_validation_{model_name}_{iteration}.csv
        """
 

        file_path = f"results/account_value_validation_{model_name}_{iteration}.csv"
        df_total_value = pd.read_csv(file_path)

        # If agent did not make any transaction or daily_return=0
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
        
    @staticmethod
    def get_validation_sharpe_custom(file_path):
        """
        Read the environment-saved CSV from disk and compute Sharpe ratio.
        Matches library approach of reading:
        results/account_value_validation_{model_name}_{iteration}.csv
        """

        if not os.path.exists(file_path):
            print(f"{file_path} not found, returning -999")
            return -999

        df_total_value = pd.read_csv(file_path)
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    @staticmethod    
    def _run_env_to_end(model, vec_env):
        """
        Step the model in vec_env until done=True, 
        ensuring we produce the final CSV.
        """
        obs = vec_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            results = vec_env.step(action)

            # Adjust if using gym vs gymnasium
            if len(results) == 4:
                obs, rewards, done, info = results
            else:
                obs, rewards, done, truncated, info = results
                done = np.logical_or(done, truncated)

            if done[0]:
                break

    
    def search_best_hparams(
        self, 
        model_name,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        param_grid: list[dict],
        total_timesteps=50_000,
        env_constructor=None,
        eval_freq=10_000,
        best_model_save_path="./best_model",
        **env_kwargs
    ):
        """
        Conduct hyperparameter search:
      - For each hyperparam dictionary in param_grid:
        1) Train on train_df with `eval_env=val_env` so that
           EvalCallback can save the best checkpoint by mean reward.
        2) After training, load that 'best checkpoint' from disk.
        3) Compute Sharpe ratio on val_df.
      - Keep whichever param set yields the highest Sharpe ratio on val_df.

        Parameters
        ----------
        model_name : str
            One of {"ppo", "a2c", "ddpg", "td3", "sac"} in your code.
        train_df : pd.DataFrame
            In-sample training data.
        val_df : pd.DataFrame
            Validation data used for hyperparam selection.
        param_grid : list of dict
            Each dict is a hyperparam combination to try (e.g., {"learning_rate": 1e-4, "n_steps": 2048}).
        total_timesteps : int
            Number of timesteps to train each candidate.
        env_constructor : callable
            A function/lambda that builds your StockTradingEnv given df & env_kwargs.
        eval_freq : int
            Frequency (in timesteps) at which EvalCallback runs & checks for best reward.
        best_model_base_path : str
            Folder where we store the best model checkpoint for each param set.
        env_kwargs : dict
            Additional args to pass into env_constructor (like initial_amount, etc.).
         """

        best_sharpe = -999
        best_params = None
        best_model = None

        # If no env_constructor was provided, default to the static method:
        if env_constructor is None:
            env_constructor = DRLAgent.env_constructor
        # Build the training env
        train_env_gym = env_constructor(train_df, **env_kwargs)
        env_train, _ = train_env_gym.get_sb_env()

        # Build the validation env
        val_env_gym = env_constructor(val_df, **env_kwargs)
        env_val, _ = val_env_gym.get_sb_env()

        for i, param_dict in enumerate(param_grid, start=1):
            print(f"Testing param set {i}/{len(param_grid)}: {param_dict}")

            # Create model
            self.env = env_train
            model = self.get_model(model_name, model_kwargs=param_dict, verbose=0)

            # Train
            model = self.train_model(
                model,
                tb_log_name=f"{model_name}_search_{i}",
                total_timesteps=total_timesteps,
                eval_env=env_val,  
                eval_freq=eval_freq,
                best_model_save_path=best_model_save_path,
                model_name=model_name,  # ensures a subfolder for each model
            )

            # 4) Load the best checkpoint from the callback (best_model.zip)
            model_subfolder = os.path.join(best_model_save_path, f"{model_name}_best")
            best_checkpoint_path = os.path.join(model_subfolder, "best_model.zip")

            if not os.path.exists(best_checkpoint_path):
                # If no best_model.zip was saved (e.g., no improvement found), 
                # fallback to the final model as a last resort:
                print("Warning: No best_model.zip found. Using final trained model instead.")
                best_model_checkpoint = model
            else:
                # Load the best checkpoint
                # (We assume the class is the same as model_name)
                if model_name == "ppo":
                    best_model_checkpoint = PPO.load(best_checkpoint_path)
                elif model_name == "a2c":
                    best_model_checkpoint = A2C.load(best_checkpoint_path)
                elif model_name == "ddpg":
                    best_model_checkpoint = DDPG.load(best_checkpoint_path)
                elif model_name == "td3":
                    best_model_checkpoint = TD3.load(best_checkpoint_path)
                elif model_name == "sac":
                    best_model_checkpoint = SAC.load(best_checkpoint_path)
                else:
                    best_model_checkpoint = model  # fallback

             # 4) Compare final model vs checkpoint by their CSV-based Sharpe

            # --(A) Build validation environment for the final model
            # We pass iteration=i, but let's do mode="validation_final"
            # so we produce e.g. account_value_validation_final_{model_name}_{i}.csv
            val_env_final = env_constructor(
                val_df,
                iteration=i,
                model_name=f"{model_name}_final",  # or some suffix
                mode="validation",
                **env_kwargs
            )
            val_env_final_vec = DummyVecEnv([lambda: val_env_final])
            # short rollout
            self._run_env_to_end(model, val_env_final_vec)
            # now environment writes: "account_value_validation_{model_name}_final_{i}.csv"
            file_final = f"results/account_value_validation_{model_name}_final_{i}.csv"
            if not os.path.exists(file_final):
                print("Warning: final model CSV not found, perhaps environment didn't terminate?")
                final_sharpe = -999
            else:
                final_sharpe = self.get_validation_sharpe_custom(file_final)
            print(f"Final model Sharpe for param {i}: {final_sharpe:.4f}")

            # --(B) Build validation environment for the checkpoint
            # so we produce e.g. account_value_validation_ckpt_{model_name}_{i}.csv
            val_env_ckpt = env_constructor(
                val_df,
                iteration=i,
                model_name=f"{model_name}_ckpt",  # or some suffix
                mode="validation",
                **env_kwargs
            )
            val_env_ckpt_vec = DummyVecEnv([lambda: val_env_ckpt])
            self._run_env_to_end(best_model_checkpoint, val_env_ckpt_vec)
            file_ckpt = f"results/account_value_validation_{model_name}_ckpt_{i}.csv"
            if not os.path.exists(file_ckpt):
                print("Warning: checkpoint model CSV not found.")
                ckpt_sharpe = -999
            else:
                ckpt_sharpe = self.get_validation_sharpe_custom(file_ckpt)
            print(f"Checkpoint model Sharpe for param {i}: {ckpt_sharpe:.4f}")
                
            # 5) Decide which is better for this param set
            if ckpt_sharpe >= final_sharpe:
                param_best_sharpe = ckpt_sharpe
                param_best_model  = best_model_checkpoint
            else:
                param_best_sharpe = final_sharpe
                param_best_model  = model

            # 6) Compare across all param sets
            if param_best_sharpe > best_sharpe:
                best_sharpe = param_best_sharpe
                best_params = param_dict
                best_model  = param_best_model

        print("===== Hyperparam Search Finished =====")
        print(f"Best Sharpe: {best_sharpe:.4f}")
        print(f"Best Params: {best_params}")
        return best_params, best_model
    
    def walk_forward_final_vs_checkpoint(
        self,
        df: pd.DataFrame,
        unique_trade_dates: np.ndarray,
        start_date: str,
        end_date: str,
        model_name: str,
        fixed_params: dict,
        rebalance_window: int,
        val_window: int,
        total_timesteps=50_000,
        env_constructor=None,
        eval_freq=10_000,
        best_model_prefix="./walkforward_best_model",
        seed="",
        **env_kwargs
    ):
        """
        An alternative walk-forward method:
        1) For each iteration, define a 'train window' and a small 'validation window'.
        2) Train from scratch with an EvalCallback on that validation window => saving best model by reward.
        3) Compare the final model vs. best-checkpoint by Sharpe ratio (on same validation window).
        4) Whichever is higher Sharpe => used to 'trade' the next (rebalance) window.

        Parameters
        ----------
        df : pd.DataFrame
            Your entire dataset with 'date' column.
        unique_trade_dates : np.ndarray
            Sorted unique dates from df.
        start_date : str
            The date from which you start the first iteration (must be after your big initial train+val).
        end_date : str
            The date to stop.
        model_name : str
            e.g., "ppo"
        fixed_params : dict
            The hyperparams found from search_best_hparams, e.g. {"learning_rate":..., "n_steps":...}
        rebalance_window : int
            Number of days to 'trade' after each iteration.
        val_window : int
            Number of days used for that iteration's 'EvalCallback' validation + final checkpoint vs. final comparison.
        total_timesteps : int
            SB3 timesteps each iteration.
        env_constructor : callable
            e.g., lambda some_df, **kwargs: StockTradingEnv(df=some_df, **kwargs).
        eval_freq : int
            How often (in timesteps) the EvalCallback checks mean reward.
        best_model_prefix : str
            The folder prefix for storing checkpoint logs, e.g. "./walkforward_best_model".
        seed : str
            A seed string to append to filenames for reproducibility.
        env_kwargs : extra arguments
            Additional environment arguments like initial_amount, stock_dim, etc.

        Returns
        -------
        pd.DataFrame
            A summary of each iteration with final vs. checkpoint Sharpe, which model was chosen, and the trading segment.
        """
        # If no env_constructor was provided, default to the static method:
        if env_constructor is None:
            env_constructor = DRLAgent.env_constructor

        if not isinstance(seed, int):
            seed = 0

        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)

        sorted_dates = np.sort(unique_trade_dates)
        start_idx = np.searchsorted(sorted_dates, np.datetime64(start_date))
        end_idx = np.searchsorted(sorted_dates, np.datetime64(end_date))

        if start_idx >= end_idx:
            print("No walk-forward possible, start_date >= end_date.")
            return None

        iteration_list = []
        final_sharpe_list = []
        checkpoint_sharpe_list = []
        chosen_type_list = []
        # We collect trading data for analysis:
        # Each trading window can produce "account_memory" and "actions_memory".
        # We'll store them in lists of DataFrames, then concat at the end.
        all_account_memories = []
        all_actions_memories = []

        current_idx = start_idx
        iteration_no = 0

        while current_idx < end_idx:
            iteration_no += 1

            # The training window ends right before the 'val_window' days
            # so: train_end_date = sorted_dates[current_idx] -> used for training
            # next val_window: [current_idx, current_idx + val_window)
            # then trade = [current_idx + val_window, current_idx + val_window + rebalance_window)
            val_end_idx = min(current_idx + val_window, end_idx)
            if val_end_idx == current_idx:
                print("No space for validation window, stopping.")
                break

            train_end_date = sorted_dates[current_idx]
            val_start_date = sorted_dates[current_idx]
            val_end_date_ = sorted_dates[val_end_idx]  # inclusive

            print("========================================")
            print(f"Iteration {iteration_no}")
            print(f"Train up to: {train_end_date}")
            print(f"Validation window: {val_start_date} to {val_end_date_}")

            # 1) Prepare train data
            train_df = data_split(df, start=sorted_dates[0], end=train_end_date)

            # 2) Prepare validation data
            val_df = data_split(df, start=val_start_date, end=val_end_date_)

            # Build train env
            train_env_gym = env_constructor(train_df, **env_kwargs)
            env_train, _ = train_env_gym.get_sb_env()
            env_train.seed(seed)

            # Build val env
            val_env_gym = env_constructor(val_df, **env_kwargs)
            env_val, _ = val_env_gym.get_sb_env()
            env_val.seed(seed)

            # 3) Create model & setup callback
            self.env = env_train
            model = self.get_model(model_name, model_kwargs=fixed_params, verbose=0, seed=seed)

            # Use existing train_model to avoid repeating code:
            iteration_save_dir = os.path.join(best_model_prefix, f"{model_name}_iter_{iteration_no}")
            final_model = DRLAgent.train_model(
            model,
            tb_log_name=f"walk_forward_{iteration_no}",
            total_timesteps=total_timesteps,
            eval_env=env_val,        # optionally pass a different env if you want checkpoint by reward
            eval_freq=eval_freq,
            best_model_save_path=iteration_save_dir,
            model_name=model_name,
        )
            # final model is after training all timesteps

            # Does best_model.zip exist?
            model_subfolder = os.path.join(iteration_save_dir, f"{model_name}_best")
            best_model_path = os.path.join(model_subfolder, "best_model.zip")

            if not os.path.exists(best_model_path):
                print(f"No best_model.zip found => no improvement by reward? Using final.")
                checkpoint_model = final_model
                checkpoint_sharpe = -999
            else:
                # Load the best checkpoint
                if model_name == "ppo":
                    checkpoint_model = PPO.load(best_model_path)
                elif model_name == "a2c":
                    checkpoint_model = A2C.load(best_model_path)
                elif model_name == "ddpg":
                    checkpoint_model = DDPG.load(best_model_path)
                elif model_name == "td3":
                    checkpoint_model = TD3.load(best_model_path)
                elif model_name == "sac":
                    checkpoint_model = SAC.load(best_model_path)
                else:
                    checkpoint_model = final_model  # fallback

            # 5) Evaluate final vs checkpoint by environment-based CSV

            # (A) Rollout final model => environment with mode="validation"
            # Use a suffix in model_name so we don't overwrite checkpoint CSV
            val_env_gym_final = env_constructor(
                val_df,
                iteration=iteration_no,
                seed = seed,
                model_name=f"{model_name}_final",  # e.g. "ppo_final"
                mode="validation",
                **env_kwargs
            )
            val_env_final = DummyVecEnv([lambda: val_env_gym_final])
            val_env_final.seed(seed)
            self._run_env_to_end(final_model, val_env_final)

            file_final = f"results/account_value_validation_{model_name}_final_{iteration_no}_{seed}.csv"
            final_sharpe = self.get_validation_sharpe_custom(file_final)

             # (B) Rollout checkpoint model => environment with mode="validation"
            val_env_gym_ckpt = env_constructor(
                val_df,
                iteration=iteration_no,
                seed = seed,
                model_name=f"{model_name}_ckpt",   # e.g. "ppo_ckpt"
                mode="validation",
                **env_kwargs
            )
            val_env_ckpt = DummyVecEnv([lambda: val_env_gym_ckpt])
            val_env_ckpt.seed(seed)
            # short rollout => writes results/account_value_validation_{model_name}_ckpt_{iteration_no}.csv
            self._run_env_to_end(checkpoint_model, val_env_ckpt)

            file_ckpt = f"results/account_value_validation_{model_name}_ckpt_{iteration_no}_{seed}.csv"
            checkpoint_sharpe = self.get_validation_sharpe_custom(file_ckpt)

            print(f"Final Sharpe = {final_sharpe:.4f}")
            print(f"Checkpoint Sharpe = {checkpoint_sharpe:.4f}")

            if checkpoint_sharpe >= final_sharpe:
                best_policy = checkpoint_model
                best_type   = "checkpoint"
            else:
                best_policy = final_model
                best_type   = "final"

            iteration_list.append(iteration_no)
            final_sharpe_list.append(final_sharpe)
            checkpoint_sharpe_list.append(checkpoint_sharpe)
            chosen_type_list.append(best_type)

            # 6) Trading window => from val_end_idx to val_end_idx + rebalance_window
            trade_start_idx = val_end_idx
            trade_end_idx = min(val_end_idx + rebalance_window, end_idx)
            if trade_start_idx >= end_idx:
                print("No more trade windows left after validation.")
                break

            trade_start_date_ = sorted_dates[trade_start_idx]
            trade_end_date_ = sorted_dates[trade_end_idx] if trade_end_idx > trade_start_idx else trade_start_date_
            print(f"Trading from {trade_start_date_} to {trade_end_date_}")

            trade_df = data_split(df, start=trade_start_date_, end=trade_end_date_)
            if len(trade_df) == 0:
                print("No trade data => break.")
                break

            # 7) Reuse DRL_prediction => collects account_memory & actions_memory
            trade_env_gym = env_constructor(trade_df, **env_kwargs)
            account_mem, actions_mem = self.DRL_prediction(best_policy, trade_env_gym, deterministic=True)

            # We can store these in lists for later analysis.
            # Example: store them as dataframes with iteration info
            df_account = account_mem.copy()  # account_mem is presumably a DataFrame (depends on your env)
            df_actions = actions_mem.copy()  # actions_mem is presumably a DataFrame

            # If "save_asset_memory()" returns a DataFrame with 'date' and 'account_value' columns,
            # we can tag them with the iteration number, or start/end dates:
            df_account["Iteration"] = iteration_no
            df_account["TradeStart"] = trade_start_date_
            df_account["TradeEnd"] = trade_end_date_

            # Same for actions
            df_actions["Iteration"] = iteration_no
            df_actions["TradeStart"] = trade_start_date_
            df_actions["TradeEnd"] = trade_end_date_

            all_account_memories.append(df_account)
            all_actions_memories.append(df_actions)

            # Move index forward
            current_idx = val_end_idx

        # Summaries
        df_res = pd.DataFrame({
            "Iteration": iteration_list,
            "FinalSharpe": final_sharpe_list,
            "CheckpointSharpe": checkpoint_sharpe_list,
            "ChosenType": chosen_type_list,
        })

        # Concatenate all the stored memories
        if len(all_account_memories) > 0:
            df_account_all = pd.concat(all_account_memories, ignore_index=True)
        else:
            df_account_all = pd.DataFrame()

        if len(all_actions_memories) > 0:
            df_actions_all = pd.concat(all_actions_memories, ignore_index=False)
        else:
            df_actions_all = pd.DataFrame()

        return df_res, df_account_all, df_actions_all
    
    @staticmethod
    def evaluate_sharpe(model, vec_env):
        """
        Example function to compute Sharpe on a vectorized environment's data.
        """
        obs = vec_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = vec_env.step(action)

        # retrieve daily returns from the env
        df_account_val = vec_env.env_method("save_asset_memory")[0]
        # daily_return might be account_value.pct_change or already in the env data
        if "account_value" in df_account_val.columns:
            daily_rets = df_account_val["account_value"].pct_change().dropna()
        else:
            # or if your env logs daily_return directly, adapt accordingly
            daily_rets = df_account_val["daily_return"]

        if daily_rets.std() == 0:
            return 0.0
        sharpe = (252**0.5) * daily_rets.mean() / daily_rets.std()
        return sharpe

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets


class DRLEnsembleAgent:
    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=0,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            f"results/account_value_validation_{model_name}_{iteration}.csv"
        )
        # If the agent did not make any transaction
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
        seed = 0
    ):
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.train_env = None  # defined in train_validation() function
        self.seed = seed

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, name, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs, deterministic=True)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.envs[0].render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{name}_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        model_name,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps_dict,
        i,
        validation,
        turbulence_threshold,
    ):
        """
        Train the model for a single window.
        """
        if model_kwargs is None:
            return None, sharpe_list, -1

        print(f"======{model_name} Training========")
        model = self.get_model(
            model_name, self.train_env, policy="MlpPolicy", model_kwargs=model_kwargs,
            seed=self.seed
        )
        model = self.train_model(
            model,
            model_name,
            tb_log_name=f"{model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps_dict[model_name],
        )  # 100_000
        print(
            f"======{model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=validation,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                    model_name=model_name,
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        val_env.seed(self.seed)
        self.DRL_validation(
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        sharpe = self.get_validation_sharpe(i, model_name=model_name)
        print(f"{model_name} Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe

    def run_ensemble_strategy(
        self,
        A2C_model_kwargs,
        PPO_model_kwargs,
        DDPG_model_kwargs,
        SAC_model_kwargs,
        TD3_model_kwargs,
        timesteps_dict,
    ):
        # Model Parameters
        kwargs = {
            "a2c": A2C_model_kwargs,
            "ppo": PPO_model_kwargs,
            "ddpg": DDPG_model_kwargs,
            "sac": SAC_model_kwargs,
            "td3": TD3_model_kwargs,
        }
        # Model Sharpe Ratios
        model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in MODELS.keys()}
        

        """Ensemble Strategy that combines A2C, PPO, DDPG, SAC, and TD3"""
        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[
                start_date_index : (end_date_index + 1), :
            ]

            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            # print(historical_turbulence_mean)

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # Environment Setup starts
            # training env
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )
            self.train_env.seed(self.seed)

            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment Setup ends

            # Training and Validation starts
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")
            # Train Each Model
            for model_name in MODELS.keys():
                # Train The Model
                model, sharpe_list, sharpe = self._train_window(
                    model_name,
                    kwargs[model_name],
                    model_dct[model_name]["sharpe_list"],
                    validation_start_date,
                    validation_end_date,
                    timesteps_dict,
                    i,
                    validation,
                    turbulence_threshold,
                )
                # Save the model's sharpe ratios, and the model itself
                model_dct[model_name]["sharpe_list"] = sharpe_list
                model_dct[model_name]["model"] = model
                model_dct[model_name]["sharpe"] = sharpe

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment setup for model retraining up to first trade date
            # train_full = data_split(self.df, start=self.train_period[0],
            # end=self.unique_trade_date[i - self.rebalance_window])
            # self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
            #                                               self.stock_dim,
            #                                               self.hmax,
            #                                               self.initial_amount,
            #                                               self.buy_cost_pct,
            #                                               self.sell_cost_pct,
            #                                               self.reward_scaling,
            #                                               self.state_space,
            #                                               self.action_space,
            #                                               self.tech_indicator_list,
            #                                              print_verbosity=self.print_verbosity
            # )])
            # Model Selection based on sharpe ratio
            # Same order as MODELS: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
            sharpes = [model_dct[k]["sharpe"] for k in MODELS.keys()]
            # Find the model with the highest sharpe ratio
            max_mod = list(MODELS.keys())[np.argmax(sharpes)]
            model_use.append(max_mod.upper())
            model_ensemble = model_dct[max_mod]["model"]
            # Training and Validation ends

            # Trading starts
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )
            # Trading ends

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                model_dct["a2c"]["sharpe_list"],
                model_dct["ppo"]["sharpe_list"],
                model_dct["ddpg"]["sharpe_list"],
                model_dct["sac"]["sharpe_list"],
                model_dct["td3"]["sharpe_list"],
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
            "PPO Sharpe",
            "DDPG Sharpe",
            "SAC Sharpe",
            "TD3 Sharpe",
        ]

        return df_summary
    