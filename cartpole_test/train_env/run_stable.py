from train_env import myTrainEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

env = myTrainEnv.TrainEnv()
check_env(env, warn=True)


def main():
    model = DQN(
        "MlpPolicy",
        env=env,
        learning_rate=5e-4,
        batch_size=256,
        buffer_size=1000000,
        learning_starts=1000,
        target_update_interval=5000,
        gamma=0.99,
        exploration_fraction=5e-6,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./tensorboard/mytrain/"
    )
    model.learn(total_timesteps=1e6)