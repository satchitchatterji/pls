import gym

import torch as th
from torch import nn
import os

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from wandb.integration.sb3 import WandbCallback

# register the custom environments
import carracing_gym
import pacman_gym

def main(
    config_folder,                           # path to the config file
    config,                                  # a dict containing the configuration
    model_cls,                               # PPO_shielded
    get_sensor_value_ground_truth,           # get_ground_wall or get_ground_truth_of_grass
    custom_callback_cls,                     # Pacman_Callback or Carracing_Callback
    monitor_cls,                             # Pacman_Monitor or Carracing_Monitor
    features_extractor_cls,                  # Pacman_FeaturesExtractor or Carracing_FeaturesExtractor
    observation_net_cls                      # Pacman_Observation_Net or Carracing_Observation_Net
):
    """
    Executing PLPG

    :param config_folder: location of the config file
    :param config: a dict containing the configuration
    :param model_cls: rl algorithm
    :param get_sensor_value_ground_truth: function used to compute ground truth observations from image input
    :param custom_callback_cls: callback class
    :param monitor_cls: monitor class
    :param features_extractor_cls:  features extractor class
    :param observation_net_cls:     observation net class
    :return:    
    """

    net_arch = config["policy_params"]["net_arch_shared"] + [
        dict(
            pi=config["policy_params"]["net_arch_pi"],
            vf=config["policy_params"]["net_arch_vf"],
        )
    ] 

    observation_params = config["observation_params"]
    shield_params = config["shield_params"]
    policy_safety_params = config["policy_safety_params"]
    policy_safety_params["config_folder"] = config_folder
    policy_safety_params[
        "get_sensor_value_ground_truth"
    ] = get_sensor_value_ground_truth

    policy_safety_params["observation_net_cls"] = observation_net_cls

    policy_safety_params.update(observation_params) # this is a dictionary




    # initialize the loggers
    new_logger = configure(config_folder, ["log", "tensorboard"])

    # initialize the environment
    env = gym.make(config["env"], **config["env_features"])

    if config["monitor_features"] is not None:
        env = monitor_cls(
            env,
            allow_early_resets=False,
            **config["monitor_features"]
        )
    else:
        env = monitor_cls(
            env,
            allow_early_resets=False
        )

    # create a callback for logging

    if shield_params is not None:
        shield_params["observation_net_cls"] = observation_net_cls
        shield_params.update(observation_params)

    custom_callback = custom_callback_cls(policy_safety_params=policy_safety_params)
    progress_callback = ProgressBarCallback(config["policy_params"]["total_timesteps"])
    # initialize the rl algorithm
    model = model_cls(
        env=env,
        learning_rate=config["policy_params"]["learning_rate"],
        n_steps=config["policy_params"]["n_steps"],  # number of steps to run for each environment per update
        batch_size=config["policy_params"]["batch_size"],
        n_epochs=config["policy_params"]["n_epochs"],
        gamma=config["policy_params"]["gamma"],
        clip_range=config["policy_params"]["clip_range"],
        tensorboard_log=config_folder,
        policy_kwargs={
            "shield_params": shield_params,
            "net_arch": net_arch,
            "activation_fn": nn.ReLU,
            "optimizer_class": th.optim.Adam,
            "config_folder": config_folder,
            "get_sensor_value_ground_truth": get_sensor_value_ground_truth,
            "features_extractor_class": features_extractor_cls,
        },
        verbose=0,
        seed=config["policy_params"]["seed"],
        _init_setup_model=True,
        alpha=config["policy_params"]["alpha"],
        policy_safety_params=policy_safety_params,
    )

    model.set_random_seed(config["policy_params"]["seed"])
    model.set_logger(new_logger)

    # start training the rl algorithm
    intermediate_model_path = os.path.join(config_folder, "model_checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=5e4, save_path=intermediate_model_path
    )
    
    wandb_callback = EmptyCallback()
    if config["monitor_wandb"]:
        wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=config["policy_params"]["total_timesteps"],
        callback=[custom_callback, checkpoint_callback, progress_callback, wandb_callback]
    )
    print(env.get_episode_rewards())
    # save the train policy
    model.save(os.path.join(config_folder, "model"))


from tqdm import tqdm

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):  # total_timesteps is total number of steps
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        self.pbar.update(1)
        return True  # returns True to continue training

    def _on_training_end(self):
        self.pbar.close()

class EmptyCallback(BaseCallback):
    def __init__(self):
        super(EmptyCallback, self).__init__()

    def _on_training_start(self):
        pass

    def _on_step(self):
        return True  # returns True to continue training
    
    def _on_training_end(self):
        pass