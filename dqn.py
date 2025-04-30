# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from dynamic_env import UncertaintyGridEnv
from tqdm import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "uncertainty_grid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "UncertaintyGrid-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 5000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # Environment specific arguments
    grid_size: int = 5
    """size of the grid environment"""
    obstacle_prob: float = 0.2
    """probability of obstacles in the grid"""
    alpha: float = 0.1
    """uncertainty decay rate"""
    max_steps: int = 100
    """maximum steps per episode"""
    dynamic: bool = True
    """whether the environment is dynamic"""
    sensor_range: int = 1
    """sensor range of the agent"""
    dynamic_interval: int = 4
    """interval between dynamic changes"""
    initial_full_observation: bool = True
    """whether to provide full observation at the start"""
    use_directional_reward: bool = True
    """whether to use directional rewards"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # Create UncertaintyGridEnv
        env = UncertaintyGridEnv(
            grid_size=5,
            num_agents=1,  # DQN works with single agent
            obstacle_prob=0.2,
            alpha=0.1,
            max_steps=100,
            dynamic=True,
            sensor_range=1,
            dynamic_interval=4,
            initial_full_observation=True,
            use_directional_reward=True  # Enable directional rewards
        )
        
        # Add wrappers for recording
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Set seed using numpy's random seed
        np.random.seed(seed + idx)  # Different seed for each environment
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Get grid size from observation space
        grid_size = env.single_observation_space["grid"].shape[0]
        
        # Grid processing network
        self.grid_net = nn.Sequential(
            layer_init(nn.Linear(grid_size * grid_size, 64)),
            nn.ReLU(),
        )
        
        # Position and goal processing
        self.position_goal_net = nn.Sequential(
            layer_init(nn.Linear(4, 32)),  # 2 for position + 2 for goal
            nn.ReLU(),
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            layer_init(nn.Linear(64 + 32, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, env.single_action_space.n))
        )

    def forward(self, x):
        # Process grid
        grid = x["grid"].float().view(-1, 25)  # Flatten 5x5 grid
        grid_features = self.grid_net(grid)
        
        # Process position and goal
        position_goal = torch.cat([x["position"].float(), x["goal"].float()], dim=1)
        position_goal_features = self.position_goal_net(position_goal)
        
        # Combine features
        combined = torch.cat([grid_features, position_goal_features], dim=1)
        q_vals = self.combined_net(combined)
        return q_vals


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@dataclass
class DictReplayBufferSamples:
    observations: Dict[str, torch.Tensor]
    next_observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

class DictReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, handle_timeout_termination=True):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.handle_timeout_termination = handle_timeout_termination
        self.pos = 0
        self.full = False

        # Create buffers for each observation key
        self.observations = {
            key: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for key, space in observation_space.spaces.items()
        }
        self.next_observations = {
            key: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for key, space in observation_space.spaces.items()
        }

        self.actions = np.zeros((buffer_size,), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, info):
        # Copy each observation component
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Ensure actions, rewards, and dones are properly shaped
        self.actions[self.pos] = np.array(action).item()  # Convert to scalar
        self.rewards[self.pos] = np.array(reward).item()  # Convert to scalar
        self.dones[self.pos] = np.array(done).item()  # Convert to scalar

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        data = {
            "observations": {
                key: torch.as_tensor(obs[batch_inds], device=self.device)
                for key, obs in self.observations.items()
            },
            "next_observations": {
                key: torch.as_tensor(next_obs[batch_inds], device=self.device)
                for key, next_obs in self.next_observations.items()
            },
            "actions": torch.as_tensor(self.actions[batch_inds], device=self.device).unsqueeze(1),  # Add dimension for gather
            "rewards": torch.as_tensor(self.rewards[batch_inds], device=self.device),
            "dones": torch.as_tensor(self.dones[batch_inds], device=self.device),
        }

        return DictReplayBufferSamples(**data)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # Convert initial observations to GPU tensors
    obs = {k: torch.Tensor(v).to(device) for k, v in obs.items()}
    progress_bar = tqdm(total=args.total_timesteps, desc="Training Progress")
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(obs)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Convert next observations to GPU tensors immediately
        next_obs = {k: torch.Tensor(v).to(device) for k, v in next_obs.items()}

        # Log rewards
        if args.track:
            wandb.log({
                "rewards/raw_reward": rewards.mean(),  # Average reward across environments
                "rewards/min_reward": rewards.min(),  # Minimum reward
                "rewards/max_reward": rewards.max(),  # Maximum reward
                "rewards/std_reward": rewards.std(),  # Standard deviation of rewards
                "global_step": global_step
            })
            if "goals_reached" in infos:
                wandb.log({
                    "rewards/goals_reached": np.mean(infos["goals_reached"]),
                    "rewards/goals_reached_min": np.min(infos["goals_reached"]),
                    "rewards/goals_reached_max": np.max(infos["goals_reached"]),
                    "rewards/goals_reached_std": np.std(infos["goals_reached"]),
                    "global_step": global_step
                })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if "goals_reached" in info:
                        writer.add_scalar("charts/goals_reached", info["goals_reached"], global_step)
                    
                    # Log to wandb
                    if args.track:
                        wandb.log({
                            "rewards/episodic_return": info["episode"]["r"],
                            "rewards/episodic_length": info["episode"]["l"],
                            "rewards/mean_reward": info["episode"]["r"] / info["episode"]["l"],
                            "rewards/goals_reached": info["goals_reached"],
                            "rewards/goals_per_episode": info["goals_reached"] / info["episode"]["l"],
                            "rewards/episodic_return_min": info["episode"]["r"],
                            "rewards/episodic_return_max": info["episode"]["r"],
                            "rewards/episodic_length_min": info["episode"]["l"],
                            "rewards/episodic_length_max": info["episode"]["l"],
                            "global_step": global_step
                        })

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = torch.Tensor(infos["final_observation"][idx]).to(device)
        
        # Add experiences from all environments to the replay buffer
        for i in range(envs.num_envs):
            # Convert tensors back to numpy for storage in replay buffer
            obs_np = {k: v[i].cpu().numpy() for k, v in obs.items()}
            next_obs_np = {k: v[i].cpu().numpy() for k, v in real_next_obs.items()}
            rb.add(
                obs_np,
                next_obs_np,
                actions[i],
                rewards[i],
                terminations[i],
                {k: v[i] for k, v in infos.items() if isinstance(v, (list, np.ndarray))}
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    if args.track:
                        wandb.log({
                            "losses/td_loss": loss.item(),
                            "losses/q_values": old_val.mean().item(),
                            "losses/q_values_min": old_val.min().item(),
                            "losses/q_values_max": old_val.max().item(),
                            "losses/q_values_std": old_val.std().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                            "global_step": global_step
                        })

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # Update progress bar
        progress_bar.update(1)
        
        # Only update episode info if we have it
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" in info:
                    progress_bar.set_postfix({
                        "episodic_return": info["episode"]["r"],
                        "episodic_length": info["episode"]["l"],
                        "goals_reached": info.get("goals_reached", 0),
                        "SPS": int(global_step / (time.time() - start_time))
                    })
                    break
        else:
            # If no episode info, just show SPS
            progress_bar.set_postfix({
                "SPS": int(global_step / (time.time() - start_time))
            })

    envs.close()
    writer.close()
    progress_bar.close()
    if args.track:
        wandb.finish()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")