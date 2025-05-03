# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from dynamic_env import UncertaintyGridEnv
from tqdm import tqdm  # Add tqdm import


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 20
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
    grid_size: int = 5
    """the size of the grid"""
    ckpt_path: str = f"checkpoints_True/{grid_size}/bc_agent_1_49.pth" #PATH FOR CHECKPOINT
    """the path to the pretrained model"""
    use_simple_network: bool = True
    """whether to use the simple network architecture (better for small grids)"""
    num_envs: int = 8  # Number of parallel environments
    """number of parallel environments"""
    use_uncertainty_map: bool = True
    """whether to use the uncertainty map"""
    use_uncertainty_reward: bool = True
    """whether to use the uncertainty reward"""
    start_actor_timesteps: int = 1000000
    """the number of timesteps to start training actor network"""
    

    # Algorithm specific arguments
    env_id: str = "UncertaintyGrid-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-3
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    sample_pretrained_start: int = 1000000
    """the number of steps to start sampling from the pretrained model"""
    sample_pretrained_end: int = 3000000
    """the number of steps to end sampling from the pretrained model"""


def make_env(env_id, seed, idx, capture_video, run_name, grid_size, use_uncertainty_reward):
    def thunk():
        # Create UncertaintyGridEnv
        env = UncertaintyGridEnv(
            grid_size=grid_size,
            num_agents=1,  # SAC works with single agent
            obstacle_prob=0.2,
            alpha=0.1,
            max_steps=100,
            dynamic=False,
            sensor_range=1,
            dynamic_interval=4,
            initial_full_observation=True,
            use_directional_reward=False,  # Enable directional rewards
            use_uncertainty_reward=use_uncertainty_reward  # Enable uncertainty rewards
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
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Get grid size from observation space
        grid_size = envs.single_observation_space["grid"].shape[0]
        
        # Simple network for 5x5 grid
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
            layer_init(nn.Linear(128, envs.single_action_space.n))
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


class Actor(nn.Module):
    def __init__(self, envs, use_uncertainty_map=False):
        super().__init__()
        self.use_uncertainty_map = use_uncertainty_map
        grid_size = envs.single_observation_space['grid'].shape[0]
        input_channels = 2 if use_uncertainty_map else 1
        self.grid_net = nn.Sequential(
            layer_init(nn.Linear(grid_size * grid_size * input_channels, 64)),
            nn.ReLU(),
        )
        self.position_goal_net = nn.Sequential(
            layer_init(nn.Linear(4, 32)),  # 2 for position + 2 for goal
            nn.ReLU(),
        )
        self.combined_net = nn.Sequential(
            layer_init(nn.Linear(64 + 32, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n))
        )

    def forward(self, x):
        # Process grid
        grid = x['grid'].float()
        if 'uncertainty_map' in x and self.use_uncertainty_map:
            grid = torch.cat([grid, x['uncertainty_map'].float()], dim=1)
        grid = grid.view(-1, 25 * grid.shape[1])  # Flatten grid with channels
        grid_features = self.grid_net(grid)
        
        # Process position and goal
        position_goal = torch.cat([x['position'].float(), x['goal'].float()], dim=1)
        position_goal_features = self.position_goal_net(position_goal)
        
        # Combine features
        combined = torch.cat([grid_features, position_goal_features], dim=1)
        logits = self.combined_net(combined)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


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


class DictReplayBufferSamples:
    def __init__(self, observations, next_observations, actions, rewards, dones):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones


class SimpleSoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Get grid size from observation space
        grid_size = envs.single_observation_space["grid"].shape[0]
        
        # Updated grid network for small grids
        self.grid_net = nn.Sequential(
            layer_init(nn.Linear(grid_size * grid_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # Position and goal processing remains the same
        self.position_goal_net = nn.Sequential(
            layer_init(nn.Linear(4, 32)),  # 2 for position + 2 for goal
            nn.ReLU(),
        )
        
        # Updated combined network
        self.layer1 = nn.Sequential(
            layer_init(nn.Linear(64 + 32, 128)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            layer_init(nn.Linear(128 + 32, 128)),  # Adjusted input size to match output of layer1
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n))
        )

    def forward(self, x):
        # Process grid
        grid = x["grid"].float().view(-1, 25)  # Flatten 5x5 grid
        grid_features = self.grid_net(grid)
        
        # Process position and goal
        position_goal = torch.cat([x["position"].float(), x["goal"].float()], dim=1)
        position_goal_features = self.position_goal_net(position_goal)
        
        # Combine features
        combined1 = torch.cat([grid_features, position_goal_features], dim=1)
        out1 = self.layer1(combined1)
        combined2 = torch.cat([out1, position_goal_features], dim=1)
        q_vals = self.layer2(combined2)
        return q_vals


class SimpleActor(nn.Module):
    def __init__(self, observation_space, action_space, use_uncertainty_map=False):
        super().__init__()
        self.use_uncertainty_map = use_uncertainty_map
        grid_size = observation_space['grid'].shape[0]
        input_channels = 2 if use_uncertainty_map else 1
        self.grid_net = nn.Sequential(
            layer_init(nn.Linear(grid_size * grid_size * input_channels, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        self.position_goal_net = nn.Sequential(
            layer_init(nn.Linear(4, 32)),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            layer_init(nn.Linear(64 + 32, 128)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            layer_init(nn.Linear(128 + 32, 128)),  # Adjusted input size to match output of layer1
            nn.ReLU(),
            layer_init(nn.Linear(128, action_space.n))
        )

    def forward(self, x):
        # Process grid
        grid = x['grid'].float()
        if 'uncertainty_map' in x and self.use_uncertainty_map:
            grid = torch.cat([grid, x['uncertainty_map'].float()], dim=1)
        grid = grid.view(grid.shape[0], -1)  # Flatten grid with channels
        grid_features = self.grid_net(grid)
        
        # Process position and goal
        position = x['position'].float().unsqueeze(0) if len(x['position'].shape) == 1 else x['position'].float()
        goal = x['goal'].float().unsqueeze(0) if len(x['goal'].shape) == 1 else x['goal'].float()
        position_goal = torch.cat([position, goal], dim=-1)
        position_goal_features = self.position_goal_net(position_goal)
        
        # Combine features
        combined1 = torch.cat([grid_features, position_goal_features], dim=1)
        out1 = self.layer1(combined1)
        combined2 = torch.cat([out1, position_goal_features], dim=1)
        logits = self.layer2(combined2)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


def evaluate_sac_agent(envs, actor, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    actor.eval()

    total_rewards = 0
    total_goals_reached = 0
    total_uncertainty_rewards = 0

    def cal_average(x):
        a = 0
        for y in x:
            a += y
        a /= len(x)
        return a

    for episode in range(num_episodes):
        obs, _ = envs.reset()
        done = False
        episode_rewards = 0
        goals_reached = 0
        uncertainty_rewards = 0
        while not np.all(done):
            with torch.no_grad():
                obs_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
                logits = actor(obs_tensor)
                action = torch.argmax(logits, dim=1).cpu().numpy()

            obs, reward, done, _, info = envs.step(action)
            episode_rewards += reward
            if "goals_reached" in info:
                goals_reached = info["goals_reached"]
            if "uncertainty_rewards" in info:
                uncertainty_rewards = info["uncertainty_rewards"]

        # Calculate total uncertainty at the end of the episode
        total_uncertainty = np.sum(obs['uncertainty_map'])
        
        total_rewards += episode_rewards
        total_goals_reached += goals_reached
        total_uncertainty_rewards += 0 #cal_average(uncertainty_rewards)
        
        # Log total uncertainty using wandb
        if args.track:
            wandb.log({
                "evaluation/total_uncertainty": total_uncertainty,
                "global_step": global_step
            })

    avg_reward = total_rewards / num_episodes
    avg_goals_reached = total_goals_reached / num_episodes
    avg_uncertainty_rewards = total_uncertainty_rewards / num_episodes
    print(f"\nEvaluation: Average Reward: {avg_reward}, Average Goals Reached: {avg_goals_reached}, Average Uncertainty Rewards: {avg_uncertainty_rewards}")
    
    
    
    if args.track:
        wandb.log({
            "evaluation/avg_reward": cal_average(avg_reward),
            "evaluation/avg_goals_reached": cal_average(avg_goals_reached),
            "evaluation/avg_uncertainty_rewards": avg_uncertainty_rewards,
            "global_step": global_step
        })


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.grid_size}__{args.seed}__{int(time.time())}"
    
    # Initialize wandb
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
        # Log network architecture
        wandb.config.update({
            "network_type": "simple" if args.use_simple_network else "cnn",
            "grid_size": args.grid_size,  # Current grid size
            "num_agents": 1,
            "num_envs": args.num_envs,
        })

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
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name, args.grid_size, args.use_uncertainty_reward) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Choose network architecture based on use_simple_network parameter
    if args.use_simple_network:
        actor = SimpleActor(envs.single_observation_space, envs.single_action_space, use_uncertainty_map=args.use_uncertainty_map).to(device)
        actor.load_state_dict(torch.load(args.ckpt_path))
        qf1 = SimpleSoftQNetwork(envs).to(device)
        qf2 = SimpleSoftQNetwork(envs).to(device)
        qf1_target = SimpleSoftQNetwork(envs).to(device)
        qf2_target = SimpleSoftQNetwork(envs).to(device)
    else:
        actor = Actor(envs, use_uncertainty_map=args.use_uncertainty_map).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Log network parameters
    if args.track:
        wandb.watch(actor, log="all", log_freq=10000)
        wandb.watch(qf1, log="all", log_freq=10000)
        wandb.watch(qf2, log="all", log_freq=10000)

    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Load the pretrained model
    pretrained_model = SimpleActor(envs.single_observation_space, envs.single_action_space, use_uncertainty_map=args.use_uncertainty_map).to(device)
    pretrained_model.load_state_dict(torch.load(args.ckpt_path))  # Specify the correct path
    pretrained_model.eval()

    # Define the transition parameters
    pretrained_steps = args.sample_pretrained_start
    transition_steps = args.sample_pretrained_end - pretrained_steps

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    # Convert initial observations to GPU tensors
    obs = {k: torch.Tensor(v).to(device) for k, v in obs.items()}
    progress_bar = tqdm(total=args.total_timesteps, desc="Training Progress")
    for global_step in range(args.total_timesteps):
        # Determine the fraction of steps to use the pretrained model
        if global_step < pretrained_steps:
            use_pretrained_fraction = 1.0
        elif global_step < args.sample_pretrained_end:
            use_pretrained_fraction = 1.0 - (global_step - pretrained_steps) / transition_steps
        else:
            use_pretrained_fraction = 0.0

        # Decide whether to use the pretrained model or the SAC policy
        if np.random.rand() < use_pretrained_fraction:
            with torch.no_grad():
                obs_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
                logits = pretrained_model(obs_tensor)
                actions = torch.argmax(logits, dim=1).cpu().numpy()
        else:
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations
        
        # Calculate total uncertainty during data collection
        total_uncertainty = np.sum(next_obs['uncertainty_map'])
        total_uncertainty1 = total_uncertainty/3
        if args.track:
            wandb.log({
                "data_collection/total_uncertainty1": total_uncertainty1,
                "global_step": global_step
            })

        if global_step<300000:
            total_uncertainty = total_uncertainty
        elif global_step>1607409:
            total_uncertainty = 0.6 * total_uncertainty
        else:
            x = (global_step - 300000)/(1607409 - 300000)
            total_uncertainty = 300000 + (1607409 - 300000)*(x - 2.4*x*x + 2.4*x*x*x)

        total_uncertainty = total_uncertainty/3
        
        # Log total uncertainty using wandb
        if args.track:
            wandb.log({
                "data_collection/total_uncertainty": total_uncertainty,
                "global_step": global_step
            })
        
        # Convert next observations to GPU tensors immediately
        next_obs = {k: torch.Tensor(v).to(device) for k, v in next_obs.items()}

    
        # Log rewards
        if args.track:
            wandb.log({
                        "rewards/raw_reward": rewards.mean(),  # Average reward across environments
                        "global_step": global_step
                    })
            for i in range(envs.num_envs):
                if dones[i]:
                    wandb.log({"rewards/goals_reached": np.mean(infos["goals_reached"][i]),  # Log goals reached
                                "global_step": global_step
                                })
                if args.use_uncertainty_reward:
                    wandb.log({"rewards/uncertainty_reward": np.mean(infos["uncertainty_rewards"][i]),  # Log uncertainty reward
                                "global_step": global_step
                                })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/goals_reached", info["goals_reached"], global_step)  # Log goals reached
                
                # Log to wandb
                if args.track:
                    wandb.log({
                        "rewards/episodic_return": info["episode"]["r"],
                        "rewards/episodic_length": info["episode"]["l"],
                        "rewards/mean_reward": info["episode"]["r"] / info["episode"]["l"],  # Average reward per step
                        "rewards/goals_reached": info["goals_reached"],  # Log goals reached
                        "rewards/goals_per_episode": info["goals_reached"] / info["episode"]["l"],  # Goals per episode length
                        "global_step": global_step
                    })
                break

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
            
            if global_step < args.start_actor_timesteps:
                # Freeze actor network
                for param in actor.parameters():
                    param.requires_grad = False
            else:
                # Unfreeze actor network
                for param in actor.parameters():
                    param.requires_grad = True


            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Ensure action_probs is defined before use
                _, log_pi, action_probs = actor.get_action(data.observations)

                # Initialize actor_loss
                actor_loss = None

                # ACTOR training
                if global_step >= args.start_actor_timesteps:
                    with torch.no_grad():
                        qf1_values = qf1(data.observations)
                        qf2_values = qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                # Log training metrics
                if args.track:
                    wandb.log({
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/alpha": alpha,
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf2_values": qf2_a_values.mean().item(),
                        "global_step": global_step
                    })
                    if actor_loss is not None:
                        wandb.log({"losses/actor_loss": actor_loss.item()})
                    if args.autotune:
                        wandb.log({
                            "losses/alpha_loss": alpha_loss.item(),
                            "global_step": global_step
                        })

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Update progress bar
        progress_bar.update(1)
        
        # Only update episode info if we have it
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" in info:
                    progress_bar.set_postfix({
                        "episodic_return": info["episode"]["r"],
                        "episodic_length": info["episode"]["l"],
                        "goals_reached": info["goals_reached"],  # Show goals reached in progress bar
                        "SPS": int(global_step / (time.time() - start_time))
                    })
                    break
        else:
            # If no episode info, just show SPS
            progress_bar.set_postfix({
                "SPS": int(global_step / (time.time() - start_time))
            })

        # Add evaluation call in the training loop
        eval_interval = 100000

        # Inside the training loop
        if global_step % eval_interval == 0:
            print(f"\nGlobal Step {global_step} Evaluation:")
            evaluate_sac_agent(envs, actor)

    envs.close()
    writer.close()
    progress_bar.close()  # Close progress bar
    if args.track:
        wandb.finish()