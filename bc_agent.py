import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from dynamic_env import UncertaintyGridEnv
from inter_astar import astar, compute_action
from tqdm import tqdm
import os

class BCDataset(Dataset):
    def __init__(self, observations, actions, device):
        self.observations = observations
        self.actions = actions
        self.device = device

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        #action = torch.tensor(self.actions[idx], dtype=torch.long).to(self.device)
        action = self.actions[idx].to(self.device)
        # Move tensors to the device
        obs_tensor = {
            "grid": torch.tensor(obs["grid"], dtype=torch.float32).to(self.device),
            "uncertainty_map": torch.tensor(obs["uncertainty_map"], dtype=torch.float32).to(self.device),
            "position": torch.tensor(obs["position"], dtype=torch.float32).to(self.device),
            "goal": torch.tensor(obs["goal"], dtype=torch.float32).to(self.device)
        }
        return obs_tensor, action

class BCActor(nn.Module):
    def __init__(self, env, use_uncertainty_map=False):
        super().__init__()
        grid_size = env.observation_space['grid'].shape[0]
        input_channels = 2 if use_uncertainty_map else 1
        self.grid_net = nn.Sequential(
            self.layer_init(nn.Linear(grid_size * grid_size * input_channels, 512)),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            self.layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        self.position_goal_net = nn.Sequential(
            self.layer_init(nn.Linear(4, 32)),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            self.layer_init(nn.Linear(128 + 32, 128)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            self.layer_init(nn.Linear(128 + 32, 128)),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            self.layer_init(nn.Linear(128 + 32, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, env.action_space.n))
        )

    """def __init__(self, env, use_uncertainty_map=False):
        super().__init__()
        grid_size = env.observation_space['grid'].shape[0]
        input_channels = 2 if use_uncertainty_map else 1
        self.grid_net = nn.Sequential(
            self.layer_init(nn.Linear(grid_size * grid_size * input_channels, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        self.position_goal_net = nn.Sequential(
            self.layer_init(nn.Linear(4, 32)),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            self.layer_init(nn.Linear(64 + 32, 128)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            self.layer_init(nn.Linear(128 + 32, 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, env.action_space.n))
        )"""

    def forward(self, x):
        # Process grid
        grid_size = x['grid'].shape[0]
        grid = x['grid'].float().unsqueeze(1)  # Add channel dimension
        if 'uncertainty_map' in x:
            uncertainty_map = x['uncertainty_map'].float().unsqueeze(1)  # Add channel dimension
            grid = torch.cat([grid, uncertainty_map], dim=1)  # Concatenate along channel dimension
        num_channels = grid.shape[1]  # Get the number of channels after concatenation
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
        out2 = self.layer2(combined2)
        combined3 = torch.cat([out2, position_goal_features], dim=1)
        logits = self.layer3(combined3)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def layer_init(self, layer, bias_const=0.0):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, bias_const)
        return layer


def collect_expert_data(env, num_episodes=10000):
    observations = []
    actions = []
    print("num_episodes: ", num_episodes)
    for episode in tqdm(range(num_episodes), desc="Collecting Expert Data"):
        obs, _ = env.reset()
        done = False
        while not done:
            current_pos = tuple(obs['position'])
            goal_pos = tuple(obs['goal'])
            planning_grid = np.where(obs['grid'] == 1, 1, 0)
            path, _ = astar(planning_grid, current_pos, goal_pos)
            if path and len(path) > 1:
                action = compute_action(current_pos, path[1])
            else:
                action = env.action_space.sample()  # Fallback to random action if no path
            observations.append(obs)
            actions.append(torch.tensor(action, dtype=torch.long))
            obs, _, done, _, _ = env.step(action)  # Advance the environment state
        del obs, path, planning_grid, current_pos, goal_pos, action  # Delete unused variables
        torch.cuda.empty_cache()  # Release unused memory
    return observations, actions


def evaluate_bc_agent(env, model, num_episodes=10, grid_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_goals_reached = 0
    total_rewards = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = 0
        goals_reached = 0

        while not done:
            with torch.no_grad():
                obs_tensor = {
                    "grid": torch.tensor(obs["grid"], dtype=torch.float32).to(device).view(-1, grid_size * grid_size),
                    "uncertainty_map": torch.tensor(obs["uncertainty_map"], dtype=torch.float32).to(device).view(-1, grid_size * grid_size),
                    "position": torch.tensor(obs["position"], dtype=torch.float32).to(device),
                    "goal": torch.tensor(obs["goal"], dtype=torch.float32).to(device)
                }
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, done, _, info = env.step(action)
            episode_rewards += reward
            if "goals_reached" in info:
                goals_reached = info["goals_reached"]

        total_goals_reached += goals_reached
        total_rewards += episode_rewards

    avg_goals_reached = total_goals_reached / num_episodes
    avg_reward = total_rewards / num_episodes
    print(f"\nEvaluation: Average Goals Reached: {avg_goals_reached}, Average Reward: {avg_reward}")


def train_bc_agent(env, model, expert_data, num_epochs=1000, batch_size=128, lr=1e-3, grid_size=10, use_uncertainty_map=False, iteration=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BCDataset(*expert_data, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # Ensure the checkpoints directory exists
    os.makedirs(f'checkpoints_{use_uncertainty_map}/{grid_size}', exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Training BC Agent"):
        for obs, action in dataloader:
            optimizer.zero_grad()
            logits = model(obs)
            loss = criterion(logits, action)
            loss.backward()
            optimizer.step()
            del obs, action, logits, loss  # Delete unused variables
            torch.cuda.empty_cache()  # Release unused memory

        # Evaluate the model every 25 epochs
        if epoch % 25 == 0:
            print(f"\Iteration {iteration}_{epoch} Evaluation:")
            evaluate_bc_agent(env, model, grid_size=grid_size)
            torch.save(model.state_dict(), f'checkpoints_{use_uncertainty_map}/{grid_size}/bc_agent_1_{iteration}_{epoch}.pth')
            print(f"Iteration {iteration}_{epoch} model saved")

    print(f"\Iteration {iteration} Evaluation:")
    evaluate_bc_agent(env, model, grid_size=grid_size)
    torch.save(model.state_dict(), f'checkpoints_{use_uncertainty_map}/{grid_size}/bc_agent_1_{iteration}.pth')
    print(f"Iteration {iteration} model saved")
    del dataset, dataloader, optimizer, criterion  # Delete unused variables
    torch.cuda.empty_cache()  # Release unused memory
    return model


def main():
    grid_size = 8
    use_uncertainty_map = True  # Toggle this to use the uncertainty map
    env = UncertaintyGridEnv(grid_size=grid_size, num_agents=1, obstacle_prob=0.2, alpha=0.1, max_steps=4 * grid_size * grid_size)
    num_iterations = 1  # Number of iterations for data collection and training
    epochs_per_iteration = 500  # Number of epochs to train in each iteration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCActor(env, use_uncertainty_map=use_uncertainty_map).to(device)
    #model.load_state_dict(torch.load(f'checkpoints_{use_uncertainty_map}/{grid_size}/bc_agent_1_99.pth'))

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}: Collecting expert data")
        expert_data = collect_expert_data(env)
        print(f"Iteration {iteration + 1}/{num_iterations}: Training BC agent")
        model = train_bc_agent(env, model, expert_data, num_epochs=epochs_per_iteration, grid_size=grid_size, use_uncertainty_map=use_uncertainty_map, iteration=iteration)


if __name__ == "__main__":
    main() 