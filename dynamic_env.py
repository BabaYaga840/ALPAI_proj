import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

class UncertaintyGridEnv(gym.Env):
    """
    A gridworld environment with dynamic changes and an uncertainty-driven observation model.

    - The true state is held in self.true_grid (0: free, 1: obstacle).
    - When dynamic=True, every dynamic_interval steps a new obstacle is randomly added.
    - Agents only sense cells within sensor_range. Their observation (self.observed_grid)
      only shows the true state in sensed regions; unknown cells remain -1.
    - Each agent has a goal that is reset when reached.
    - Uncertainty of each cell is updated as: u = 1 - exp(-alpha*(t - t_last)).
    - The render() function supports three modes:
         "human" : prints a textual representation,
         "video" : displays/updates a Matplotlib figure,
         "none"  : no output.
    - If initial_full_observation is True, the initial (timestep 0) observation is the full grid.
    """
    def __init__(self,
                 grid_size=10,
                 num_agents=2,
                 obstacle_prob=0.2,
                 alpha=0.1,
                 max_steps=100,
                 dynamic=False,
                 sensor_range=1,
                 dynamic_interval=4,
                 initial_full_observation=False,
                 use_directional_reward=False,
                 use_uncertainty_reward=False):  # New parameter
        super(UncertaintyGridEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacle_prob = obstacle_prob
        self.alpha = alpha
        self.max_steps = max_steps
        self.dynamic = dynamic
        self.sensor_range = sensor_range
        self.dynamic_interval = dynamic_interval
        self.initial_full_observation = initial_full_observation
        self.use_directional_reward = use_directional_reward  # Store new parameter
        self.use_uncertainty_reward = use_uncertainty_reward  # Store new parameter
        self.initial_uncertainty = None  # Track initial total uncertainty
        self.uncertainty_rewards = []  # Track uncertainty rewards
        self.unc_rew_scale = 0.1

        self.time = 0
        self.goals_reached = 0  # Track number of goals reached
        print(self.grid_size)
        # Ground-truth grid (0: free, 1: obstacle)
        self.true_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Observed grid: if dynamic, unknown cells are -1 by default.
        self.observed_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        # Tracking last observation time for each cell
        self.last_observed = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.uncertainty_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Agent positions and goals (stored as lists of numpy arrays)
        self.agent_positions = []
        self.agent_goals = []

        # Each agent has 4 discrete actions: 0 (up), 1 (down), 2 (left), 3 (right)
        if num_agents == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.MultiDiscrete([4] * num_agents)
            
        # Observation: dict with "grid" (partial or full) and "uncertainty_map"
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-1, high=10, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            "uncertainty_map": spaces.Box(low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32),
            "position": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "goal": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        })

        # Attributes for video rendering.
        self.fig = None
        self.axs = None
        self.grid_im = None
        self.uncertainty_im = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.time = 0
        self.goals_reached = 0  # Reset goals reached counter
        # Build the true grid: obstacles appear with probability obstacle_prob.
        self.true_grid = (np.random.rand(self.grid_size, self.grid_size) < self.obstacle_prob).astype(np.int32)
        self.last_observed = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.uncertainty_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # If initial_full_observation is True, show the full grid at timestep 0 regardless of dynamic flag.
        if self.dynamic and not self.initial_full_observation:
            self.observed_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        else:
            self.observed_grid = self.true_grid.copy()
        
        # Place agents in free cells.
        self.agent_positions = []
        for _ in range(self.num_agents):
            pos = self._get_random_free_cell(exclusions=self.agent_positions)
            self.agent_positions.append(pos)
        
        # Allocate goals (each agent gets its own goal)
        self.agent_goals = []
        for _ in range(self.num_agents):
            goal = self._get_random_free_cell(exclusions=self.agent_positions + self.agent_goals)
            self.agent_goals.append(goal)
        
        # Update sensor readings based on agents' positions.
        if self.dynamic:
            self._update_sensor()
        
        # Reset rendering parameters.
        self.fig = None
        self.grid_im = None
        self.uncertainty_im = None
        
        self.initial_uncertainty = np.sum(self.uncertainty_map)  # Initialize initial uncertainty
        self.uncertainty_rewards = []  # Reset uncertainty rewards
        
        return self._get_observation(), self._get_info()

    def step(self, actions):
        self.time += 1
        rewards = [0 for _ in range(self.num_agents)]
        
        # Apply dynamic change every dynamic_interval steps.
        if self.dynamic and (self.time % self.dynamic_interval == 0):
            self._apply_dynamic_changes()

        # Mapping: 0 up, 1 down, 2 left, 3 right.
        direction_mapping = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }
        
        # Convert single action to list for single agent case
        if self.num_agents == 1:
            actions = [actions]
        
        for i, action in enumerate(actions):
            current_pos = self.agent_positions[i].copy()
            move = direction_mapping.get(action)
            new_pos = current_pos + move
            # Keep agent within grid bounds.
            new_pos[0] = int(np.clip(new_pos[0], 0, self.grid_size - 1))
            new_pos[1] = int(np.clip(new_pos[1], 0, self.grid_size - 1))
            
            # If destination in the true grid is an obstacle, agent stays in place.
            if self.true_grid[new_pos[0], new_pos[1]] == 1:
                new_pos = current_pos
            
            self.agent_positions[i] = new_pos
            # Mark cell as observed at current time.
            self.last_observed[new_pos[0], new_pos[1]] = self.time
            
            # For dynamic environments, update sensor reading for the agent.
            if self.dynamic:
                self._update_sensor_for_agent(new_pos)
            
            # Check goal achievement; if reached, assign a new goal.
            if np.array_equal(new_pos, self.agent_goals[i]):
                rewards[i] = 1
                self.goals_reached += 1  # Increment goals reached counter
                new_goal = self._get_random_free_cell(exclusions=self.agent_positions + self.agent_goals)
                self.agent_goals[i] = new_goal
            elif self.use_directional_reward:  # Add directional reward if enabled
                # Calculate vector from agent to goal
                goal_vector = self.agent_goals[i] - current_pos
                # Normalize vectors
                if np.linalg.norm(goal_vector) > 0:
                    goal_vector = goal_vector / np.linalg.norm(goal_vector)
                if np.linalg.norm(move) > 0:
                    move = move / np.linalg.norm(move)
                # Calculate dot product and scale reward
                directional_reward = 0.2 * np.dot(goal_vector, move)
                rewards[i] += directional_reward
        
        initial_uncertainty = np.sum(self.uncertainty_map)  # Calculate initial uncertainty
        # Update uncertainty map: u = 1 - exp(-alpha*(time - t_last))
        dt = self.time - self.last_observed
        self.uncertainty_map = 1 - np.exp(-self.alpha * dt)
        final_uncertainty = np.sum(self.uncertainty_map)  # Calculate final uncertainty
        uncertainty_change = initial_uncertainty - final_uncertainty
        if self.use_uncertainty_reward:
            uncertainty_reward = self.unc_rew_scale * uncertainty_change / (self.grid_size * self.grid_size)
            for i in range(self.num_agents):
                rewards[i] += uncertainty_reward
            self.uncertainty_rewards.append(uncertainty_reward)  # Log uncertainty reward
        
        done = self.time >= self.max_steps
        info = {
            "agent_goals": {f"agent_{i}": tuple(goal) for i, goal in enumerate(self.agent_goals)},
            "goals_reached": self.goals_reached,  # Add goals reached to info
            "uncertainty_rewards": self.uncertainty_rewards  # Log uncertainty rewards
        }
        
        # Return single reward for single agent case
        if self.num_agents == 1:
            rewards = rewards[0]
            
        return self._get_observation(), rewards, done, {}, info
    
    def reset_goals(self, reset_goal):
        for i in range(self.num_agents):
            if reset_goal[i] == 1:
                goal = self._get_random_free_cell(exclusions=self.agent_positions + self.agent_goals)
                self.agent_goals[i] = goal

    def _get_random_free_cell(self, exclusions=None):
        """
        Get a random free cell that is not in the exclusions list.
        Returns None if no free cell is available.
        """
        if exclusions is None:
            exclusions = []
        
        # Get all free cells
        free_cells = np.argwhere(self.true_grid == 0)
        
        # Remove excluded cells
        if exclusions:
            exclusions = np.array(exclusions)
            mask = ~np.any(np.all(free_cells[:, None] == exclusions, axis=2), axis=1)
            free_cells = free_cells[mask]
        
        if len(free_cells) > 0:
            # Choose a random free cell
            idx = np.random.randint(len(free_cells))
            return free_cells[idx]
        else:
            # If no free cells are available, return None
            return None

    def _apply_dynamic_changes(self):
        """
        Dynamically modify the environment's true grid.
        With 50% probability, add an obstacle in a free cell not occupied by any agent.
        With 50% probability, remove an obstacle from a cell (again, only if it's not occupied by an agent).
        These changes are applied only to the true grid; the observed grid is updated later when agents sense their surroundings.
        """
        if np.random.random() < 0.5:
            # Add an obstacle
            free_cells = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.true_grid[i, j] == 0:
                        cell = np.array([i, j])
                        if not any(np.array_equal(cell, agent) for agent in self.agent_positions):
                            free_cells.append((i, j))
            if free_cells:
                i, j = free_cells[np.random.choice(len(free_cells))]
                self.true_grid[i, j] = 1
        else:
            # Remove an obstacle
            obstacle_cells = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.true_grid[i, j] == 1:
                        cell = np.array([i, j])
                        if not any(np.array_equal(cell, agent) for agent in self.agent_positions):
                            obstacle_cells.append((i, j))
            if obstacle_cells:
                i, j = obstacle_cells[np.random.choice(len(obstacle_cells))]
                self.true_grid[i, j] = 0

    def _update_sensor(self):
        """Update the observed grid based on all agents' sensor ranges."""
        for pos in self.agent_positions:
            self._update_sensor_for_agent(pos)

    def _update_sensor_for_agent(self, pos):
        """Update the observed grid for one agent up to sensor_range."""
        x, y = pos[0], pos[1]
        x_low = max(0, x - self.sensor_range)
        x_high = min(self.grid_size, x + self.sensor_range + 1)
        y_low = max(0, y - self.sensor_range)
        y_high = min(self.grid_size, y + self.sensor_range + 1)
        self.observed_grid[x_low:x_high, y_low:y_high] = self.true_grid[x_low:x_high, y_low:y_high]

    def _get_observation(self):
        """
        Returns the observation:
         - For dynamic environments, use observed_grid; otherwise, true_grid.
         - Overlay agents (mark with 2) onto the grid.
         - Include the uncertainty_map.
         - Include agent position and goal.
        """
        if self.dynamic:
            grid_obs = self.observed_grid.copy()
        else:
            grid_obs = self.true_grid.copy()
            
        for pos in self.agent_positions:
            grid_obs[pos[0], pos[1]] = 2
            
        # For single agent, return the first position and goal directly
        if self.num_agents == 1:
            return {
                "grid": grid_obs,
                "uncertainty_map": self.uncertainty_map.copy(),
                "position": np.array(self.agent_positions[0], dtype=np.int32),
                "goal": np.array(self.agent_goals[0], dtype=np.int32)
            }
        # For multi-agent, return lists of positions and goals
        else:
            return {
                "grid": grid_obs,
                "uncertainty_map": self.uncertainty_map.copy(),
                "position": np.array(self.agent_positions, dtype=np.int32),
                "goal": np.array(self.agent_goals, dtype=np.int32)
            }

    def _get_info(self):
        """Return information mapping agent IDs to their current goals."""
        return {"agent_goals": {f"agent_{i}": tuple(goal) for i, goal in enumerate(self.agent_goals)},
                "goals_reached": self.goals_reached,
                "uncertainty_rewards": self.uncertainty_rewards}

    def render(self, mode="human", save_frame=False, frame_dir="frames", step=0):
        """
        Render the environment.
        Modes:
        - "human": Prints the observed grid and uncertainty map.
        - "video": Displays/updates a Matplotlib window showing the observed grid (with agents overlaid in red
                    and agent goals overlaid in blue) and the uncertainty map.
        - "none" : Produces no output.
        - "save": Saves the current frame as an image file in the specified directory.
        """
        obs = self._get_observation()
        grid = obs["grid"]
        uncertainty = obs["uncertainty_map"]

        if mode == "human":
            print("Observed Grid (Unknown=-1, Free=0, Obstacle=1, Agent=2):")
            print(grid)
            print("Uncertainty Map:")
            print(np.round(uncertainty, 2))
        elif mode == "video":
            if self.fig is None:
                self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
                # Left subplot: the observed grid using a gray colormap.
                self.grid_im = self.axs[0].imshow(grid, cmap='gray', vmin=-1, vmax=2)
                self.axs[0].set_title("Observed Grid")
                # Overlay agent positions as red markers.
                agent_positions = np.array(self.agent_positions)
                if agent_positions.size > 0:
                    # Note: imshow uses (row, col) coordinates; scatter expects (x=col, y=row)
                    self.agent_scatter = self.axs[0].scatter(agent_positions[:, 1],
                                                            agent_positions[:, 0],
                                                            c='red', marker='o', s=100)
                else:
                    self.agent_scatter = None
                # Overlay agent goals as blue markers.
                goal_positions = np.array(self.agent_goals)
                if goal_positions.size > 0:
                    self.goal_scatter = self.axs[0].scatter(goal_positions[:, 1],
                                                            goal_positions[:, 0],
                                                            c='blue', marker='x', s=100)
                else:
                    self.goal_scatter = None

                # Right subplot: the uncertainty map.
                self.uncertainty_im = self.axs[1].imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
                self.axs[1].set_title("Uncertainty Map")
                plt.ion()
                plt.show()
            else:
                self.grid_im.set_data(grid)
                self.uncertainty_im.set_data(uncertainty)
                # Update agent positions
                agent_positions = np.array(self.agent_positions)
                if agent_positions.size > 0 and self.agent_scatter is not None:
                    self.agent_scatter.set_offsets(np.c_[agent_positions[:, 1], agent_positions[:, 0]])
                # Update goal positions
                goal_positions = np.array(self.agent_goals)
                if goal_positions.size > 0 and self.goal_scatter is not None:
                    self.goal_scatter.set_offsets(np.c_[goal_positions[:, 1], goal_positions[:, 0]])
                for ax in self.axs:
                    ax.relim()
                    ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)
        elif mode == "save":
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            if self.fig is None:
                self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
                self.grid_im = self.axs[0].imshow(grid, cmap='gray', vmin=-1, vmax=2)
                self.axs[0].set_title("Observed Grid")
                agent_positions = np.array(self.agent_positions)
                if agent_positions.size > 0:
                    self.agent_scatter = self.axs[0].scatter(agent_positions[:, 1],
                                                            agent_positions[:, 0],
                                                            c='red', marker='o', s=100)
                else:
                    self.agent_scatter = None
                goal_positions = np.array(self.agent_goals)
                if goal_positions.size > 0:
                    self.goal_scatter = self.axs[0].scatter(goal_positions[:, 1],
                                                            goal_positions[:, 0],
                                                            c='blue', marker='x', s=100)
                else:
                    self.goal_scatter = None
                self.uncertainty_im = self.axs[1].imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
                self.axs[1].set_title("Uncertainty Map")
            else:
                self.grid_im.set_data(grid)
                self.uncertainty_im.set_data(uncertainty)
                agent_positions = np.array(self.agent_positions)
                if agent_positions.size > 0 and self.agent_scatter is not None:
                    self.agent_scatter.set_offsets(np.c_[agent_positions[:, 1], agent_positions[:, 0]])
                goal_positions = np.array(self.agent_goals)
                if goal_positions.size > 0 and self.goal_scatter is not None:
                    self.goal_scatter.set_offsets(np.c_[goal_positions[:, 1], goal_positions[:, 0]])
                for ax in self.axs:
                    ax.relim()
                    ax.autoscale_view()
            self.fig.savefig(f"{frame_dir}/frame_{step}.png")
        elif mode == "none":
            pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        if self.initial_uncertainty is not None:
            final_uncertainty = np.sum(self.uncertainty_map)
            print(f"Total Uncertainty Change: {final_uncertainty - self.initial_uncertainty}")  # Log total uncertainty change

if __name__ == "__main__":
    # Example usage:
    env = UncertaintyGridEnv(grid_size=10,
                             num_agents=2,
                             obstacle_prob=0.2,
                             alpha=0.001,
                             max_steps=500,
                             dynamic=True,
                             sensor_range=1,
                             dynamic_interval=4,
                             initial_full_observation=True,
                             use_directional_reward=True,
                             use_uncertainty_reward=True)  # Set new variable here
    obs, info = env.reset()
    print("Initial Observation:")
    print(obs)
    print("Info:", info)
    
    for step in range(500):
        actions = env.action_space.sample()
        print("Actions:", actions)
        obs, rewards, done, _, info = env.step(actions)
        env.render(mode="video")  # Try "human", "video", or "none"
        print(f"Step {step}, Rewards: {rewards}")
        if done:
            break
    env.close()
