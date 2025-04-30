import numpy as np
import heapq
import time
import gym
from dynamic_env import UncertaintyGridEnv

def astar(grid, start, goal):
    """
    Compute and return a path from start to goal on grid using A*.
    grid: 2D numpy array (cells == 1 are obstacles; any other value is free)
    start, goal: tuples (row, col)

    Returns:
        A list of (row,col) tuples from start to goal, or None if no path found.
    """
    rows, cols = grid.shape
    start = tuple(start)
    goal = tuple(goal)

    def heuristic(a, b):
        # Manhattan distance heuristic.
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # If either start or goal is blocked, no path can be found.
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return None

    open_set = []
    count = 0
    # Each entry in open_set is (f_score, count, current_node)
    heapq.heappush(open_set, (heuristic(start, goal), count, start))

    came_from = dict()
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        current_f, _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct the path from goal to start
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        # Check neighbors (up, down, left, right)
        for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + delta[0], current[1] + delta[1])
            # Skip out-of-bound neighbors.
            if neighbor[0] < 0 or neighbor[0] >= rows or neighbor[1] < 0 or neighbor[1] >= cols:
                continue
            # In this planner we treat unknowns (-1) as free and ignore agent markings (2).
            if grid[neighbor[0], neighbor[1]] == 1:
                continue  # obstacle cell
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + 1  # cost to move is 1 per step
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                count += 1
                heapq.heappush(open_set, (f_score, count, neighbor))
    # No path found
    return None


def path_to_actions(path):
    """
    Convert a path (list of (row,col) tuples) to a sequence of discrete actions."
    Discrete action mapping:
        `0: up    (row-1)
        1: down  (row+1)
        2: left  (col-1)
        3: right (col+1)
    """
    actions = []
    for idx in range(1, len(path)):
        current = path[idx - 1]
        next_cell = path[idx]
        dr = next_cell[0] - current[0]
        dc = next_cell[1] - current[1]
        if (dr, dc) == (-1, 0):
            actions.append(0)  # up
        elif (dr, dc) == (1, 0):
            actions.append(1)  # down
        elif (dr, dc) == (0, -1):
            actions.append(2)  # left
        elif (dr, dc) == (0, 1):
            actions.append(3)  # right
        else:
            # Should not happen for valid adjacent cells.
            actions.append(0)
    return actions


#Helper: Process Observed Grid

def get_obstacle_grid(grid):
    """"
    Given the observed grid from the environment, create a grid for planning:
    - Cells with value 1 (obstacle) remain as 1."
    - All other cells (free, unknown, or agent positions) are treated as free (0).
    """
    obstacle_grid = np.where(grid == 1, 1, 0)
    return obstacle_grid

#Main Simulation Loop

def main():
    # Initialize your gym environment
    env = UncertaintyGridEnv(
        grid_size=10,
        num_agents=2,
        obstacle_prob=0.2,
        alpha=0.001,
        max_steps=200,
        dynamic=True,
        sensor_range=1,
        dynamic_interval=4,
        initial_full_observation=True)
    
    obs, info = env.reset()
    done = False
    total_steps = 0

    # Dictionary to store current A*-derived action list for each agent
    agent_paths = {}

    # Use the observed grid to extract obstacles (ignoring agent markings)
    current_grid = obs["grid"]
    obstacle_grid = get_obstacle_grid(current_grid)
    # Make a copy to check for changes at each step.
    previous_obstacle_grid = obstacle_grid.copy()

    # Initially plan a path for every agent.
    for i in range(env.num_agents):
        # Get the agent’s start position (from the environment’s internal state)
        start = tuple(env.agent_positions[i])
        # Extract the goal for agent_i from info (e.g., "agent_0", "agent_1")
        goal = info["agent_goals"][f"agent_{i}"]
        plan = astar(obstacle_grid, start, goal)
        if plan is not None:
            agent_paths[i] = path_to_actions(plan)
        else:
            print(f"[Init] No path found for agent {i} from {start} to {goal}.")
            agent_paths[i] = []  # fallback to an empty plan

    # Simulation loop
    while not done:
        # Get the current observed grid and extract obstacles.
        current_grid = obs["grid"]
        obstacle_grid = get_obstacle_grid(current_grid)
        
        # If obstacles have changed, replan for all agents.
        if not np.array_equal(obstacle_grid, previous_obstacle_grid):
            print("Change in obstacle grid detected! Replanning paths.")
            for i in range(env.num_agents):
                start = tuple(env.agent_positions[i])
                goal = info["agent_goals"][f"agent_{i}"]
                new_plan = astar(obstacle_grid, start, goal)
                if new_plan is not None:
                    agent_paths[i] = path_to_actions(new_plan)
                else:
                    print(f"[Replan] No path found for agent {i} from {start} to {goal}.")
                    agent_paths[i] = []
            previous_obstacle_grid = obstacle_grid.copy()
        
        # For each agent, if it has finished its current plan (or is already at goal),
        # then replan from its current position.
        for i in range(env.num_agents):
            start = tuple(env.agent_positions[i])
            goal = info["agent_goals"][f"agent_{i}"]
            # Check if the agent has reached its goal (or its plan is empty).
            if start == goal or len(agent_paths.get(i, [])) == 0:
                new_plan = astar(obstacle_grid, start, goal)
                if new_plan is not None:
                    agent_paths[i] = path_to_actions(new_plan)
                else:
                    # If no plan is found, choose a default no-move action (here: up=0).
                    agent_paths[i] = []
        
        # Choose the next action for each agent:
        actions = []
        for i in range(env.num_agents):
            if agent_paths.get(i) and len(agent_paths[i]) > 0:
                # Pop the first action from the agent's current plan.
                actions.append(agent_paths[i].pop(0))
            else:
                # If no plan, default to a no-op (here we choose action "up" i.e., 0).
                actions.append(0)
        
        # Execute the action in the environment.
        obs, rewards, done, _, info = env.step(actions)
        env.render(mode="video")
        total_steps += 1
        
        print(f"Step {total_steps} | Actions: {actions} | Rewards: {rewards}")
        print("Current Goals:", info["agent_goals"])
        # Optionally, render using your desired mode ("human" or "video")
        #time.sleep(0.1)  # slow down for visualization (optional)

    print("Episode finished.")
    env.close()


if __name__ == "__main__":
    main()