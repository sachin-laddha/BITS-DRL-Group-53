import gym
from gym.envs.toy_text import FrozenLakeEnv
import numpy as np

# Custom environment for the Treasure Hunt problem
class FrozenLakeTreasureEnv(FrozenLakeEnv):
    """
    Custom environment for the Treasure Hunt problem.
    Inherits from FrozenLakeNotSlippery-v0 and modifies it to include treasures.
    """
    def __init__(self):
        super().__init__(desc=np.asarray([
            [b'S', b'F', b'F', b'H', b'T'],
            [b'F', b'H', b'F', b'F', b'F'],
            [b'F', b'F', b'F', b'T', b'F'],
            [b'T', b'F', b'H', b'F', b'F'],
            [b'F', b'F', b'F', b'F', b'G']
        ], dtype='|S1'), is_slippery=False)

        # Initialize the treasure locations
        self.treasure_locations = [(0, 4), (2, 3), (3, 0)]  # Coordinates of treasures
        self.treasures_collected = set()

        self.nS = self.observation_space.n  # number of states (number of unique positions)
        self.nA = self.action_space.n  # number of actions (4)
        self.ncol = 5  # Number of columns in the grid

    def reset(self):
        """Reset environment to start state."""
        self.s = 0  # Start state (state index in the flattened state space)
        self.treasures_collected = set()  # Reset collected treasures
        return self.s

    def step(self, a):
        """Take action and return new state, reward, done flag, and info."""
        self.lastaction = a
        result = super().step(a)

        state, reward, done, info = result[:4]  # Unpacking the result as we know it returns 4 values

        # Check if treasure is collected
        state_2d = (state // self.ncol, state % self.ncol)  # Convert flattened state back to 2D grid coordinates
        if state_2d in self.treasure_locations:
            reward += 5  # Bonus reward for collecting treasure
            self.treasures_collected.add(state_2d)
            self._update_desc()  # Update environment description to remove collected treasure

        return state, reward, done, info

    def _update_desc(self):
        """Update the environment description to remove collected treasures."""
        desc = self.desc.tolist()
        for treasure in self.treasures_collected:
            row, col = treasure
            desc[row][col] = b'F'  # Replace treasure tile with frozen tile
        self.desc = np.asarray(desc, dtype='|S1')


# Value Iteration Algorithm
def value_iteration(env, gamma=0.9, theta=1e-8):
    """
    Performs value iteration to find the optimal value function and policy.

    Args:
        env: The environment.
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        V: Optimal value function.
        policy: Optimal policy (array of actions).
    """
    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n, dtype=int)

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            action_values = np.zeros(env.action_space.n)
            actions = env.P[s]  # Transition probabilities for state s
            for a in range(env.action_space.n):
                action_value = 0
                for prob, next_state, reward, done in actions[a]:
                    action_value += prob * (reward + gamma * V[next_state])
                action_values[a] = action_value
            best_action = np.argmax(action_values)
            V[s] = action_values[best_action]
            policy[s] = best_action
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break

    return V, policy


# Policy Improvement Function
def policy_improvement(env, V):
    """
    Computes the optimal policy given the value function.

    Args:
        env: The environment.
        V: Value function.

    Returns:
        policy: Optimal policy (array of actions).
    """
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        actions = env.P[s]
        action_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in actions[a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)
    return policy


# Print the Optimal Value Function
def print_value_function(V, env):
    """
    Prints the value function in a grid format.
    """
    print("Optimal Value Function:")
    map_size = int(np.sqrt(env.observation_space.n))  # Map size based on number of states
    for i in range(map_size):
        for j in range(map_size):
            print("{:.2f}".format(V[i * map_size + j]), end=" ")
        print()


# Visualization of the learned optimal policy
def visualize_policy(policy, map_size):
    """
    Visualizes the learned optimal policy on the grid.
    """
    actions = ["<", "v", ">", "^"]  # Map action indices to arrows
    policy_grid = policy.reshape(map_size, map_size)
    print("Optimal Policy:")
    for row in policy_grid:
        print(" ".join([actions[action] for action in row]))


def evaluate_policy(env, policy, num_episodes=100):
    """
    Evaluates the policy by running multiple episodes and calculating the average reward.

    Args:
        env: The environment.
        policy: The policy to evaluate.
        num_episodes: Number of episodes to run.

    Returns:
        Average reward over the episodes.
    """
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        # Handle both single value and tuple returns from reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False
        while not done:
            action = policy[int(state)]
            # Handle both 4-value and 5-value returns from step()
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

# Evaluate the policy
def evaluate_policy_old(env, policy, num_episodes=100):
    """
    Evaluates the policy by running multiple episodes and calculating the average reward.

    Args:
        env: The environment.
        policy: The policy to evaluate.
        num_episodes: Number of episodes to run.

    Returns:
        Average reward over the episodes.
    """
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = policy[state]  # Use flattened index to access the policy
            next_state, reward, done, info = env.step(action)  # Correct unpacking of result
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


# Main Execution
if __name__ == "__main__":
    env = FrozenLakeTreasureEnv()

    # Run value iteration
    V, policy = value_iteration(env)

    # Print the optimal value function (V*)
    print_value_function(V, env)
    # Visualizing the learned optimal policy
    visualize_policy(policy, 5)  # 5x5 grid

    reward_with_treasures = evaluate_policy(env, policy)
    print("Average Reward with Treasures:", reward_with_treasures)

    # Evaluate the policy without treasures
    env_no_treasures = FrozenLakeEnv(desc=np.asarray([
        [b'S', b'F', b'F', b'F', b'F'],
        [b'F', b'H', b'F', b'F', b'F'],
        [b'F', b'F', b'F', b'F', b'F'],
        [b'F', b'F', b'H', b'F', b'F'],
        [b'F', b'F', b'F', b'F', b'G']
    ], dtype='|S1'), is_slippery=False)

    V_no_treasures, policy_no_treasures = value_iteration(env_no_treasures)
    reward_no_treasures = evaluate_policy(env_no_treasures, policy_no_treasures)
    print("Average Reward without Treasures:", reward_no_treasures)