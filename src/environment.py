import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random

class Snake3DEnv(gym.Env):
    def __init__(self, grid_size = 10):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low = 0,
            high = 3,
            shape=(grid_size, grid_size, grid_size),
            dtype=np.uint8
        )
        self.action_to_direction = {
            0: (1, 0, 0),   # +X
            1: (-1, 0, 0),  # -X
            2: (0, 1, 0),   # +Y
            3: (0, -1, 0),  # -Y
            4: (0, 0, 1),   # +Z
            5: (0, 0, -1)   # -Z
        }
        self.opposite_actions = {
            0: 1, 1: 0,  # +X <-> -X
            2: 3, 3: 2,  # +Y <-> -Y
            4: 5, 5: 4   # +Z <-> -Z
        }
        self.snake = None
        self.food_position = None
        self.current_direction = None
        self.last_action = None
        self.steps = 0
        self.max_steps = 1000  

    def reset(self, seed=None, option=None):
        super().reset(seed=seed)
        center = self.grid_size // 2
        self.snake = deque([
            (center, center, center),
            (center, center, center -1),
            (center, center, center - 2)
        ])

        self.current_direction = (0,0,1)
        self.last_action = 4

        self._spawn_food()

        self.steps = 0

        observation = self._get_observation()
        info = {
                'snake_length': len(self.snake),
                'steps': self.steps
        }

        return observation,info


    def _calculate_distance(self, pos1, pos2):
        return np.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2 +
            (pos1[2] - pos2[2])**2
        )

    def step(self, action):
        self.steps += 1
        
        old_head = self.snake[0]
        old_distance = np.sqrt(
            (old_head[0] - self.food_position[0])**2 +
            (old_head[1] - self.food_position[1])**2 +
            (old_head[2] - self.food_position[2])**2
        )
        
        if action == self.opposite_actions[self.last_action]:
            action = self.last_action
        
        self.current_direction = self.action_to_direction[action]
        self.last_action = action
        
        head = self.snake[0]
        new_head = (
            head[0] + self.current_direction[0],
            head[1] + self.current_direction[1],
            head[2] + self.current_direction[2]
        )
        
        new_distance = np.sqrt(
            (new_head[0] - self.food_position[0])**2 +
            (new_head[1] - self.food_position[1])**2 +
            (new_head[2] - self.food_position[2])**2
        )
        
        distance_reward = (old_distance - new_distance) * 1.0  # Reward for getting closer
        
        terminated = False
        
        if not self._is_valid_position(new_head):
            terminated = True
            reward = -10  # Less harsh than before
        
        elif new_head in self.snake:
            terminated = True
            reward = -10  # Less harsh than before
        
        elif new_head == self.food_position:
            reward = 100 + distance_reward  # Big reward for food
            self.snake.appendleft(new_head)
            self._spawn_food()
        
        else:
            reward = -0.01 + distance_reward  # Small step penalty + distance reward
            self.snake.appendleft(new_head)
            self.snake.pop()  # Remove tail
        
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        
        info = {
            'snake_length': len(self.snake),
            'steps': self.steps,
            'food_eaten': reward > 50,  # Changed condition since reward structure changed
            'distance_to_food': new_distance
        }
        
        return observation, reward, terminated, truncated, info    


    def _is_valid_position(self, position):
        x, y, z = position
        return (0 <= x < self.grid_size and
                0 <= y < self.grid_size and
                0 <= z < self.grid_size)

    def _spawn_food(self):
        while True:
            position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )

            if position not in self.snake:
                self.food_position = position
                break

    def _get_observation(self):
        grid = np.zeros(
            (self.grid_size, self.grid_size, self.grid_size),
            dtype = np.uint8
        )

        for segment in list(self.snake)[1:]:
            grid[segment] = 1

        head = self.snake[0]
        grid[head] = 3

        grid[self.food_position] = 2

        return grid

