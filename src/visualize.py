# src/visualize.py

import csv
import os

class TrainingLogger:
    """
    Logs training progress to CSV file
    """
    
    def __init__(self, filename='data/training_log.csv'):
        self.filename = filename
        self.data = []
    
    def log_episode(self, episode, reward, steps, epsilon, loss=None):
        """Record one episode's data"""
        self.data.append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'loss': loss if loss is not None else ''
        })
    
    def save(self):
        """Save all logged data to CSV file"""
        if not self.data:
            return
        
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w', newline='') as f:
            fieldnames = ['episode', 'reward', 'steps', 'epsilon', 'loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"✓ Saved training log to {self.filename}")
    
    def __len__(self):
        return len(self.data)


class DetailedEpisodeLogger:
    """
    Logs every action in episodes for detailed analysis
    """
    
    def __init__(self, filename='data/episode_actions.csv'):
        self.filename = filename
        self.current_episode = []
        self.all_episodes = []
        self.episode_num = 0
        
        self.action_names = {
            0: '+X', 1: '-X',
            2: '+Y', 3: '-Y', 
            4: '+Z', 5: '-Z'
        }
    
    def start_episode(self, episode_num):
        """Start logging a new episode"""
        self.current_episode = []
        self.episode_num = episode_num
    
    def log_step(self, step, action, snake_head, food_pos, reward, done):
        """Log one step/action"""
        self.current_episode.append({
            'episode': self.episode_num,
            'step': step,
            'action': action,
            'action_name': self.action_names[action],
            'head_x': snake_head[0],
            'head_y': snake_head[1],
            'head_z': snake_head[2],
            'food_x': food_pos[0],
            'food_y': food_pos[1],
            'food_z': food_pos[2],
            'reward': reward,
            'done': done
        })
    
    def end_episode(self):
        """Finish logging current episode"""
        self.all_episodes.extend(self.current_episode)
        self.current_episode = []
    
    def save(self):
        """Save all logged actions to CSV"""
        if not self.all_episodes:
            return
        
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w', newline='') as f:
            fieldnames = ['episode', 'step', 'action', 'action_name',
                         'head_x', 'head_y', 'head_z',
                         'food_x', 'food_y', 'food_z',
                         'reward', 'done']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_episodes)
        
        print(f"✓ Saved {len(self.all_episodes)} actions to {self.filename}")
    
    def __len__(self):
        return len(self.all_episodes)


def print_training_progress(episode, reward, steps, epsilon, memory_size, avg_reward=None):
    """Print formatted training progress line"""
    line = (f"Ep {episode:4d} | "
            f"R: {reward:7.2f} | "
            f"Steps: {steps:3d} | "
            f"ε: {epsilon:.3f} | "
            f"Mem: {memory_size:5d}")
    
    if avg_reward is not None:
        line += f" | Avg: {avg_reward:7.2f}"
    
    print(line)


# Placeholder functions that train.py tries to import
def plot_training_curves(csv_file, save_path):
    """Placeholder - Person 3 will implement"""
    print("⚠ plot_training_curves not implemented yet")

def analyze_actions(csv_file):
    """Placeholder - Person 3 will implement"""
    print("⚠ analyze_actions not implemented yet")

def visualize_episode_path(csv_file, episode_num, save_path):
    """Placeholder - Person 3 will implement"""
    print("⚠ visualize_episode_path not implemented yet")
