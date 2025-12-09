# scripts/train.py

import numpy as np
import os
import sys
from src.environment import Snake3DEnv
from src.agent import DQNAgent
from src.visualize import TrainingLogger, DetailedEpisodeLogger, print_training_progress

def train(n_episodes=50, batch_size=32, target_update_freq=10, save_freq=25, 
          log_actions=True, log_action_freq=5):
    """
    Train DQN agent on 3D Snake
    
    Args:
        n_episodes: Number of episodes to train (default: 50)
        batch_size: Batch size for training (default: 32)
        target_update_freq: How often to update target network (default: 10)
        save_freq: How often to save model and logs (default: 25)
        log_actions: Whether to log detailed actions (default: True)
        log_action_freq: Log actions every N episodes (default: 5)
    """
    print("="*60)
    print("TRAINING DQN AGENT ON 3D SNAKE")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Initialize environment and agent
    print("\n1. Initializing...")
    env = Snake3DEnv(grid_size=10)
    agent = DQNAgent(state_shape=(10, 10, 10), n_actions=6)
    logger = TrainingLogger('data/training_log.csv')
    
    # Initialize detailed action logger
    if log_actions:
        action_logger = DetailedEpisodeLogger('data/episode_actions.csv')
        print(f"✓ Action logging enabled (every {log_action_freq} episodes)")
    
    print(f"✓ Environment created")
    print(f"✓ Agent created")
    print(f"  - Policy network: {agent.policy_net.count_params():,} parameters")
    print(f"  - Initial epsilon: {agent.epsilon}")
    print(f"  - Memory capacity: {agent.memory.buffer.maxlen}")
    
    # Training settings
    print(f"\n2. Training settings:")
    print(f"  - Episodes: {n_episodes}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Target update frequency: every {target_update_freq} episodes")
    print(f"  - Save frequency: every {save_freq} episodes")
    if log_actions:
        print(f"  - Action logging: every {log_action_freq} episodes")
    
    # Training loop
    print(f"\n3. Starting training...\n")
    print("Ep   | Reward  | Steps | Snake | Epsilon | Memory | Avg(10)")
    print("-" * 65)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        # Decide if we should log actions this episode
        should_log_actions = log_actions and (episode % log_action_freq == 0)
        
        if should_log_actions:
            action_logger.start_episode(episode)
        
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        training_loss = None
        
        # Play one episode
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Log action details if enabled
            if should_log_actions:
                action_logger.log_step(
                    step=episode_steps,
                    action=action,
                    snake_head=env.snake[0],
                    food_pos=env.food_position,
                    reward=reward,
                    done=done or truncated
                )
            
            # Store experience
            agent.memory.push(obs, action, reward, next_obs, done)
            
            # Train if enough samples
            if len(agent.memory) >= batch_size:
                loss = agent.train_step(batch_size)
                training_loss = loss
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Check if truncated
            if truncated:
                done = True
        
        # End action logging for this episode
        if should_log_actions:
            action_logger.end_episode()
        
        # Episode finished - update agent
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % target_update_freq == 0 and episode > 0:
            agent.update_target_network()
        
        # Log episode data
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            loss=training_loss
        )
        
        # Track for statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Print progress
        if episode % max(1, n_episodes // 20) == 0 or episode == n_episodes - 1:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            snake_length = info['snake_length']
            
            print(f"{episode:4d} | {episode_reward:7.2f} | {episode_steps:5d} | "
                  f"{snake_length:5d} | {agent.epsilon:7.3f} | {len(agent.memory):6d} | "
                  f"{avg_reward:7.2f}")
        
        # Save periodically
        if episode % save_freq == 0 and episode > 0:
            logger.save()
            agent.policy_net.save(f'models/checkpoint_ep{episode}.keras')
            if log_actions:
                action_logger.save()
            print(f"\n  ✓ Saved checkpoint at episode {episode}\n")
    
    # Training complete
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Final statistics
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Memory size: {len(agent.memory)}")
    print(f"  Average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Best reward: {max(episode_rewards):.2f}")
    print(f"  Worst reward: {min(episode_rewards):.2f}")
    print(f"  Average steps (last 10): {np.mean(episode_lengths[-10:]):.1f}")
    print(f"  Max steps: {max(episode_lengths)}")
    
    # Save final model and logs
    print(f"\n4. Saving final results...")
    logger.save()
    agent.policy_net.save('models/final_model.keras')
    if log_actions:
        action_logger.save()
    print(f"✓ Model saved to models/final_model.keras")
    print(f"✓ Training log saved to data/training_log.csv")
    if log_actions:
        print(f"✓ Action log saved to data/episode_actions.csv")
        print(f"  ({len(action_logger.all_episodes)} actions logged)")
    
    # Create plots and analysis
    print(f"\n5. Creating visualizations...")
    try:
        from src.visualize import plot_training_curves, analyze_actions, visualize_episode_path
        
        # Training curves
        plot_training_curves('data/training_log.csv', 'data/training_curves.png')
        print(f"✓ Training curves saved to data/training_curves.png")
        
        # Action analysis
        if log_actions and len(action_logger.all_episodes) > 0:
            print(f"\n" + "-"*60)
            analyze_actions('data/episode_actions.csv')
            print("-"*60)
            
            # Visualize first and last logged episodes
            first_ep = 0
            last_ep = (n_episodes // log_action_freq) * log_action_freq
            if last_ep == 0:
                last_ep = n_episodes - 1
            
            print(f"\nCreating 3D path visualizations...")
            visualize_episode_path('data/episode_actions.csv', 
                                  episode_num=first_ep, 
                                  save_path=f'data/episode_{first_ep}_path.png')
            
            if last_ep != first_ep and last_ep < n_episodes:
                visualize_episode_path('data/episode_actions.csv', 
                                      episode_num=last_ep, 
                                      save_path=f'data/episode_{last_ep}_path.png')
            
            print(f"✓ Episode path visualizations saved")
    
    except ImportError as e:
        print(f"\n⚠ Visualization error: {e}")
        print(f"  Make sure matplotlib and pandas are installed")
    except Exception as e:
        print(f"\n⚠ Error during visualization: {e}")
    
    print("\n" + "="*60)
    print("✅ ALL DONE! Check data/ and models/ folders")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - models/final_model.keras")
    print(f"  - data/training_log.csv")
    if log_actions:
        print(f"  - data/episode_actions.csv")
    print(f"  - data/training_curves.png")
    if log_actions:
        print(f"  - data/episode_0_path.png")
        if last_ep != 0:
            print(f"  - data/episode_{last_ep}_path.png")
    print()
    
    return agent, logger


def quick_test():
    """Quick 20-episode test run"""
    print("🔬 Running quick test (20 episodes)...\n")
    return train(
        n_episodes=20,
        batch_size=32,
        target_update_freq=5,
        save_freq=10,
        log_actions=True,
        log_action_freq=5
    )


def standard_run():
    """Standard 50-episode training"""
    print("🎮 Running standard training (50 episodes)...\n")
    return train(
        n_episodes=50,
        batch_size=32,
        target_update_freq=10,
        save_freq=25,
        log_actions=True,
        log_action_freq=5
    )


def full_training():
    """Full 1000-episode training"""
    print("🚀 Running full training (1000 episodes)...\n")
    return train(
        n_episodes=1000,
        batch_size=32,
        target_update_freq=10,
        save_freq=100,
        log_actions=True,
        log_action_freq=10
    )


if __name__ == "__main__":
    import time
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == 'test':
            agent, logger = quick_test()
        elif arg == 'full':
            agent, logger = full_training()
        elif arg.isdigit():
            n_eps = int(arg)
            print(f"🎮 Custom training run ({n_eps} episodes)...\n")
            agent, logger = train(
                n_episodes=n_eps,
                batch_size=32,
                target_update_freq=10,
                save_freq=max(10, n_eps // 4),
                log_actions=True,
                log_action_freq=max(1, n_eps // 10)
            )
        else:
            print("Usage:")
            print("  python scripts/train.py          # Standard (50 episodes)")
            print("  python scripts/train.py test     # Quick test (20 episodes)")
            print("  python scripts/train.py full     # Full training (1000 episodes)")
            print("  python scripts/train.py 100      # Custom number of episodes")
            sys.exit(1)
    else:
        # Default: standard 50-episode run
        agent, logger = standard_run()
    
    print("\n🎉 Training session complete!")
