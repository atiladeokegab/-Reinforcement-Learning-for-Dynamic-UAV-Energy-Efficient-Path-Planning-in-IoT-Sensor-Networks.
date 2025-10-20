"""
Visualize trained agent policy

Run after training to create visualization of agent behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import os

from src.environment.uav_env import UAVEnvironment
from src.agents.q_learning_implementation.q_learning_agent import QLearningAgent


def save_episode_frames(env, agent, num_steps=100, output_dir='visualization'):
    """
    Run one episode and save frames as images.

    Args:
        env: Environment
        agent: Trained agent
        num_steps: Maximum steps
        output_dir: Where to save frames
    """
    os.makedirs(output_dir, exist_ok=True)

    obs, info = env.reset()
    state = agent.discretize_state(obs, grid_size=env.grid_size)

    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT']

    for step in range(num_steps):
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw grid
        for i in range(env.grid_size[0] + 1):
            ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

        # Draw sensors
        for sensor in env.sensors:
            x, y = sensor.position

            if sensor.data_buffer <= 0:
                color = 'green'
                alpha = 0.5
            elif sensor.data_buffer < sensor.max_buffer_size * 0.5:
                color = 'yellow'
                alpha = 0.7
            else:
                color = 'blue'
                alpha = 0.9

            circle = patches.Circle((x, y), 0.4, color=color, alpha=alpha,
                                    linewidth=2, edgecolor='black')
            ax.add_patch(circle)

            ax.text(x, y, f'{sensor.sensor_id}',
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white')

        # Draw UAV
        uav_x, uav_y = env.uav.position
        uav_marker = patches.FancyBboxPatch((uav_x - 0.3, uav_y - 0.3), 0.6, 0.6,
                                            boxstyle="round,pad=0.1",
                                            edgecolor='red', facecolor='orange',
                                            linewidth=3)
        ax.add_patch(uav_marker)
        ax.text(uav_x, uav_y, '✈', ha='center', va='center',
                fontweight='bold', fontsize=16, color='white')

        # Settings
        ax.set_xlim(-1, env.grid_size[0])
        ax.set_ylim(-1, env.grid_size[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')

        # Title with status
        title = f'Step: {env.current_step}/{env.max_steps} | '
        title += f'Battery: {env.uav.battery:.1f}Wh ({env.uav.get_battery_percentage():.0f}%) | '
        title += f'Collected: {len(env.sensors_visited)}/{env.num_sensors}\n'
        title += f'Action: {action_names[agent.select_action(state, training=False)]} | '
        title += f'Total Reward: {env.total_reward:.1f}'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # Legend
        legend_elements = [
            Patch(facecolor='blue', alpha=0.9, label='Sensor: Full Buffer'),
            Patch(facecolor='yellow', alpha=0.7, label='Sensor: Partial'),
            Patch(facecolor='green', alpha=0.5, label='Sensor: Collected'),
            Patch(facecolor='orange', edgecolor='red', label='UAV', linewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                  framealpha=0.9)

        # Save frame
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frame_{step:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Take action
        action = agent.select_action(state, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        state = agent.discretize_state(obs, grid_size=env.grid_size)

        if step % 10 == 0:
            print(f"Saved frame {step}/{num_steps}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break

    print(f"\n✓ Saved {step + 1} frames to {output_dir}/")
    return step + 1


def create_video(input_dir='visualization', output_file='trained_agent.mp4', fps=5):
    """
    Create MP4 video from saved frames.

    Requires ffmpeg: sudo apt install ffmpeg
    """
    import subprocess

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', f'{input_dir}/frame_*.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        output_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Created video: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create video: {e}")
        print("Install ffmpeg: sudo apt install ffmpeg")
        return False
    except FileNotFoundError:
        print("✗ ffmpeg not found. Install: sudo apt install ffmpeg")
        return False


def main():
    """Visualize trained agent."""
    print("=" * 70)
    print("Visualizing Trained Q-Learning Agent")
    print("=" * 70)
    print()

    # Create environment (no rendering)
    print("Creating environment...")
    env = UAVEnvironment(
        grid_size=(20, 20),
        num_sensors=20,
        max_steps=200,
        rssi_threshold=-80.0,
        render_mode=None  # No live rendering
    )
    print("✓ Environment created")

    # Load trained agent
    print("\nLoading trained agent...")
    agent = QLearningAgent(num_actions=5)

    try:
        agent.load("checkpoints/best_agent.pkl")
        print(f"✓ Loaded agent with Q-table size: {len(agent.q_table)}")
    except FileNotFoundError:
        print("✗ No trained agent found at checkpoints/best_agent.pkl")
        print("Train an agent first using: python train.py")
        return

    print()

    # Save frames
    print("Running episode and saving frames...")
    num_frames = save_episode_frames(env, agent, num_steps=200,
                                     output_dir='visualization_frames')

    print()

    # Create video
    print("Creating video...")
    if create_video('visualization_frames', 'trained_agent.mp4', fps=5):
        print("\n" + "=" * 70)
        print("✓ Visualization complete!")
        print("=" * 70)
        print(f"\nView frames: ls visualization_frames/")
        print(f"View video: explorer.exe trained_agent.mp4")
        print(f"Or copy to Windows: cp trained_agent.mp4 /mnt/c/Users/okeat/Desktop/")
    else:
        print("\n" + "=" * 70)
        print("✓ Frames saved (video creation failed)")
        print("=" * 70)
        print(f"\nView frames in: visualization_frames/")
        print(f"Copy to Windows: cp -r visualization_frames /mnt/c/Users/okeat/Desktop/")


if __name__ == "__main__":
    main()
