import sys
import os

from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent

def test_telemetry():
    cfg = Config()
    cfg.training.device = "cpu"
    cfg.checkpoint.drive_sync_dir = None
    
    env = NeuroEvasionEnv(cfg)
    snake = DQNAgent(in_channels=env.obs_channels, grid_size=cfg.game.grid_size, num_actions=env.snake_num_actions, config=cfg.agent, device="cpu")
    bait  = DQNAgent(in_channels=env.obs_channels, grid_size=cfg.game.grid_size, num_actions=env.bait_num_actions, config=cfg.agent, device="cpu")
    
    print("Testing render()...")
    img = env.render()
    assert img.shape == (320, 320, 3), f"Bad image shape: {img.shape}"
    print(f"✅ Render generated shape: {img.shape}")
    
    print("Testing record_eval_episode()...")
    frames = env.record_eval_episode(snake, bait, max_steps=10)
    assert len(frames) > 0, "No frames recorded"
    assert frames[0].shape == (320, 320, 3), f"Bad frame shape: {frames[0].shape}"
    print(f"✅ Recorded {len(frames)} frames successfully")
    
    print("All tests passed.")

if __name__ == "__main__":
    test_telemetry()
