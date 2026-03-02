"""
visualization/renderer.py — Real-time game visualization with Pygame.

This renders the game grid with color-coded entities and a HUD
showing episode stats. Used in 'demo' mode to watch trained agents.

DESIGN:
    - Dark background with neon-style colors for a modern look
    - Snake rendered as a gradient from bright to dark green
    - Bait pulses with a glow effect
    - HUD shows real-time stats and scores
"""

import pygame
import numpy as np
import sys
import math
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent

# --- Color Palette ---
BLACK = (10, 10, 15)
DARK_GRAY = (25, 25, 35)
WALL_COLOR = (40, 45, 60)
GRID_LINE = (30, 30, 40)
SNAKE_HEAD = (0, 255, 120)
SNAKE_BODY = (0, 180, 80)
SNAKE_TAIL = (0, 100, 50)
BAIT_COLOR = (255, 60, 80)
BAIT_GLOW = (255, 100, 120)
TEXT_COLOR = (200, 210, 230)
HIGHLIGHT = (100, 200, 255)


class GameRenderer:
    """Renders the NeuroEvasion game with Pygame."""

    def __init__(self, grid_size: int, cell_size: int = 30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.hud_width = 280
        self.width = grid_size * cell_size + self.hud_width
        self.height = grid_size * cell_size

        pygame.init()
        pygame.display.set_caption("NeuroEvasion — Pursuit-Evasion AI")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.frame_count = 0

    def render(self, state: dict, info: dict = None) -> None:
        """Draw the current game state."""
        self.screen.fill(BLACK)
        self.frame_count += 1

        # Draw grid lines
        grid_pixel_size = self.grid_size * self.cell_size
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, self.height), 1)
            pygame.draw.line(self.screen, GRID_LINE, (0, x), (grid_pixel_size, x), 1)

        grid = state["grid"]
        snake_body = state.get("snake_body", [])
        bait_pos = state.get("bait_pos")

        # Draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] == 1:  # WALL
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                       self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)

        # Draw snake body with gradient
        if snake_body:
            for i, (r, c) in enumerate(snake_body):
                ratio = i / max(len(snake_body) - 1, 1)
                color = self._lerp_color(SNAKE_HEAD, SNAKE_TAIL, ratio)
                rect = pygame.Rect(c * self.cell_size + 1, r * self.cell_size + 1,
                                   self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

            # Snake head highlight
            hr, hc = snake_body[0]
            rect = pygame.Rect(hc * self.cell_size + 1, hr * self.cell_size + 1,
                               self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(self.screen, SNAKE_HEAD, rect, border_radius=6)

        # Draw bait with pulse effect
        if bait_pos:
            br, bc = bait_pos
            pulse = abs(math.sin(self.frame_count * 0.1)) * 4
            cx = bc * self.cell_size + self.cell_size // 2
            cy = br * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 2 + int(pulse)

            # Glow effect
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*BAIT_GLOW, 40),
                               (radius * 2, radius * 2), radius * 2)
            self.screen.blit(glow_surf, (cx - radius * 2, cy - radius * 2))

            # Core
            pygame.draw.circle(self.screen, BAIT_COLOR, (cx, cy), radius - 2)

        # Draw HUD
        self._draw_hud(state, info)

        pygame.display.flip()

    def _draw_hud(self, state: dict, info: dict) -> None:
        """Draw the heads-up display panel."""
        hud_x = self.grid_size * self.cell_size + 10

        # Panel background
        panel_rect = pygame.Rect(self.grid_size * self.cell_size, 0,
                                  self.hud_width, self.height)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)

        # Title
        title = self.font_large.render("NeuroEvasion", True, HIGHLIGHT)
        self.screen.blit(title, (hud_x, 15))

        y = 55
        lines = [
            f"Step: {state.get('step', 0)}",
            f"Done: {state.get('done', False)}",
            f"Winner: {state.get('winner', '-')}",
            "",
            f"Snake len: {len(state.get('snake_body', []))}",
        ]

        if info:
            lines.extend([
                f"Distance: {info.get('distance', '?')}",
                f"Event: {info.get('event', 'step')}",
            ])

        for line in lines:
            if line:
                surf = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(surf, (hud_x, y))
            y += 22

    def _lerp_color(self, c1, c2, t):
        """Linear interpolation between two colors."""
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def handle_events(self) -> str:
        """Handle Pygame events. Returns 'quit', 'pause', or 'continue'."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_p:
                    return "pause"
                if event.key == pygame.K_r:
                    return "restart"
        return "continue"

    def close(self) -> None:
        pygame.quit()


def run_demo(config: Config, snake_model_path: str,
             bait_model_path: str, speed: int = 10) -> None:
    """Run a visual demo with trained agents."""
    env = NeuroEvasionEnv(config)
    renderer = GameRenderer(config.game.grid_size)

    in_channels = env.obs_channels
    grid_size = config.game.grid_size

    snake_agent = DQNAgent(in_channels, grid_size, env.snake_num_actions,
                           config.agent, device="cpu")
    bait_agent = DQNAgent(in_channels, grid_size, env.bait_num_actions,
                          config.agent, device="cpu")

    snake_agent.load(snake_model_path)
    bait_agent.load(bait_model_path)

    # Set to pure exploitation (no random actions)
    snake_agent.epsilon = 0.0
    bait_agent.epsilon = 0.0

    print("Demo mode — Press Q to quit, P to pause, R to restart")

    while True:
        snake_obs, bait_obs = env.reset()
        done = False

        while not done:
            event = renderer.handle_events()
            if event == "quit":
                renderer.close()
                return
            if event == "restart":
                break
            if event == "pause":
                while renderer.handle_events() != "pause":
                    renderer.clock.tick(10)

            snake_action = snake_agent.select_action(snake_obs)
            bait_action = bait_agent.select_action(bait_obs)

            snake_obs, bait_obs, _, _, done, info = env.step(snake_action, bait_action)

            state = env.engine.get_state()
            renderer.render(state, info)
            renderer.clock.tick(speed)

    renderer.close()
