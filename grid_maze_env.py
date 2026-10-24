"""Custom Gymnasium environment: Grid Maze.

The agent starts in a random cell "S" and must reach a random goal "G"
while avoiding randomly placed bad cells "X".

Observation : int array [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, ...]
              (8 integers for the default 1 goal + 2 bad cells)
Actions     : 0=right, 1=up, 2=left, 3=down (deterministic; walls = stay put)
Reward      : +1 on reaching a goal, -1 on reaching a bad cell, 0 otherwise
Termination : reaching a goal or a bad cell. Truncation after `max_steps`.

Coordinates: x = column (0 is left), y = row (0 is top). "up" decreases y.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# (dx, dy) for: right, up, left, down
ACTION_DELTAS = np.array([(1, 0), (0, -1), (-1, 0), (0, 1)])


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, n_goals=1, n_bad=2, max_steps=None):
        assert size * size >= 1 + n_goals + n_bad, "grid too small for all cells"
        self.size = size
        self.n_goals = n_goals
        self.n_bad = n_bad
        self.max_steps = max_steps or 4 * size * size

        n_coords = 2 * (1 + n_goals + n_bad)  # 8 for the default setup
        self.observation_space = spaces.MultiDiscrete([size] * n_coords)
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.cell_size = max(2, 500 // size)  # keeps frames ~500px (even dims)
        self.window = None
        self.clock = None

        self._agent = None
        self._start = None
        self._goals = []
        self._bads = []
        self._steps = 0

    # ------------------------------------------------------------------ utils
    def _obs(self):
        cells = [self._agent] + self._goals + self._bads
        return np.array([c for xy in cells for c in xy], dtype=np.int64)

    def _info(self):
        return {"steps": self._steps}

    def _idx(self, xy):
        return int(xy[1]) * self.size + int(xy[0])

    def _xy(self, idx):
        return (idx % self.size, idx // self.size)

    # -------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "agent" in options:  # fixed layout (handy for experiments / demos)
            self._agent = tuple(options["agent"])
            self._goals = [tuple(g) for g in options["goals"]]
            self._bads = [tuple(b) for b in options["bads"]]
        else:
            n_cells = 1 + self.n_goals + self.n_bad
            picks = self.np_random.choice(self.size**2, size=n_cells, replace=False)
            cells = [self._xy(int(p)) for p in picks]  # all distinct
            self._agent = cells[0]
            self._goals = cells[1 : 1 + self.n_goals]
            self._bads = cells[1 + self.n_goals :]

        self._start = self._agent
        self._steps = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        dx, dy = ACTION_DELTAS[int(action)]
        x = int(np.clip(self._agent[0] + dx, 0, self.size - 1))
        y = int(np.clip(self._agent[1] + dy, 0, self.size - 1))
        self._agent = (x, y)
        self._steps += 1

        reward, terminated = 0.0, False
        if self._agent in self._goals:
            reward, terminated = 1.0, True
        elif self._agent in self._bads:
            reward, terminated = -1.0, True
        
        truncated = (not terminated) and self._steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    # ------------------------------------------------- model for DP planning
    def get_model(self):
        """Return the (known) deterministic dynamics of the CURRENT maze layout.

        States are the agent's cell index (y * size + x), so there are
        size*size states. Returns:
            next_state[s, a] : int   resulting cell index
            reward[s, a]     : float reward for that transition
            terminal[s]      : bool  True for goal / bad cells (absorbing)
        """
        n = self.size**2
        s = np.arange(n)
        xs, ys = s % self.size, s // self.size
        next_state = np.empty((n, 4), dtype=np.int64)

        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            nx = np.clip(xs + dx, 0, self.size - 1)
            ny = np.clip(ys + dy, 0, self.size - 1)
            next_state[:, a] = ny * self.size + nx

        goal_idx = [self._idx(g) for g in self._goals]
        bad_idx = [self._idx(b) for b in self._bads]
        reward = np.zeros((n, 4))
        reward[np.isin(next_state, goal_idx)] = 1.0
        reward[np.isin(next_state, bad_idx)] = -1.0

        terminal = np.zeros(n, dtype=bool)
        terminal[goal_idx + bad_idx] = True
        return next_state, reward, terminal

    # ------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _render_frame(self):
        import pygame

        cs, W = self.cell_size, self.size * self.cell_size
        if not pygame.font.get_init():
            pygame.font.init()

        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((W, W))
            pygame.display.set_caption("Grid Maze")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((W, W))
        canvas.fill((255, 255, 255))
        font = pygame.font.SysFont(None, max(10, int(cs * 0.6)))

        def cell_rect(xy):
            return pygame.Rect(xy[0] * cs, xy[1] * cs, cs, cs)

        def label(xy, text):
            img = font.render(text, True, (0, 0, 0))
            canvas.blit(img, img.get_rect(center=cell_rect(xy).center))

        pygame.draw.rect(canvas, (255, 235, 59), cell_rect(self._start))

        for g in self._goals:
            pygame.draw.rect(canvas, (0, 160, 80), cell_rect(g))
        for b in self._bads:
            pygame.draw.rect(canvas, (220, 0, 0), cell_rect(b))
        
        if cs >= 20:
            label(self._start, "S")
            for g in self._goals:
                label(g, "G")
            for b in self._bads:
                label(b, "X")

        # agent
        pygame.draw.circle(canvas, (30, 90, 220), cell_rect(self._agent).center, int(cs * 0.3))

        # grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, i * cs), (W, i * cs), 1)
            pygame.draw.line(canvas, (0, 0, 0), (i * cs, 0), (i * cs, W), 1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None


gym.register(id="GridMaze-v0", entry_point=GridMazeEnv)