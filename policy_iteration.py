"""Policy Iteration (dynamic programming) for the Grid Maze.

Because the maze layout changes every episode, the agent plans on each new
layout: the DP state is the agent's cell (size*size states) and the goal/bad
cells are part of the (fixed-during-the-episode) model.
"""
import time
import numpy as np


class PolicyIterationAgent:
    def __init__(self, gamma=0.99, theta=1e-8, exact_eval=False, seed=0):
        self.gamma = gamma
        self.theta = theta
        self.exact_eval = exact_eval  # True: solve linear system instead of sweeping
        self.rng = np.random.default_rng(seed)
        self.policy = None
        self.V = None
        self.last_stats = {}
        self._cache = {}  # layout -> (policy, V, stats); same maze is planned once

    # ---------------------------------------------------------------- planning
    def plan(self, env, layout_key=None):
        """Run policy iteration on the env's current layout (env = unwrapped)."""
        if layout_key is not None and layout_key in self._cache:
            self.policy, self.V, self.last_stats = self._cache[layout_key]
            return self.last_stats

        t0 = time.perf_counter()
        NS, R, terminal = env.get_model()
        n = NS.shape[0]
        idx = np.arange(n)

        policy = self.rng.integers(0, 4, size=n)  # arbitrary initial policy
        V = np.zeros(n)
        iterations, sweeps = 0, 0

        while True:
            iterations += 1
            # 1) policy evaluation
            if self.exact_eval:
                V = self._evaluate_exact(policy, NS, R, terminal)
            else:
                V, k = self._evaluate_iterative(policy, V, NS, R, terminal)
                sweeps += k

            # 2) policy improvement (greedy w.r.t. V)
            Q = R + self.gamma * V[NS]  # (n, 4); V[terminal] == 0
            best = Q.max(axis=1)
            keep = Q[idx, policy] >= best - 1e-10  # don't flip between ties
            new_policy = np.where(keep, policy, Q.argmax(axis=1))
            new_policy[terminal] = 0

            if np.array_equal(new_policy, policy):  # policy stable -> optimal
                break
            policy = new_policy

        self.policy, self.V = policy, V
        self.last_stats = {
            "iterations": iterations,  # number of evaluation+improvement rounds
            "eval_sweeps": sweeps,
            "time_s": time.perf_counter() - t0,
            "n_states": n,
        }

        if layout_key is not None:
            self._cache[layout_key] = (self.policy, self.V, self.last_stats)
        
        return self.last_stats

    def _evaluate_iterative(self, policy, V, NS, R, terminal):
        idx = np.arange(len(policy))
        ns, r = NS[idx, policy], R[idx, policy]
        sweeps = 0

        while True:
            V_new = r + self.gamma * V[ns]
            V_new[terminal] = 0.0
            delta = np.abs(V_new - V).max()
            V = V_new
            sweeps += 1
            if delta < self.theta:
                return V, sweeps

    def _evaluate_exact(self, policy, NS, R, terminal):
        # Solve (I - gamma * P_pi) V = r_pi directly
        n = len(policy)
        idx = np.arange(n)
        P = np.zeros((n, n))
        P[idx, NS[idx, policy]] = 1.0
        P[terminal] = 0.0
        r = R[idx, policy].copy()
        r[terminal] = 0.0
        return np.linalg.solve(np.eye(n) - self.gamma * P, r)

    # ------------------------------------------------------------------ acting
    def act(self, obs, size):
        x, y = int(obs[0]), int(obs[1])
        return int(self.policy[y * size + x])

    @staticmethod
    def layout_key(obs):
        """Everything except the agent position defines the maze layout."""
        return tuple(int(v) for v in obs[2:])