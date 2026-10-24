"""Run Policy Iteration on the Grid Maze.

Examples
  python main.py --eval 1000                 # stats over 1000 random 5x5 mazes
  python main.py --record 5                  # record 5 episodes to ./videos
  python main.py --watch 3                   # live PyGame window
  python main.py --eval 20 --size 100        # scaling experiment
  python main.py --eval 200 --goals 3        # multiple goal cells
"""
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import grid_maze_env  # noqa: F401  (registers GridMaze-v0)
from policy_iteration import PolicyIterationAgent


def run_episode(env, agent, seed=None):
    obs, _ = env.reset(seed=seed)
    stats = agent.plan(env.unwrapped, PolicyIterationAgent.layout_key(obs))
    size = env.unwrapped.size
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(agent.act(obs, size))
        total += r
    return total, info["steps"], stats


def make_env(args, render_mode=None):
    return gym.make("GridMaze-v0", render_mode=render_mode, size=args.size,
                    n_goals=args.goals, n_bad=args.bad)


def evaluate(args):
    env = make_env(args)
    agent = PolicyIterationAgent(gamma=args.gamma, exact_eval=args.exact)
    results, iters, times = [], [], []

    for ep in range(args.eval):
        ret, steps, st = run_episode(env, agent, seed=args.seed + ep)
        results.append((ret, steps))
        iters.append(st["iterations"])
        times.append(st["time_s"])

    rets = np.array([r for r, _ in results])
    print(f"Episodes            : {args.eval}")
    print(f"Reached goal        : {(rets > 0).mean() * 100:.1f}%")
    print(f"Hit bad cell        : {(rets < 0).mean() * 100:.1f}%")
    print(f"Avg steps           : {np.mean([s for _, s in results]):.2f}")
    print(f"PI iterations       : mean={np.mean(iters):.2f}  min={min(iters)}  "
          f"max={max(iters)}  (distribution: {({int(k): int(v) for k, v in zip(*np.unique(iters, return_counts=True))})})")
    print(f"Planning time / maze: {np.mean(times) * 1000:.2f} ms")
    env.close()


def record(args):
    env = make_env(args, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=args.video_dir,
                      episode_trigger=lambda e: True, name_prefix="policy_iteration")
    agent = PolicyIterationAgent(gamma=args.gamma, exact_eval=args.exact)

    for ep in range(args.record):
        ret, steps, st = run_episode(env, agent, seed=args.seed + ep)
        print(f"episode {ep}: return={ret:+.0f} steps={steps} PI iterations={st['iterations']}")

    env.close()
    print(f"Videos saved in ./{args.video_dir}")


def watch(args):
    env = make_env(args, render_mode="human")
    agent = PolicyIterationAgent(gamma=args.gamma, exact_eval=args.exact)

    for ep in range(args.watch):
        ret, steps, _ = run_episode(env, agent, seed=args.seed + ep)
        print(f"episode {ep}: return={ret:+.0f} steps={steps}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--goals", type=int, default=1)
    p.add_argument("--bad", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--exact", action="store_true", help="exact policy evaluation (linear solve)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", type=int, default=0)
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--watch", type=int, default=0)
    p.add_argument("--video-dir", default="videos")
    args = p.parse_args()

    if args.eval:
        evaluate(args)

    if args.record:
        record(args)

    if args.watch:
        watch(args)

    if not (args.eval or args.record or args.watch):
        p.print_help()