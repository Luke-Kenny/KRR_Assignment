import sys, os
import random
from collections import deque
import heapq
import math

# parent directory AIMA-python
parent_dir = os.path.dirname(os.getcwd())

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

"""
@Author: Luke Kenny
@StudentId: R00212866
@Task: Assignment 1 KRR
@Group: A
@Python-Version: 3.13.2
"""

"""
Design NOTE:
As a Software Development graduate, I chose not to import any modules or classes 
from the AIMA repository to challenge myself. I used the AIMA text and codebase 
purely as references, implementing the agents, environment, and search algorithms 
myself to demonstrate, and grasp a full understanding of these concepts.
"""

"""1.1 CODE BELOW"""
"""Grid Environment"""
class GridWorld:
    """
    2D grid world:
    - Cells: '.' empty, 'O' obstacle, 'S' start (0,0), 'G' goal (n-1,n-1)
    - Actions are 4-connected moves; invalid if out of bounds or obstacle.
    Design choice: deterministic, static, discrete → isolates agent design effects.
    """
    def __init__(self,
                 size=5,
                 obstacle_probability=0.2,
                 rng: random.Random | None = None,
                 mode: str = "uniform", # modes: Uniform or Weighted to test and compare
                 terrain_probs: dict[str, float] | None = None,
                 terrain_cost: dict[str, int] | None = None):
        self.size = size
        self.variant = mode # used to label my runs in the output
        self.grid = [['.' for _ in range(size)] for _ in range(size)]

        # defining my start and goal
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.grid[self.start[0]][self.start[1]] = 'S' # start
        self.grid[self.goal[0]][self.goal[1]] = 'G' # goal

        rng = rng or random

        """
        Added this mode selection below as I noticed in uniform, results of BFS and UCS are identical
        However, in weighted environments I know this isn't the case as they perform differently
        """
        # setting my values for weighted mode test
        self.terrain_probs = terrain_probs or {'.': 0.65, '^': 0.20, '~': 0.15}
        self.terrain_cost = terrain_cost or {'.': 1, '^': 2, '~': 3, 'S': 1, 'G': 1}

        # populates the obstacles + weighted terrain
        for i in range(size):
            for j in range(size):
                if (i, j) in [self.start, self.goal]:
                    continue
                if rng.random() < obstacle_probability:
                    self.grid[i][j] = 'O' # obstacle
                else:
                    if mode == "uniform":
                        self.grid[i][j] = '.'
                    else: # weighted terrain mode
                        r = rng.random()
                        acc = 0.0
                        for sym, p in self.terrain_probs.items():
                            acc += p
                            if r <= acc:
                                self.grid[i][j] = sym
                                break


    def display(self):
        print(f"[GridWorld] size={self.size} mode={self.variant}")
        legend = "Legend: S(start)=1, G(goal)=1, .=1, ^=2, ~=3, O=obstacle"
        if self.variant == "uniform":
            legend = "Legend: S(start)=1, G(goal)=1, .=1, O=obstacle"
        print(legend)
        for row in self.grid:
            print(" ".join(row))
        print()


    def is_valid(self, position):
        x, y = position
        return (0 <= x < self.size and 0 <= y < self.size
                and self.grid[x][y] != 'O')


    def step_cost(self, to_pos):
        """
        Cost to ENTER 'to_pos'. For uniform worlds this returns 1 for all non-obstacles.
        For weighted worlds it returns cell-dependent costs.
        """
        x, y = to_pos
        cell = self.grid[x][y]
        # uniform mode: treats any non-obstacle as cost 1 BFS and UCS will produce the same results
        if self.variant == "uniform":
            return 1
        # weighted mode: uses terrain_cost map, obstacles are invalid so won't be queried
        return self.terrain_cost.get(cell, float('inf'))

# unifies directions and makes them available everywhere (important for my part 1.2)
DIRECTIONS = [
    ("UP",    (-1, 0)),
    ("DOWN",  ( 1, 0)),
    ("LEFT",  ( 0,-1)),
    ("RIGHT", ( 0, 1)),
]
ACTIONS = dict(DIRECTIONS) # actions gets to stay the same for my 1.1 impl

"""Base Agent Class: holds the position, step count, and visited memory"""
class Agent:
    ACTIONS = dict(DIRECTIONS)
    def __init__(self, env: GridWorld):
        self.env = env
        self.position = env.start
        self.steps = 0
        self.visited = {self.position}


    def movement(self, action):
        dx, dy = self.ACTIONS.get(action, (0, 0)) # default is no move
        x, y = self.position
        new_position = (x + dx, y + dy)

        if self.env.is_valid(new_position):
            self.position = new_position
            self.visited.add(new_position)
        self.steps += 1


    def at_goal(self):
        return self.position == self.env.goal


"""1: Simple Reflex Agent"""
class ReflexAgent(Agent):
    """
    A Simple Reflex Agent whose action depends *only* on the current percept.
    Percept here is the label of the current cell ('.', 'S', 'G'), not a map.
    No internal state is used (no memory), satisfying the reflex definition.
    """
    def percept(self):
        x, y = self.position
        return self.env.grid[x][y]  # current cell label


    def act(self):
        # 1 — if already at the goal, does nothing.
        if self.at_goal():
            return None
        # 2 — chooses uniformly amongst actions that lead to valid cells.
        x, y = self.position
        valid_actions = [
            a for a, (dx, dy) in self.ACTIONS.items()
            if self.env.is_valid((x + dx, y + dy))
        ]
        # 3 — if there's no valid actions (e.g. surrounded), does nothing.
        if not valid_actions:
            return None
        # 4 — tie-break stochastically (no state, purely reflexive).
        return random.choice(valid_actions)


"""2: Model-Based Agent"""
class ModelBasedAgent(Agent):
    def act(self):
        # preference towards unvisited cells
        unvisited_moves = []
        valid_moves = []
        x, y = self.position

        for action, (dx, dy) in self.ACTIONS.items():
            new_position = (x + dx, y + dy)
            if self.env.is_valid(new_position):
                valid_moves.append(action)
                if new_position not in self.visited:
                    unvisited_moves.append(action)

        if unvisited_moves:
            return random.choice(unvisited_moves)
        if valid_moves:
            return random.choice(valid_moves)
        return None

# after ACTIONS / DIRECTIONS
DELTA_TO_ACTION = {v: k for k, v in ACTIONS.items()}
def path_to_actions(path):
    return [] if not path else [
        DELTA_TO_ACTION[(x2-x1, y2-y1)]
        for (x1,y1),(x2,y2) in zip(path[:-1], path[1:])
]

"""3. Goal-Based Agent"""
class GoalBasedAgent(Agent):
    def __init__(self, env: GridWorld, planner: str = "BFS"):
        super().__init__(env)
        self.planner = planner.upper()
        self.plan = self._plan()

    def _plan(self):
        fn = PLANNER_FUNCS.get(self.planner, bfs_search)
        path, *_ = fn(self.env)
        return path_to_actions(path) if path else []

    def act(self):
        return None if not self.plan or self.at_goal() else self.plan.pop(0)

"""Runner Function"""
def run_agent(agent_class, env, max_steps=200):
    agent = agent_class(env)
    for _ in range(max_steps):
        if agent.at_goal():
            return True, agent.steps
        action = agent.act()
        if action is None:
            break
        agent.movement(action)
    return agent.at_goal(), agent.steps


def evaluate_agents(size=5, obstacle_probability=0.2, trials=30, max_steps=400, seed=43):
    agents = [ReflexAgent, ModelBasedAgent, GoalBasedAgent]
    stats = {A.__name__: {"success": 0, "steps": []} for A in agents}

    for t in range(trials):
        # on map per trial for each agent type
        trial_rng = random.Random(seed + t)
        random.seed(seed + t)
        env = GridWorld(size=size, obstacle_probability=obstacle_probability, rng=trial_rng)
        for A in agents:
            success, steps = run_agent(A, env, max_steps=max_steps)
            if success:
                stats[A.__name__]["success"] += 1
                stats[A.__name__]["steps"].append(steps)
    # summarising results
    eval_summary = {}
    for name, rec in stats.items():
        success_rate = 100.0 * rec["success"] / trials
        avg_steps = (sum(rec["steps"]) / len(rec["steps"])) if rec["steps"] else None
        eval_summary[name] = (success_rate, avg_steps)
    return eval_summary


def print_results_table(title, eval_summary):
    print(f"\n{title}")
    print(f"{'Agent':20} | {'Success %':>9} | {'Avg Steps (on success)':>24}")
    print("-" * 60)
    for name, (succ, avg) in eval_summary.items():
        avg_txt = f"{avg:.1f}" if avg is not None else "—"
        print(f"{name:20} | {succ:9.1f} | {avg_txt:>24}")


"""1.2 CODE BELOW - Search Formulation + Algorithms"""
def neighbors(env: GridWorld, pos):
    """Yield next_pos, action_name, step_cost."""
    x, y = pos
    for name, (dx, dy) in DIRECTIONS:
        nxt = (x + dx, y + dy)
        if env.is_valid(nxt):
            yield nxt, name, env.step_cost(nxt)


def reconstruct_path(parent, start, goal):
    """Returns (path, steps) under unit step cost; (None, None) if no path."""
    if goal not in parent:
        return None, None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path, len(path) - 1  # unit-cost edges


def path_cost(env: GridWorld, path):
    """Sums of entry costs along path (excluding the start cell)."""
    if not path:
        return None
    return sum(env.step_cost(pos) for pos in path[1:])

"""Heuristics for informed algos

I use Manhattan distance (L1) scaled by the cheapest legal step cost in the current
environment. This scaling keeps the heuristic admissible and consistent in weighted
worlds: h(n) never overestimates the true least cost because the cheapest way to close
one grid unit cannot be cheaper than the globally minimal per-step cost.
This makes h(n) consistent on 4-connected grids.
"""
def _min_traversable_cost(env: GridWorld) -> int:
    # cheapest legal step cost keeps heuristics admissible in weighted grid worlds
    # in uniform mode all traversable cells cost 1 by definition.
    if env.variant == "uniform":
        return 1
    # excluding obstacles ('O'), which are non-traversable.
    return min(v for k, v in env.terrain_cost.items() if k != 'O')


def manhattan_heuristic(env: GridWorld, pos) -> float:
    # L1 distance to the goal, scaled by min step cost. Admissible and consistent for 4-connected grids.
    (x, y), (gx, gy) = pos, env.goal
    dx, dy = abs(x - gx), abs(y - gy)
    return (dx + dy) * _min_traversable_cost(env)


# fast heuristic accessor: uses precomputed table when available
def heuristic(env: GridWorld, pos) -> float:
    if hasattr(env, "h_map"):
        return env.h_map[pos]
    return manhattan_heuristic(env, pos)


"""UNINFORMED Algorithms"""
def bfs_search(env: GridWorld):
    """Breadth-First Search - Returns: (path, steps, expanded, max_frontier)"""
    start, goal = env.start, env.goal
    q = deque([start])
    parent = {start: None}
    expanded = 0
    max_frontier = 1

    while q:
        max_frontier = max(max_frontier, len(q))
        s = q.popleft()
        expanded += 1
        if s == goal:
            break
        # neighbors yield (nxt, action_name, step_cost) I ignore action and cost here
        for nxt, _, _ in neighbors(env, s):
            if nxt not in parent: # serves as a visited set
                parent[nxt] = s
                q.append(nxt)
    path, steps = reconstruct_path(parent, start, goal) # steps == path length - 1
    return path, steps, expanded, max_frontier


def dfs_search(env: GridWorld, max_visits=20_000):
    """Depth-First Search (low memory, not optimal, can get lost).

    I use a hard cap on total pops (max_visits) to prevent pathological
    wandering on awkward maps.

    Returns: (path, steps, expanded, max_frontier)
    - expanded: nodes popped from the stack
    - max_frontier: peak stack size
    """
    start, goal = env.start, env.goal
    stack = [start]
    parent = {start: None} # doubles as a visited set
    visits = 0
    expanded = 0
    max_frontier = 1

    while stack and visits < max_visits:
        max_frontier = max(max_frontier, len(stack))
        s = stack.pop()
        visits += 1
        expanded += 1
        if s == goal:
            break
        """NOTE: I used reversed() here so that when pushing neighbors onto the stack (LIFO),
         the effective expansion order matches the intended UP → DOWN → LEFT → RIGHT order."""
        for nxt, _, _ in reversed(list(neighbors(env, s))):
            if nxt not in parent:
                parent[nxt] = s
                stack.append(nxt)
    path, steps = reconstruct_path(parent, start, goal)
    return path, steps, expanded, max_frontier


def ucs_search(env: GridWorld):
    """Uniform-Cost Search (optimal for non-negative step costs).

    Equals BFS on unit-cost grids; differs in weighted mode.

    Returns: (path, steps, expanded, max_frontier)
    - expanded: nodes popped from the priority queue with their best-known g
    - max_frontier: peak PQ length
    """
    start, goal = env.start, env.goal
    pq = [(0, start)] # (g_cost, state)
    parent = {start: None}
    g_cost = {start: 0}
    expanded = 0
    max_frontier = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))
        g, s = heapq.heappop(pq)

        # UCS guard: skips if it popped a stale entry
        if g > g_cost.get(s, float("inf")):
            continue
        expanded += 1
        if s == goal:
            break
        for nxt, _, step_c in neighbors(env, s):
            ng = g + step_c  # accumulates true step cost
            if nxt not in g_cost or ng < g_cost[nxt]:
                g_cost[nxt] = ng
                parent[nxt] = s
                heapq.heappush(pq, (ng, nxt))
    path, steps = reconstruct_path(parent, start, goal)
    return path, steps, expanded, max_frontier


"""INFORMED Algorithms"""
def gbfs_search(env: GridWorld):
    """
    Greedy Best-First Search guided purely by h(n).
    - Fast, low memory, not optimal nor complete in graphs with zero-cost cycles.
    - Note: contrasted against A*: expects fewer expansions but higher path cost in weighted grids.
    Returns: (path, steps, expanded, max_frontier)
    """
    start, goal = env.start, env.goal
    pq = [(heuristic(env, start), start)]
    parent = {start: None}
    visited = {start}
    expanded = 0
    max_frontier = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))
        _, s = heapq.heappop(pq)
        expanded += 1
        if s == goal:
            break
        for nxt, _, _ in neighbors(env, s):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = s
            heapq.heappush(pq, (heuristic(env, nxt), nxt))

    path, steps = reconstruct_path(parent, start, goal)
    return path, steps, expanded, max_frontier


def astar_search(env: GridWorld):
    """A* with f = g + h

    In uniform grids this behaves like UCS but prunes more aggressively via h.

    Returns: (path, steps, expanded, max_frontier)
    """
    start, goal = env.start, env.goal
    g_cost = {start: 0}
    pq = [(heuristic(env, start), 0, start)] # (f, g, state)
    parent = {start: None}
    expanded = 0
    max_frontier = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))
        f, g, s = heapq.heappop(pq)
        if g > g_cost.get(s, float("inf")): # stale entry
            continue
        expanded += 1
        if s == goal:
            break
        for nxt, _, step_c in neighbors(env, s):
            ng = g + step_c
            if ng < g_cost.get(nxt, float('inf')):
                g_cost[nxt] = ng
                parent[nxt] = s
                nf = ng + heuristic(env, nxt)
                heapq.heappush(pq, (nf, ng, nxt))
    path, steps = reconstruct_path(parent, start, goal)
    return path, steps, expanded, max_frontier


def idastar_search(env: GridWorld):
    """IDA* (Iterative Deepening A*) with f = g + h.

    Rationale:
    - **Optimal** under admissible & consistent h (ours is).
    - **Linear memory** (depth-first contour expansions).
    - Typically more re-expansions than A*, but much friendlier memory profile.

    Implementation notes:
    - threshold starts at f(start) and increases to the smallest f that exceeded the bound.
    - I keep a global best_g to avoid revisiting states via strictly worse g.
      I use a *strict* comparison (ng > best_g.get(...)) so equal-g revisits are allowed
      if they arise with different contours (safer for IDA*; fewer accidental prunes).

    Returns: (path, steps, expanded, max_frontier)
    - expanded: nodes visited within current bound
    - max_frontier: max recursion depth (DFS stack height)
    """
    start, goal = env.start, env.goal

    def f(g, s): return g + heuristic(env, s)

    threshold = f(0, start)
    expanded = 0
    max_frontier = 1

    parent = {start: None}
    best_g = {start: 0.0}  # global best g to cut duplicates across iterations

    def dfs(s, g, bound, on_path):
        nonlocal expanded, max_frontier
        Fs = f(g, s)
        if Fs > bound:
            return Fs
        if s == goal:
            return True

        expanded += 1
        # order successors by f to behave like best-first on each iteration
        succs = []
        for nxt, _, step_c in neighbors(env, s):
            if nxt in on_path:
                continue ## cycle check on current recursion stack
            ng = g + step_c
            # > means it only prunes strictly worse g, equal-g revisits are allowed
            if ng > best_g.get(nxt, math.inf):
                continue
            best_g[nxt] = ng
            parent[nxt] = s
            succs.append((nxt, ng, f(ng, nxt)))
        succs.sort(key=lambda t: t[2])  # sort by f = g + h

        on_path_size = len(on_path)
        max_frontier = max(max_frontier, on_path_size)

        # explore best-first, tracking the smallest f that exceeds the bound
        min_excess = math.inf
        for nxt, ng, _ in succs:
            on_path.add(nxt)
            t = dfs(nxt, ng, bound, on_path)
            on_path.remove(nxt)
            if t is True:
                return True
            if t < min_excess:
                min_excess = t
        return min_excess

    on_path = {start}
    while True:
        t = dfs(start, 0.0, threshold, on_path)
        if t is True:
            path, steps = reconstruct_path(parent, start, goal)
            return path, steps, expanded, max_frontier
        if t == math.inf:
            return None, None, expanded, max_frontier
        threshold = t # next contour bound

# maps names to the search functions
PLANNER_FUNCS = {"BFS": bfs_search, "DFS": dfs_search, "UCS": ucs_search,
                 "GBFS": gbfs_search, "A*": astar_search, "IDA*": idastar_search}

# planner presets \\
PLANNERS_UNINFORMED = [
    ("BFS", bfs_search),
    ("DFS", dfs_search),
    ("UCS", ucs_search),
]

PLANNERS_INFORMED = [
    ("GBFS", gbfs_search),
    ("A*",   astar_search),
    ("IDA*", idastar_search),
]


def evaluate_searches(size: int = 5,
                      obstacle_probability: float = 0.2,
                      trials: int = 30,
                      planners=None,
                      dfs_max_visits: int = 20_000,
                      seed: int = 99,
                      mode: str = "uniform",
                      terrain_probs=None,
                      terrain_cost=None,
                      precheck_unsolvable: bool = True,
                      verbose_progress: bool = True):
    """
    Ran multiple planners over freshly generated gridworld maps and aggregate metrics.
    Returns:
        dict[str, tuple]:
            {
              name: (
                success_over_all_trials_pct,
                success_over_solvable_trials_pct,
                avg_steps_on_success,
                avg_true_path_cost_on_success,
                avg_nodes_expanded,
                avg_max_frontier,
                solvable_trials_count
              )
            }
    """
    if planners is None:
        planners = PLANNERS_UNINFORMED

    # allowing caller to override dfs visit cap without changing the presets
    if dfs_max_visits != 20_000:
        planners = [
            (name, (lambda env, cap=dfs_max_visits: dfs_search(env, max_visits=cap)))
            if name == "DFS" else
            (name, fn)
            for name, fn in planners
        ]

    # small helper
    def _avg(xs): return (sum(xs) / len(xs)) if xs else None

    # precomputing h(n) table
    def _precompute_h_map(env: GridWorld) -> dict[tuple[int, int], int]:
        gx, gy = env.goal
        step = 1 if env.variant == "uniform" else min(v for k, v in env.terrain_cost.items() if k != 'O')
        return {(x, y): (abs(x - gx) + abs(y - gy)) * step
                for x in range(env.size) for y in range(env.size)}

    # stat containers
    stats = {
        name: {"success": 0, "attempts": 0, "steps": [], "cost": [], "expanded": [], "max_frontier": []}
        for name, _ in planners
    }
    solvable_trials = 0

    progress_every = max(1, trials // 5)
    for t in range(trials):
        trial_rng = random.Random(seed + t)
        random.seed(seed + t) # for reproducing agent tie-breaks
        env = GridWorld(size=size,
                        obstacle_probability=obstacle_probability,
                        rng=trial_rng,
                        mode=mode,
                        terrain_probs=terrain_probs,
                        terrain_cost=terrain_cost)

        env.h_map = _precompute_h_map(env)

        # fast pre-check, if no path at unit cost, skip the whole trial
        if precheck_unsolvable:
            pre_path, _, _, _ = bfs_search(env)
            if pre_path is None:
                # unsolvable map; do not count an attempt for any planner
                continue
            solvable_trials += 1
        else:
            # If not pre-checking, consider all trials "attempted" for each planner
            solvable_trials = trials

        for name, fn in planners:
            # counts that this planner attempted this (solvable) map
            stats[name]["attempts"] += 1

            path, steps, expanded, max_frontier = fn(env)
            if path is None:
                if verbose_progress and (t % progress_every == 0):
                    print("fail")
                continue

            stats[name]["success"] += 1
            stats[name]["steps"].append(steps)
            stats[name]["cost"].append(path_cost(env, path))
            stats[name]["expanded"].append(expanded)
            stats[name]["max_frontier"].append(max_frontier)

    # summary
    summary = {}
    for name, rec in stats.items():
        succ_all = 100.0 * rec["success"] / trials if trials else 0.0
        succ_solvable = 100.0 * rec["success"] / rec["attempts"] if rec["attempts"] else 0.0
        summary[name] = (
            succ_all,
            succ_solvable,
            _avg(rec["steps"]),
            _avg(rec["cost"]),
            _avg(rec["expanded"]),
            _avg(rec["max_frontier"]),
            solvable_trials
        )
    return summary


def print_search_table(title, summary):
    print(f"\n{title}")
    print(f"{'Algorithm':12} | {'Success %':>9} | {'Solv.%':>7} | "
          f"{'Avg Steps (on success)':>23} | {'Avg Cost':>9} | "
          f"{'Nodes Expanded':>15} | {'Max Frontier':>12}")
    print("-" * 110)
    # pull any entry to read solvable_trials (same for all algos in a run)
    any_key = next(iter(summary)) if summary else None
    solvable_trials = summary[any_key][-1] if any_key else 0
    for name, (succ_all, succ_solve, steps, cost, exp, mf, _) in summary.items():
        s_txt  = f"{steps:.1f}" if steps is not None else "—"
        c_txt  = f"{cost:.1f}"  if cost  is not None else "—"
        e_txt  = f"{int(exp):d}" if exp   is not None else "—"
        mf_txt = f"{int(mf):d}"  if mf    is not None else "—"
        print(f"{name:12} | {succ_all:9.1f} | {succ_solve:7.1f} | "
              f"{s_txt:23} | {c_txt:9} | {e_txt:15} | {mf_txt:12}")
    print(f"(solvable maps this run: {solvable_trials})")


# small helper to avoid repetition across run blocks
def _run_and_print(size, trials, obstacle_p, mode, planners, title):
    summary = evaluate_searches(
        size=size,
        obstacle_probability=obstacle_p,
        trials=trials,
        mode=mode,
        planners=planners
    )
    print_search_table(title, summary)


def run_part1_1():
    """encapsulation of running part 1.1 and 1.2"""
    print("\nPART 1.1")
    # single-map demos (visual)
    print("\nsmall world (5x5) - single map demo")
    env_small = GridWorld(size=5, obstacle_probability=0.2)
    env_small.display()
    for agent_class in [ReflexAgent, ModelBasedAgent, GoalBasedAgent]:
        success, steps = run_agent(agent_class, env_small, max_steps=200)
        print(f"{agent_class.__name__}: success={success}, steps={steps}")

    print("\nlarge world (10x10) - single map demo")
    env_large = GridWorld(size=10, obstacle_probability=0.2)
    env_large.display()
    for agent_class in [ReflexAgent, ModelBasedAgent, GoalBasedAgent]:
        success, steps = run_agent(agent_class, env_large, max_steps=400)
        print(f"{agent_class.__name__}: success={success}, steps={steps}")

    # quantitative agent evaluation (multi-trial)
    small_summary = evaluate_agents(size=5, obstacle_probability=0.2, trials=30, max_steps=200)
    print_results_table("agents - 5x5 world - 30 trials", small_summary)

    large_summary = evaluate_agents(size=10, obstacle_probability=0.2, trials=30, max_steps=400)
    print_results_table("agents - 10x10 world - 30 trials", large_summary)


def run_part1_2_uninformed():
    """encapsulates all experiments for q1.2 (uninformed search)"""
    print("\nPART 1.2: SEARCH (UNINFORMED)")

    # uniform-cost worlds (bfs and ucs perform similar)
    _run_and_print(
        size=5, trials=30, obstacle_p=0.2, mode="uniform",
        planners=PLANNERS_UNINFORMED,
        title="search - 5x5 world - 30 trials (uniform)"
    )
    _run_and_print(
        size=10, trials=30, obstacle_p=0.2, mode="uniform",
        planners=PLANNERS_UNINFORMED,
        title="search - 10x10 world - 30 trials (uniform)"
    )

    # weighted terrains (ucs minimises true path cost)
    _run_and_print(
        size=5, trials=30, obstacle_p=0.2, mode="weighted",
        planners=PLANNERS_UNINFORMED,
        title="search - 5x5 world - 30 trials (weighted)"
    )
    _run_and_print(
        size=10, trials=30, obstacle_p=0.2, mode="weighted",
        planners=PLANNERS_UNINFORMED,
        title="search - 10x10 world - 30 trials (weighted)"
    )


def run_part1_2_informed():
    """encapsulates all experiments for q1.2 (informed search)"""
    print("\nPART 1.2: SEARCH (INFORMED)")

    # uniform-cost
    _run_and_print(
        size=5, trials=30, obstacle_p=0.2, mode="uniform",
        planners=PLANNERS_INFORMED,
        title="informed - 5x5 world - 30 trials (uniform)"
    )
    _run_and_print(
        size=10, trials=30, obstacle_p=0.2, mode="uniform",
        planners=PLANNERS_INFORMED,
        title="informed - 10x10 world - 30 trials (uniform)"
    )

    # weighted terrains
    _run_and_print(
        size=5, trials=30, obstacle_p=0.2, mode="weighted",
        planners=PLANNERS_INFORMED,
        title="informed - 5x5 world - 30 trials (weighted)"
    )
    _run_and_print(
        size=10, trials=30, obstacle_p=0.2, mode="weighted",
        planners=PLANNERS_INFORMED,
        title="informed - 10x10 world - 30 trials (weighted)"
    )


def run_part1_2():
    """Runs both uninformed and informed experiments for Part 1.2."""
    run_part1_2_uninformed()
    run_part1_2_informed()


if __name__ == '__main__':
    run_part1_1()
    run_part1_2()

