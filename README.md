# Knowledge Representation & Reasoning – Assignment 1  
**Author:** Luke Kenny  
**Student ID:** R00212866  

---

## 1.1 Building Your World

### Defining the 2D World (Task Environment)

I use a simple, well-studied **GridWorld** that enables controlled experimentation and isolates agent design effects from environment complexity.

#### Mechanics
- The world is represented as a 2D grid of cells.  
- Each cell may be empty, an obstacle, or the goal.  
- The agent always starts at the top-left corner `(0, 0)`.  
- The goal is fixed at the bottom-right corner `(n-1, n-1)`.  
- The game ends successfully when the agent reaches the goal cell.  

---

### Environment Properties

| Property | Classification | Justification |
|-----------|----------------|----------------|
| Observability | Fully observable | The agent always knows its current cell; evaluation focuses on reasoning rather than perception. |
| Agents | Single-agent | Only one agent navigates the grid at a time, with no interaction with others. |
| Determinism | Deterministic | Actions always succeed, so performance reflects reasoning, not randomness. |
| Episodic / Sequential | Sequential | Each action affects future states; mistakes accumulate over time. |
| Dynamics | Static | The environment does not change while the agent plans. |
| Discrete / Continuous | Discrete | The world has a finite grid and finite actions. |
| Known / Unknown | Known | Environment rules are fixed and defined; only obstacle placement is uncertain. |

---

### PEAS Description (Performance, Environment, Actuators, Sensors)

| Agent Type | Performance | Environment | Actuators | Sensors |
|-------------|--------------|--------------|-------------|-----------|
| Simple Reflex Agent | Reach goal, minimize steps, success rate | 2D grid with obstacles, fixed start/goal | Up/Down/Left/Right | Current cell contents (empty, obstacle, goal) |
| Model-Based Agent | Higher success than reflex, fewer redundant moves | 2D grid world with obstacles; scalable | Up/Down/Left/Right | Current cell contents + memory of visited cells |
| Goal-Based Agent | Optimal/near-optimal paths, path cost, and success rate | 2D grid world, variable size & density with weights | Up/Down/Left/Right | Current cell contents + planning/search model |

---

### Agent Designs & Critiques

#### Simple Reflex Agent
**Advantages**
- Very fast, lightweight model — acts immediately.  
**Disadvantages**
- No memory or planning.  
- Frequently attempts invalid moves or loops indefinitely.  
- Works in small, static worlds but collapses as it scales.  

---

#### Model-Based Reflex Agent
**Advantages**
- Utilizes memory of visited states to avoid cycles.  
- Scales better than simple reflex.  
**Disadvantages**
- Requires bookkeeping and memory overhead.  
- Still lacks the ability to generate optimal paths.  

> More robust in deterministic environments but less efficient than planned reasoning.

---

#### Goal-Based Agent
**Advantages**
- Finds optimal solutions using search (e.g., A*, UCS).  
- Scales well with environment size.  
**Disadvantages**
- Computationally expensive for large grids.  
- Performance depends heavily on the search heuristic.  

> Excels in discrete, known, sequential environments where planning is feasible.

---

### Early Performance Demonstration (1.1)

| Agent | Observations |
|--------|---------------|
| **Reflex Agent** | Moderate success in small grids but collapses in larger ones. |
| **Model-Based Agent** | Avoids loops and redundant moves but lacks true planning. |
| **Goal-Based Agent** | Achieves near-optimal paths through reasoning; slower but reliable. |

---

## 1.2 Problem Statement – Search Formulation

### Problem Setup
In GridWorld, the task is to navigate from `(0, 0)` to `(n–1, n–1)` with obstacles and optional weighted terrain.

**State:** agent’s position `(row, col)`  
**Initial state:** `(0, 0)`  
**Actions:** `{Up, Down, Left, Right}`  
**Transition model:** deterministic movement  
**Goal test:** reaching `(n–1, n–1)`  
**Path cost:**  
- Uniform: each step = 1  
- Weighted: terrain costs (flat=1, hill=2, water=3)

---

## Uninformed Search Techniques

### Breadth-First Search (BFS)
- Expands nodes level by level.  
- Guarantees an optimal solution (shortest path) in uniform-cost worlds.  
- Complete but memory intensive.  

### Depth-First Search (DFS)
- Explores one branch deeply before backtracking.  
- Uses little memory but produces non-optimal paths.  
- Sensitive to action ordering.  

### Uniform Cost Search (UCS)
- Expands by cumulative path cost `g(n)`.  
- Optimal for all positive-cost domains.  
- Matches BFS in uniform grids, outperforms it in weighted ones.

---

### Uninformed Search Results

| World | Algorithm | Success (%) | Avg Steps | Avg Cost | Nodes Expanded | Max Frontier |
|--------|------------|-------------|------------|-----------|----------------|---------------|
| 5×5 (Uniform) | BFS | 73.3 | 8.0 | 8.0 | 20 | 4 |
| | DFS | 73.3 | 10.1 | 10.1 | 16 | 6 |
| | UCS | 73.3 | 8.0 | 8.0 | 20 | 4 |
| 10×10 (Uniform) | BFS | 83.3 | 18.1 | 18.1 | 78 | 9 |
| | DFS | 83.3 | 31.2 | 31.2 | 51 | 24 |
| | UCS | 83.3 | 18.1 | 18.1 | 78 | 9 |
| 5×5 (Weighted) | BFS | 80.0 | 8.0 | 11.1 | 20 | 4 |
| | DFS | 80.0 | 10.8 | 14.9 | 14 | 6 |
| | UCS | 80.0 | 8.0 | 9.7 | 20 | 5 |
| 10×10 (Weighted) | BFS | 83.3 | 18.0 | 26.1 | 79 | 9 |
| | DFS | 83.3 | 32.0 | 47.2 | 53 | 25 |
| | UCS | 83.3 | 18.1 | 22.2 | 77 | 10 |

---

### Comparative Analysis

- **BFS:** Optimal in steps, but high memory usage.  
- **DFS:** Low memory, poor path quality; prone to dead-ends.  
- **UCS:** Optimal in both uniform and weighted; best general choice.  

**Implications:**
- Goal-based agents should use UCS in weighted worlds, BFS/UCS in uniform.  
- Model-based agents lack optimality without `g(n)` tracking.  
- Reflex agents remain fragile as environment complexity increases.  

---

## Informed Search Techniques

### Greedy Best-First Search (GBFS)
- Uses heuristic `h(n)` only (Manhattan distance).  
- Fast but ignores path cost — not optimal.  

### A* Search
- Combines `g(n) + h(n)`; admissible and consistent.  
- Complete and optimal in both uniform and weighted grids.  

### IDA* (Iterative Deepening A*)
- DFS variant with increasing cost thresholds.  
- Optimal with linear space complexity; may re-expand nodes.  

---

### Informed Search Results

| World | Algorithm | Success (%) | Avg Steps | Avg Cost | Nodes Expanded | Max Frontier |
|--------|------------|-------------|------------|-----------|----------------|---------------|
| 5×5 (Uniform) | GBFS | 73.3 | 8.0 | 8.0 | 9 | 5 |
| | A* | 73.3 | 8.0 | 8.0 | 19 | 4 |
| | IDA* | 73.3 | 8.0 | 8.0 | 9 | 8 |
| 10×10 (Uniform) | GBFS | 83.3 | 19.4 | 19.3 | 23 | 15 |
| | A* | 83.3 | 18.1 | 18.1 | 64 | 12 |
| | IDA* | 83.3 | 18.1 | 18.1 | 55 | 18 |
| 5×5 (Weighted) | GBFS | 80.0 | 8.1 | 11.0 | 9 | 5 |
| | A* | 80.0 | 8.0 | 9.7 | 15 | 6 |
| | IDA* | 80.0 | 8.0 | 9.7 | 22 | 8 |
| 10×10 (Weighted) | GBFS | 83.3 | 19.8 | 29.1 | 24 | 16 |
| | A* | 83.3 | 18.1 | 22.0 | 53 | 16 |
| | IDA* | 83.3 | 18.1 | 22.0 | 342 | 18 |

---

### Informed Search Analysis

- **GBFS:** Extremely fast but non-optimal; cuts through high-cost terrain.  
- **A\*:** Safest, most balanced — optimal and efficient.  
- **IDA\*:** Optimal with minimal memory; slower due to re-expansions.  

**Best Choice:**  
Use **A\*** for environments where cost matters, **GBFS** for speed, and **IDA\*** when memory is limited.

---

### References
Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach (4th Edition).* Pearson Education.

---

### Author
**Luke Kenny**  
MSc Artificial Intelligence, Munster Technological University  
Student ID: R00212866  
