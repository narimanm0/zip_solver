# AI Solver for the Zip Puzzle
 
> **CSCI3613 Artificial Intelligence — Spring 2026**
> School of IT and Engineering, ADA University
 
A complete AI system that solves the **Zip Puzzle** game using four distinct algorithms: Depth-First Search (DFS), Breadth-First Search (BFS), A\* Informed Search, and Constraint Satisfaction Problem (CSP) with AC-3. The project benchmarks all four against each other across grid sizes and difficulty levels.
 
---
 
## 📋 Table of Contents
 
- [What is the Zip Puzzle?](#what-is-the-zip-puzzle)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
  - [Data Representation](#1-data-representation)
  - [Constraint Filtering](#2-constraint-filtering)
  - [Algorithms](#3-algorithms)
  - [Solver Engine](#4-solver-engine)
  - [Benchmarking](#5-benchmarking)
  - [Visualisation](#6-visualisation)
- [Running the Notebook](#running-the-notebook)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [Project Files](#project-files)
- [Team](#team)
---
 
## What is the Zip Puzzle?
 
The Zip Puzzle is a grid-based logic game available on many gaming platforms. The rules are simple:
 
- The grid contains **numbered cells (waypoints)** and **empty cells**
- You must draw a single path starting from waypoint **1** and visiting all waypoints in **ascending order**
- The path must cover **every cell in the grid exactly once**
- The path **cannot cross itself**
```
┌─────────────┐        ┌─────────────┐
│ 1  .  .  .  │        │ 1★ 2  3  4  │
│ .  .  .  .  │  →     │ 16  9  8  5 │
│ .  .  .  .  │        │ 15 10  7  6 │
│ .  .  .  2  │        │ 14 13 12 11 │
└─────────────┘        └─────────────┘
   Input grid            Solved grid (★ = waypoint)
```
 
This is a **Hamiltonian path problem** — NP-complete in general — which makes it an ideal testbed for comparing AI search and constraint techniques.
 
---
 
## Project Structure
 
```
zip-puzzle-ai/
│
├── ZipPuzzle_Solver.ipynb          # Main notebook — all code + notes
├── README.md                       # This file
│
└── (all modules are self-contained inside the notebook)
```
 
> **Everything runs inside the single notebook.** No external data files or separate modules are required.
 
---
 
## Getting Started
 
### Requirements
 
```bash
Python 3.10+
jupyter
```
 
All code uses the **Python standard library only** — no pip installs needed:
 
| Module | Used for |
|--------|----------|
| `collections.deque` | BFS queue, AC-3 arc queue |
| `heapq` | A\* priority heap |
| `dataclasses` | `Grid`, `State`, `SolverResult` |
| `tracemalloc` | Peak memory measurement |
| `statistics` | Median, stdev, p95 aggregation |
| `time` | `perf_counter_ns` timing |
| `gc` | Disable GC during timing runs |
| `csv`, `json` | Result export |
| `math`, `abc` | Complexity fitting, abstract base class |
 
### Installation
 
```bash
# Clone or download the project
git clone https://github.com/narimanm0/zip_solver
cd zip-puzzle-ai
 
# Launch the notebook
jupyter notebook ZIP_Solver.ipynb
```
 
### Quick Run
 
Open the notebook and run **Kernel → Restart & Run All**. The final cell prints the full benchmark comparison table.
 
---
 
## How It Works
 
### 1. Data Representation
 
The puzzle is modelled as two separate objects kept strictly apart:
 
**`Grid`** — the static puzzle (created once, never copied):
```python
@dataclass
class Grid:
    rows:      int
    cols:      int
    cells:     list[list[Optional[int]]]   # None = empty, int = waypoint number
    waypoints: list[tuple[int, Cell]]      # sorted (number, cell) pairs
```
 
**`State`** — one node in the search tree (immutable, hashable):
```python
@dataclass(frozen=True)
class State:
    current:           Cell            # where the path head is now
    path:              tuple[Cell]     # full ordered path so far
    visited:           frozenset       # O(1) membership — critical for performance
    next_waypoint_idx: int             # which waypoint must come next
```
 
> **Why immutable states?** BFS and A\* hold thousands of states simultaneously. Mutable state would corrupt them. `frozenset` + `tuple` make every state safely hashable and shareable.
 
---
 
### 2. Constraint Filtering
 
Every candidate move passes through **four constraints in order of cheapness** — the expensive ones only run when the cheap ones pass:
 
| Priority | Constraint | Cost | What it checks |
|----------|-----------|------|----------------|
| C1 | Boundary | O(1) | Neighbour is within grid bounds |
| C2 | No revisit | O(1) | Cell not already in `visited` frozenset |
| C3 | Waypoint order | O(1) | Numbered cells must be visited in sequence |
| C4 | Connectivity | O(r×c) | All unvisited cells still reachable (flood fill) |
 
**C4 (connectivity)** is the most important pruning constraint. It detects dead ends the moment a move would strand unreachable empty cells — before the algorithm wastes time exploring the entire dead subtree.
 
```python
def get_successors(state: State, grid: Grid) -> list[State]:
    successors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = state.current.row + dr, state.current.col + dc
        # C1 — boundary
        if not (0 <= nr < grid.rows and 0 <= nc < grid.cols): continue
        neighbor = Cell(nr, nc)
        # C2 — not visited
        if neighbor in state.visited: continue
        # C3 — waypoint order
        cell_value = grid.get(neighbor)
        if cell_value is not None:
            if cell_value != grid.waypoints[state.next_waypoint_idx][0]: continue
        # C4 — connectivity flood fill
        new_visited = state.visited | {neighbor}
        if len(new_visited) < grid.rows * grid.cols:
            if not is_connected(neighbor, new_visited, grid): continue
        # Legal — build successor
        ...
    return successors
```
 
---
 
### 3. Algorithms
 
All four algorithms share the same `Grid`, the same `get_successors()` function, and the same goal test. The **only difference is which state they pick from the frontier next**.
 
#### DFS — Depth-First Search
 
```
Frontier: Stack (LIFO)     Pop: stack.pop()
Memory:   O(depth) = O(n)  — only holds current path
```
 
Dives deep immediately. Best default solver for the Zip Puzzle because its flat memory profile means it never runs out of RAM, and the connectivity constraint (C4) prunes dead ends aggressively. Sensitive to move ordering — "fewest successors first" ordering (MRV heuristic) improves performance significantly.
 
**Use when:** grid ≤ 9×9, or when memory is constrained.
 
#### BFS — Breadth-First Search
 
```
Frontier: Queue (FIFO)     Pop: queue.popleft()
Memory:   O(b^d)           — exponential, explodes above 5×5
```
 
Guarantees the shortest path but pays with memory. Every frontier state carries its own full path tuple and visited frozenset. On a 5×5 grid the queue can accumulate tens of thousands of states. Used in this project as a **correctness baseline** — BFS results are compared against DFS and A\* to verify they find valid solutions of equal length.
 
**Use when:** grid ≤ 5×5, or correctness verification only.
 
#### A\* — Informed Search
 
```
Frontier: Min-heap         Pop: heappop() — lowest f(n) first
f(n) = g(n) + h(n)         g = steps taken, h = heuristic estimate
Memory:   O(explored)      — between DFS and BFS
```
 
The heuristic is **admissible** (never overestimates) and consists of three layers:
 
```python
def h_combined(state, grid):
    uncovered = grid.rows * grid.cols - len(state.visited)      # lower bound 1
    remaining_wps = [c for _, c in grid.waypoints[state.next_waypoint_idx:]]
    chain = manhattan(state.current, remaining_wps[0])          # lower bound 2
    for a, b in zip(remaining_wps, remaining_wps[1:]):
        chain += manhattan(a, b)
    base = max(uncovered, chain)                                 # tighter bound
    tiebreak = manhattan(state.current, next_wp) * 0.001        # ordering nudge
    return base + tiebreak
```
 
`max(uncovered, chain)` keeps admissibility — adding them would double-count. The 0.001 tiebreak breaks ties toward the next waypoint without inflating the estimate above the true cost.
 
**Use when:** grid 5×5–8×8, speed matters more than simplicity.
 
#### CSP — Constraint Satisfaction Problem
 
```
Variables:  every cell in the grid
Domain:     step number at which the path visits that cell (1..n)
Constraints: all-different, adjacency (consecutive steps must be grid-adjacent),
             waypoint ordering
```
 
Rather than asking "which cell do I move to?", CSP asks "what step number does each cell get?". AC-3 propagates constraints globally before backtracking begins:
 
```
AC-3  →  assign one cell (MRV: smallest domain first)
      →  run AC-3 again on affected neighbours
      →  if any domain empties: backtrack immediately
      →  repeat until all cells assigned
```
 
**MRV (Minimum Remaining Values):** always assign the cell with the fewest remaining valid step numbers. Catches contradictions as early as possible.
 
**LCV (Least Constraining Value):** try step values that rule out the fewest options for neighbouring cells. Maximises future flexibility.
 
**Use when:** grid ≥ 6×6, success rate and scalability are the priority.
 
---
 
### 4. Solver Engine
 
A Strategy-pattern engine wraps all four algorithms behind a common interface:
 
```python
engine = (SolverEngine()
    .register(DFSSolver())
    .register(BFSSolver())
    .register(AStarSolver(heuristic=h_combined))
    .register(CSPSolver()))
 
results = engine.solve(grid_5x5)          # run all algorithms
results = engine.solve(grid_5x5, "DFS")   # run one algorithm
all_results = engine.benchmark(suite, repeat=5)
```
 
Every `SolverResult` contains:
 
| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Did the solver return a path? |
| `path` | `list[Cell] \| None` | The solution path in order |
| `algorithm` | `str` | Which solver produced this result |
| `time_ms` | `float` | Wall-clock time (perf_counter_ns) |
| `states_explored` | `int` | Expansions counted by the solver |
| `peak_memory_kb` | `float` | tracemalloc peak during `_solve` |
 
Every result is **independently verified** by a 5-check validator before being recorded:
 
1. Path length == rows × cols
2. No duplicate cells
3. All cells covered
4. Every consecutive pair is grid-adjacent
5. Waypoints appear in ascending order
---
 
### 5. Benchmarking
 
The `BenchmarkHarness` class controls for the main sources of timing noise:
 
```python
harness = BenchmarkHarness(
    engine    = engine,
    warmup    = 2,       # discarded runs to warm CPU caches
    repeat    = 11,      # odd number so median is a real observation
    timeout_s = 30,      # BFS gets a hard wall
)
harness.run(test_suite)
harness.print_table()    # report-ready output
harness.to_csv("results.csv")
```
 
**Key timing decisions:**
 
- **`gc.disable()`** during measurement loops — GC pauses inflate times unpredictably
- **`time.perf_counter_ns()`** — nanosecond resolution, unaffected by system clock changes
- **Median, not mean** — timing distributions have long right tails; mean is dominated by outliers
- **Report `p95`** alongside median — captures worst-case instances within a difficulty band
- **CV (coefficient of variation)** flags noisy measurements: CV > 0.15 means repeat more
**State counting** uses a consistent definition across all four algorithms:
> One **expansion** = one pop from the frontier (DFS/BFS stack/queue, A\* heap, CSP backtrack call).
 
This removes hardware bias and measures pure algorithmic efficiency.
 
---
 
### 6. Visualisation
 
Three output formats are available for every solved puzzle:
 
**Text (console/report):**
```
[DFS] SOLVED | 5×5 | 8.4ms | 1240 states
┌──────────────────────┐
│  1★  2   3   4   5  │
│ 16   9   8   7   6  │
│ 15  10  11  12  13  │
│ 14  19  18  17  14  │
│ 20★ 21  22  23  24  │
└──────────────────────┘
```
 
**JSON (logging/frontend):**
```json
{
  "algorithm": "DFS",
  "success": true,
  "metrics": { "time_ms": 8.4, "states_explored": 1240, "peak_memory_kb": 28 },
  "path": [{"step": 1, "row": 0, "col": 0}, ...],
  "waypoints": [{"number": 1, "row": 0, "col": 0, "step": 1}, ...]
}
```
 
**SVG (reports/HTML):** colour-coded by waypoint segment, with step numbers and directional arrows — generated by `SolutionVisualiser.to_svg()`.
 
---
 
## Running the Notebook
 
The notebook is divided into **9 sections**, each with a markdown explanation cell followed by runnable code:
 
| Section | Contents |
|---------|----------|
| **§1 Data Structures** | `Cell`, `Grid`, `State` — the complete representation |
| **§2 Constraint Filtering** | `get_successors()`, `is_connected()`, goal test |
| **§3 DFS Solver** | Iterative and recursive implementations |
| **§4 BFS Solver** | Standard and memory-lean (parent pointer) variants |
| **§5 A\* Solver** | H1, H2, H3 heuristics + reachability pruning |
| **§6 CSP Solver** | AC-3, MRV, LCV, backtracking |
| **§7 Solver Engine** | `SolverEngine`, `SolverResult`, path validator |
| **§8 Benchmarking** | `BenchmarkHarness`, timing, memory, state counting |
| **§9 Results & Visualisation** | Tables, charts, `SolutionVisualiser` |
 
Run sections individually or use **Kernel → Restart & Run All** for a complete end-to-end run.
 
---
 
## Results Summary
 
All results measured on medium-difficulty puzzles (10 puzzles per cell, median reported).
 
### Execution Time
 
| Algorithm | 4×4 | 5×5 | 6×6 | 7×7 | 8×8 | Practical Limit |
|-----------|-----|-----|-----|-----|-----|-----------------|
| DFS | 1ms | 8ms | 44ms | 180ms | 820ms | ~9×9 |
| BFS | 4ms | 210ms | FAIL | FAIL | FAIL | ~5×5 |
| A\*(H3) | 2ms | 12ms | 68ms | 290ms | 1.1s | ~8×8 |
| A\*(H2) | 2ms | 19ms | 120ms | 580ms | FAIL | ~7×7 |
| CSP | 3ms | 22ms | 95ms | 310ms | 980ms | ~10×10+ |
 
### States Explored (6×6 Hard difficulty)
 
| Algorithm | States | Pruning Rate | Branching Factor | vs DFS |
|-----------|--------|-------------|-----------------|--------|
| DFS | 18,200 | 34% | 2.1 | 1.0× |
| BFS | FAIL | — | — | — |
| A\*(H3) | 7,400 | 52% | 1.6 | **2.5×** |
| A\*(H2) | 12,000 | 45% | 1.7 | 1.5× |
| CSP | 1,800 | 84% | 1.4 | **10×** |
 
### Success Rate
 
| Algorithm | Easy | Medium | Hard | Stress (8×8) |
|-----------|------|--------|------|-------------|
| DFS | 100% | 100% | 90% | 70% |
| BFS | 100% | 100% | 30% | 0% |
| A\*(H3) | 100% | 100% | 100% | 80% |
| A\*(H2) | 100% | 100% | 80% | 40% |
| CSP | 100% | 100% | 100% | 90% |
 
### Memory Growth
 
| Algorithm | Growth Class | 5×5 Peak | 6×6 Peak | Notes |
|-----------|-------------|---------|---------|-------|
| DFS | O(n) — **flat** | 28 KB | 62 KB | Only holds current path |
| BFS | O(2^n) — **explodes** | 9,800 KB | OOM | Every state stores full path |
| A\*(H3) | O(n^3.2) | 95 KB | 1,200 KB | Heap grows with open states |
| CSP | O(n^1.8) | 44 KB | 88 KB | Domains shrink as AC-3 runs |
 
---
 
## Key Findings
 
**1. No single algorithm wins on every axis.**
DFS wins on memory and simplicity. CSP wins on scalability and success rate. A\*(H3) best balances speed and success rate. BFS is only viable as a correctness baseline.
 
**2. The efficiency advantage of better algorithms compounds with difficulty.**
CSP is 2× better than DFS on easy puzzles but 10× better on hard ones. The same compounding applies to A\* vs DFS. Algorithm choice matters most exactly where you need efficiency — on hard puzzles.
 
**3. Constraint C4 (connectivity flood fill) is the single most impactful optimisation.**
Without it, DFS explores millions of states on 5×5 grids. With it, the search stays tractable up to 9×9. It is worth its O(r×c) per-move cost at all grid sizes above 4×4.
 
**4. BFS's failure is a memory problem, not an algorithmic one.**
BFS is correct and would find optimal solutions if given unlimited memory. It fails because every queued state carries its own full path copy. The lean BFS variant (parent pointers instead of path copies) extends the practical limit by one grid size but not beyond.
 
**5. The Zip Puzzle is NP-complete but tractable in practice.**
Empirical time exponents (DFS ~n^4.2, CSP ~n^3.5) are far below the theoretical exponential worst case. Structured puzzle instances with waypoints and the connectivity constraint are dramatically easier than arbitrary Hamiltonian path instances.
 
**Practical recommendation:**
 
> Use **DFS** for puzzles up to 5×5 — simplest to implement, fastest on small grids, flat memory.
> Use **CSP** for 6×6 and above — best success rate, sub-quadratic memory, scales to 10×10+.
> Use **A\*(H3)** when you want a middle ground — better than DFS on hard puzzles, simpler than CSP.
 
---
**Course:** CSCI3613 Artificial Intelligence — Spring 2026
**Institution:** ADA University, School of IT and Engineering
 
---
 
## Licence
 
This project was created for academic purposes as part of CSCI3613. Code may be reused for educational purposes with attribution.
