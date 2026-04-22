# Counter-UAS Swarm Defense Simulator

A multi-agent reinforcement learning system where autonomous drones learn to defend airspace against kamikaze threats — no hardcoded tactics, no communication protocol, no formation logic. Just a shaped reward landscape and PPO.



https://github.com/user-attachments/assets/d6d783b9-d994-4b68-a776-ca134eb9671d

![](Visualization-video.mp4)


## Why This Exists

The economics of modern air defense are broken. One-way attack drones cost $20-50K each. The missiles intercepting them cost $3-15M. That's a 60-750x cost asymmetry favoring the attacker. When 250+ kamikaze drones are inbound, the equipment bottleneck hits before the economics even become a thought.

The current counter-UAS research landscape falls into a few camps:
- **Handcrafted coordination frameworks** with explicit phase transitions, role assignments, and formation geometry — achieving 95%+ interception rates, but every behavior is manually designed.
- **RL-based target prioritization:** the agent learns *which* threat to engage, but effectors are stationary with predefined engagement mechanics.
- **Physics-based 1v1 pursuit:** realistic quadrotor dynamics, but only one pursuer vs one evader.

This project asks a different question: **what if we give RL agents the full 3D movement problem and let them figure everything out?**

## Results

Evaluated over **186 consecutive deterministic inference runs:**

| Outcome | Count | Percentage |
|---|---|---|
| 5/5 interceptions, 0 breaches | 127 | 68.3% |
| 4/5 interceptions, 1 breach | 55 | 29.6% |
| 3/5 interceptions, 2 breaches | 4 | 2.2% |
| **≤ 1 breach** | **182** | **97.8%** |

Zero episodes with 3+ breaches. Every tactic is emergent, the agents learned positioning, interception geometry, spatial coordination, and engagement timing entirely through reward shaping.

### Emergent Behaviors Observed

- **Spatial spread** across approach vectors without any formation logic
- **Far-from-base engagement** prioritized over close-range chasing
- **Directional positioning** ahead of hostile trajectories to cut off flight paths
- **Self-distribution** through proximity repulsion. No two friendlies cluster within 5 units

## Architecture

### Environment (`env.py`)

- **Grid:** 50×50×50 3D space
- **Base:** Center ground at (25, 25, 0)
- **Agents:** 5 friendly vs 5 hostile drones
- **Mechanics:** Kamikaze interception, both drones are destroyed on contact
- **Actions:** 27 discrete (3 choices per axis: -1, 0, +1)
- **Hostile behavior:** Spawn on grid boundary (min 30 units from base), fly toward base at speed 1.0
- **Friendly speed:** 1.5x multiplier over hostiles
- **Intercept radius:** 2.0 units
- **Breach radius:** 6.0 units
- **Max steps:** 500 per episode

### Observation Space (`sb3_env.py`)

65-dimensional vector per step:
- All friendly positions (5 × 3 = 15)
- All friendly positions from env state (5 × 3 = 15)
- All hostile positions + velocities + alive status (5 × 7 = 35)

### Reward Function (9 components)

| Component | Description |
|---|---|
| **Zone penalties** | Alive hostiles penalized by proximity to base: Zone 1 (≤12 units) = -0.2, Zone 2 (≤30) = -0.05, Zone 3 (>30) = -0.01 per tick |
| **Interception bonus** | +3.0 × distance_bonus for the intercepting drone. Distance bonus scales with how far from base the interception occurs |
| **Team bonus** | +1.5 × distance_bonus for all other friendly drones on successful interception |
| **Breach penalty** | -8.0 to all agents when a hostile reaches the base |
| **Directional cone** | +0.01 scaled reward within 30 units when friendly is positioned ahead of hostile's trajectory (dot product alignment > 0) |
| **Close-range sphere** | +0.03 scaled reward within 8 units for proximity engagement |
| **Close-range zone penalties** | Negative rewards for being too close to hostiles near the base (zones 1 and 2) — forces early interception |
| **Proximity repulsion** | -0.3 max penalty when two friendlies are within 5 units — prevents clustering |
| **Time penalty** | -0.05 per tick encouraging faster resolution |

### Training Setup (`train_ppo.py`)

- **Algorithm:** PPO via Stable Baselines3
- **Policy:** Shared MLP across all 5 agents (MultiDiscrete action space)
- **Parallelization:** 30 SubprocVecEnv workers on Apple Silicon Mac Mini M4
- **Hyperparameters (fine-tuning phase):**
  - Learning rate: 3e-5
  - Entropy coefficient: 0.002
  - Clip range: 0.08
- **Total training:** 1B+ total iterations across multiple phases

### Training Progression

| Phase | Setup | Algorithm | Result |
|---|---|---|---|
| 1 | 10³ grid, 2v2/3v3 | REINFORCE | Learned basic interception, plateaued |
| 2 | 15³ grid, 3v3 | PPO | Breakthrough to +5.8 avg, 3/3 perfect |
| 3 | 50³ grid, 5v5 | PPO | Iterative reward tuning, speed balancing |
| 4 | 50³ grid, 5v5 | PPO (fine-tuned) | Current best — 97.8% ≤1 breach |

## Visualization

### PyVista Tactical View (`visualize_tactical.py`)

Real-time 3D visualization with:
- Dark tactical aesthetic
- Friendly drones (cyan) and hostile drones (red)
- Wireframe engagement spheres (zone boundaries)
- Drone trails showing recent movement
- Command center cube at base
- Continuous episode loop with stats output


## Getting Started

### Prerequisites

- Python 3.11+
- Apple Silicon Mac (MPS) or CUDA GPU recommended

### Installation

```bash
git clone https://github.com/shreenath04/counter-uas.git
cd counter-uas
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### See the results directly

```bash
python visualize_tactical.py
```

Loads the trained model and runs continuous inference episodes with real-time 3D rendering. Close the window or Ctrl+C to stop.

## Project Structure

```
counter-uas/
├── env.py                  # Core 3D environment with reward shaping
├── sb3_env.py              # Gymnasium wrapper for Stable Baselines3
├── train_ppo.py            # PPO training loop with parallel envs
├── visualize_tactical.py   # PyVista real-time 3D visualization
├── requirements.txt        # Dependencies
└── README.md
```

## Known Limitations

- **Shared policy network:** All 5 agents share a single PPO policy. No centralized critic or agent-specific observations (CTDE).
- **Full observability:** All agents see the complete state. No partial observability or fog-of-war.
- **No action masking:** Dead agents still receive actions (ignored in env, but waste policy capacity).
- **Simplified physics:** Grid-based movement with discrete actions, no aerodynamic modeling.
- **Fixed agent count:** Trained on 5v5 only. Not tested with variable drone counts.
- **No adversarial adaptation:** Hostiles follow fixed trajectories toward base with no evasion.

## What's Next

- MAPPO with centralized critic for proper multi-agent credit assignment
- Partial observability (limited sensor range per drone)
- Action masking for destroyed agents
- Variable drone counts (1-100 per side) for generalization
- Rule-based baselines (closest-first, zone-weighted heuristic) for formal comparison
- Scale to 100³ grid
- Non-kamikaze interceptor option (reusable drones)

## Built With

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — Environment interface
- [PyTorch](https://pytorch.org/) — Neural network backend
- [PyVista](https://pyvista.org/) — 3D tactical visualization
- [NumPy](https://numpy.org/) — Environment computations
- [Matplotlib](https://matplotlib.org/) — 3D animation

## Hardware

Trained entirely on an **Apple Silicon Mac Mini M4** (10-core CPU). No cloud GPU required.
