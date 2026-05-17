# IST 597: Deep Reinforcement Learning — Homeworks
### Penn State University · Spring 2025
**Yogeshvar Reddy Kallam** · `yvk5381@psu.edu`

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green.svg)](https://gymnasium.farama.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📚 Course Overview

This repository contains all homework assignments and implementations from **IST 597: Deep Reinforcement Learning** at Penn State. The course covers the full spectrum of modern RL — from classical MDP theory through deep policy gradient methods, multi-agent systems, and offline RL.

---

## 🗂️ Repository Structure

```
ist597-deep-rl-homeworks/
│
├── IST597_Deep_RL_Complete_Homeworks.ipynb   ← 🌟 Unified notebook (run this!)
│
├── hw1/
│   └── hw1_mdp_dynamic_programming.ipynb    ← Bellman equations, MC, policy improvement
│
├── hw2/
│   ├── hw2_reinforce_baseline.ipynb         ← REINFORCE + variance reduction
│   ├── hw2_warehouse_robot_ppo.ipynb        ← PPO on custom discrete env
│   ├── hw2_partial_obs_cartpole.ipynb       ← PPO under partial observability
│   └── hw2_warehouse_mixed_actorcritic.ipynb← Actor-Critic with mixed actions
│
├── hw3/
│   ├── hw3_hedgemaze_vdn_marl.ipynb         ← Multi-Agent VDN Q-Learning
│   ├── hw3_large_frozenlake_exploration.ipynb← Optimistic exploration bonus
│   └── hw3_offline_behavioral_cloning.ipynb ← Behavioral & Filtered Cloning
│
└── final/
    ├── final_frozenlake_delivery.ipynb      ← REINFORCE on multi-stage task
    └── final_pendulum_qlearning.ipynb       ← Q-Learning with discretization
```

---

## 📋 Homework Breakdown

### HW1 — MDPs & Dynamic Programming
**Environment:** FrozenLake (1-row, then 4×4)

| Topic | Details |
|-------|---------|
| Bellman Equations | Derived closed-form `V^π(S₀) = (Pg − Ph) / (1 − Ps·γ)` |
| Variance Analysis | Computed `Var(G₀)` and min episodes via CLT for ε=0.1 accuracy |
| Policy Evaluation | Iterative MC + exact DP on 4×4 FrozenLake |
| Policy Improvement | Greedy improvement → optimal action per state |

---

### HW2 — Policy Gradient Methods

#### Part 1: REINFORCE with Baseline (Bandit)
**Problem:** 2-armed bandit (rewards: left=2, right=1), softmax policy at θ=0

| | Without Baseline | With Baseline b=1.5 |
|---|---|---|
| E[∇J] | −0.25 | −0.25 ✓ |
| Var[∇J] | **0.5625** | **0.0** 🎯 |

> Baseline b = E[R] reduces gradient variance to **zero** without changing the expected update.

#### Part 2: Warehouse Robot with PPO
Custom Gymnasium environment — 1D corridor (7 positions). Robot must navigate to pick up packages and deliver to correct endpoint.

```
State:  (position ∈ {0..6}, package ∈ {NONE, LEFT, RIGHT})
Action: MOVE_LEFT | MOVE_RIGHT | PICK_UP | DROP_OFF
Reward: +1 correct delivery | −0.1 wrong endpoint
```
**Algorithm:** PPO (Stable-Baselines3, 100k timesteps)

#### Part 3: Partially Observable CartPole
Velocity observations (cart velocity + pole angular velocity) are **zeroed out**, creating a POMDP. Tests how PPO copes with hidden state.

| Observation | Mean Reward |
|---|---|
| Full (4D) | ~500 (max) |
| Partial (pos only) | Reduced |

#### Part 4: Warehouse Robot — Mixed Actions (Actor-Critic)
Action space: `(action_type: Discrete(3), movement: Continuous[-1,1])`

```
Policy Network:
  Shared encoder → Categorical head (MOVE/PICK_UP/DROP_OFF)
                 → Normal distribution head (movement amount)
  Value Network  → Critic for advantage baseline
```

---

### HW3 — Advanced Topics

#### Part 1: HedgeMaze — Multi-Agent VDN Q-Learning
Two agents (Alice & Bob) cooperatively navigate a 7×7 maze to meet.

```
Architecture:
  Q_Alice(s, a_Alice)  ←── individual Q-tables (nn.Parameter)
  Q_Bob(s, a_Bob)      ←──
  Q_tot = Q_Alice + Q_Bob  ← VDN additive decomposition

Training: ε-greedy (1.0 → 0.05 over 15k episodes), Adam optimizer
Shared reward: +1 when both agents occupy same cell
```

#### Part 2: Large FrozenLake — Optimistic Exploration
10×10 FrozenLake with custom hole map. Standard REINFORCE fails due to sparse rewards.

```
Augmented reward = real_reward + β / √(visit_count(state))
                   β = 0.1

Bonus is highest for rarely-visited states → agent explores broadly
Shrinks as state is visited more → natural exploration decay
```

#### Part 3: Offline FrozenLake — Behavioral & Filtered Cloning
Learning from a fixed trajectory dataset (no environment interaction).

| Method | Expected Return |
|---|---|
| Behavior policy π_β | ~0.023 |
| Behavioral Cloning | ≈ π_β (clones bad behavior too) |
| **Filtered Cloning** | **≥ 0.035** (trains only on successes) |

---

### Final Project — Part 1 & 2

#### FrozenLake Delivery (REINFORCE)
5×5 multi-stage grid: `INITIAL → HAS_KEY → HAS_PRESENT → DELIVERED`

```
Reward shaping:
  +0.2  reach key       → guides sparse exploration
  +0.3  unlock chest    → milestone reward
  +1.0  deliver gift    → terminal success
  −1.0  fall in hole    → terminal failure
```

#### Pendulum Q-Learning (Discretization)
Continuous pendulum → discretized Q-table

```
State: [cos(θ), sin(θ), θ̇] → 10×10×10 bins = 1,000 states
Action: [-2, -1, 0, +1, +2] Nm torque = 5 actions
Q-table: shape [10, 10, 10, 5]
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yogeshvarreddykallam/ist597-deep-rl-homeworks.git
cd ist597-deep-rl-homeworks

# Install dependencies
pip install gymnasium[toy-text] torch stable-baselines3 numpy matplotlib tqdm

# Open the unified notebook
jupyter notebook IST597_Deep_RL_Complete_Homeworks.ipynb
```

> **Tip:** Run the unified notebook top-to-bottom — all sections are self-contained and share a global import block at the top.

---

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Neural network policy/value functions |
| `gymnasium` | ≥0.29 | RL environments (FrozenLake, CartPole, Pendulum) |
| `stable-baselines3` | ≥2.0 | PPO implementation |
| `numpy` | ≥1.24 | Numerical computation |
| `matplotlib` | ≥3.7 | Learning curve plots |
| `tqdm` | any | Progress bars |

---

## 📊 Key Results Summary

| HW | Algorithm | Environment | Key Result |
|----|-----------|-------------|------------|
| HW1 | Bellman DP | FrozenLake | V^π(S₀) derived analytically |
| HW2-1 | REINFORCE | 2-arm Bandit | Baseline reduces Var: 0.5625 → 0 |
| HW2-2 | PPO | Warehouse Robot | Learns delivery in 100k steps |
| HW2-3 | PPO | PO CartPole | Degraded performance vs full obs |
| HW2-4 | Actor-Critic | Mixed-Action Warehouse | Categorical+Normal dual-head policy |
| HW3-1 | VDN Q-Learning | HedgeMaze | Agents learn to meet cooperatively |
| HW3-2 | REINFORCE+Bonus | 10×10 FrozenLake | Exploration bonus finds sparse goal |
| HW3-3 | Filtered BC | Offline FrozenLake | Return improved from 0.023 → ≥0.035 |
| Final | REINFORCE | FrozenLake Delivery | Multi-stage task solved with shaping |
| Final | Q-Learning | Pendulum | Discretized continuous control |

---

## 🎓 Course Information

**Course:** IST 597 — Deep Reinforcement Learning  
**University:** Penn State University  
**Semester:** Spring 2025  
**Instructor:** IST Department  

Topics covered: Multi-Armed Bandits · MDPs · Dynamic Programming · Monte Carlo · TD Learning · Q-Learning · Value Function Approximation · Deep Q-Networks · Policy Gradient · Actor-Critic · PPO · TRPO · Off-Policy PG · Multi-Agent RL · Adversarial MARL · Exploration · Imitation Learning · Offline RL · MCTS · Model-Based RL

---

*Part of a broader RL portfolio — see also [deep-rl-news-recommendation](https://github.com/yogeshvarreddykallam/deep-rl-news-recommendation) for the final research project.*
