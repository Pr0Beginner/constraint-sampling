# A Feasible Region-Guided Constrained Sampling Algorithm for Design of Experiments with Non-Convex Inequality Constraints (FGCS)

[![Paper](https://img.shields.io/badge/Springer-CCIS%20%2F%20CSC%202026-blue)](https://link.springer.com/book/10.1007/978-981-92-5712-6)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![Solver](https://img.shields.io/badge/Solver-CasADi%20%2B%20IPOPT-orange)](https://web.casadi.org/)

Official Python implementation of the paper:

> **A Feasible Region-Guided Constrained Sampling Algorithm for Design of Experiments with Non-Convex Inequality Constraints**  
> Published in **[38th China Simulation Conference (CSC 2026), Springer CCIS](https://link.springer.com/book/10.1007/978-981-92-5712-6)**  
> Conference Proceedings Link: [https://link.springer.com/book/10.1007/978-981-92-5712-6](https://link.springer.com/book/10.1007/978-981-92-5712-6)

---

## 📖 Overview (项目简介)

Design of Experiments (DoE) for high-fidelity simulations often faces challenges from complex nonlinear constraints that warp the feasible space into irregular, narrow, or disconnected topologies. Existing heuristic sampling strategies lack deterministic guidance mechanisms during iteration, leading to slow convergence and local sample aggregation.

This repository implements the **Feasible Region-Guided Constrained Sampling (FGCS)** algorithm targeting complex non-convex inequality constraints. FGCS couples global evolutionary search with the deterministic constraint-solving capability of **IPOPT** (via **CasADi** symbolic automatic differentiation) and tangent-space manifold projection:

- **Adaptive Reference Vector Guidance (自适应参考向量引导)**: Dynamically evaluates directional sample sparsity and historical feasibility success rates via Exponential Moving Average (EMA) to deterministically guide samples toward under-explored feasible regions.
- **Local Energy Gradient & Neighborhood Repulsion (局部能量梯度与邻域排斥)**: Combines reference vector guidance with distance-based repulsive forces to prevent sample clustering along irregular constraint boundaries.
- **Tangent Space Mapping & Nonlinear Solver Boundary Control (切空间映射与非线性求解器边界控制)**: Projects desired movement directions onto the null space of the active constraint Jacobian ($\mathbf{d}_{\text{tangent}} = \mathbf{Z}\mathbf{Z}^T \mathbf{d}_{\text{desired}}$) for feasible boundary exploration, while adaptively blending search and gradient-repair directions (backed by IPOPT projection) for infeasible samples.

---

## 📂 Repository Structure (目录结构)

```text
constraint-sampling/
├── ConstraintOptimizer/                 # Core FGCS algorithm and experimental suites
│   ├── EvolutionaryOptimizer.py         # Main FGCS evolutionary optimizer
│   ├── ConstraintHandler.py             # CasADi + IPOPT symbolic constraint handler & projection solver
│   ├── FactorEncoder.py                 # Variable encoding/decoding for [0, 1]^D standard hypercube
│   ├── VariationManager.py              # Tangent-direction & energy-guided variation operators
│   ├── demo.py                          # Minimal quick-start demo on constrained problems
│   ├── eo_test.py                       # Benchmark evaluation script (CEC 2006 constrained problems)
│   ├── StageA/                          # Paper experiments, ablation studies, and visualization scripts
│   │   ├── EvolutionaryOptimizer.py     # FGCS optimizer for paper benchmark evaluations
│   │   ├── ExpEvolutionaryOptimizer.py  # Configurable optimizer supporting module ablation studies
│   │   ├── BaseLine.py                  # Baseline rejection-based Latin Hypercube Sampling (LHS)
│   │   ├── eo_test.py                   # Full benchmark tests (E1-E5 / CEC 2006)
│   │   ├── eo_ablation_test.py          # Ablation experiments (FGCS w/o E-Rep & FGCS w/o CO)
│   │   └── lhd_test.py                  # Baseline LHS benchmark evaluations
│   └── StageA_Multithread/              # Multi-threaded evaluation suite
├── Criterion/                           # Space-filling and uniformity evaluation criteria
│   ├── BasicCriterion.py                # Abstract base class for design criteria
│   ├── PhiPCriterion.py                 # Phi_p space-filling criterion (used in paper)
│   ├── MaxMinDistanceCriterion.py       # Maximin distance criterion
│   ├── MaxProCriterion.py               # Maximum projection criterion
│   ├── CDCriterion.py                   # Centered L2-discrepancy criterion
│   └── WDCriterion.py                   # Wrap-around L2-discrepancy criterion
├── utils/                               # General helper utilities and constraint handlers
│   └── utils.py                         # Feasibility verification and visualization utilities
├── requirements.txt                     # Python package dependencies
└── README.md
```

---

## ⚙️ Installation (环境安装)

1. Clone the repository:
   ```bash
   git clone https://github.com/Pr0Beginner/constraint-sampling.git
   cd constraint-sampling
   ```

2. Install the required dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start (快速开始)

### 1. Run a Quick Demo
```python
from ConstraintOptimizer.EvolutionaryOptimizer import EvolutionaryOptimizer

# Define experimental factors and bounds
factors = {
    'x1': {'type': 'cont', 'bounds': [0, 1]},
    'x2': {'type': 'cont', 'bounds': [0, 1]}
}

# Define non-convex inequality constraints (feasible when fun(x) >= 0)
constraints = [
    {'type': 'ineq', 'fun': lambda x: -(1 - (x['x1'] - 1) ** 2 - (x['x2'] - 1) ** 2)}
]

# Initialize and run the FGCS optimizer
optimizer = EvolutionaryOptimizer(
    factors=factors,
    constraints=constraints,
    sample_num=20,
    max_generations=1500
)
samples = optimizer.optimize()
print("Generated uniform constrained samples:\n", samples)
print("Final Phi_p uniformity score:", optimizer.best_val)
```

### 2. Run Paper Benchmarks & Ablation Studies
```bash
# Run main benchmark experiments (E1-E5)
python ConstraintOptimizer/StageA/eo_test.py

# Run ablation study experiments (FGCS w/o E-Rep and FGCS w/o CO)
python ConstraintOptimizer/StageA/eo_ablation_test.py
```

---

## 📝 Citation (论文引用)

If you find this code or our paper useful in your research, please consider citing our work published in [Springer CCIS (CSC 2026)](https://link.springer.com/book/10.1007/978-981-92-5712-6):

```bibtex
@inproceedings{zhao2026feasible,
  title     = {A Feasible Region-Guided Constrained Sampling Algorithm for Design of Experiments with Non-Convex Inequality Constraints},
  author    = {Zhao, Mingyu and Lu, Yingbo and Duan, Zhixing and Zhou, Benben and Liu, Fei},
  booktitle = {Intelligent Simulation -- 38th China Simulation Conference (CSC 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer Nature Singapore},
  year      = {2026},
  url       = {https://link.springer.com/book/10.1007/978-981-92-5712-6}
}
```
