# 🚀 HSS-MEoH: Designing Hypervolume Subset Selection Algorithms with Large Language Models

This repository contains the official implementation of **HSS-MEoH**, an LLM-based framework for automatically designing Hypervolume Subset Selection (HSS) algorithms.

📄 Paper: *Designing Hypervolume Subset Selection Algorithms with Large Language Models* 

------

## 📌 Overview

Hypervolume Subset Selection (HSS) is a fundamental problem in evolutionary multi-objective optimization, aiming to select a subset of solutions that maximizes the hypervolume (HV) indicator.

However, HSS is:

- **NP-hard** in ≥3 objectives
- Computationally expensive due to **HV evaluation**
- Highly complex due to **nonlinear interactions among selected solutions**

To address these challenges, we propose **HSS-MEoH**, the first framework that leverages **Large Language Models (LLMs)** for **automatic HSS algorithm design**.

------

## 🧠 Key Idea

We formulate algorithm design itself as a **multi-objective optimization problem**:

- 🎯 Objective 1: Maximize solution quality (Hypervolume)
- ⚡ Objective 2: Minimize runtime

Instead of designing a single algorithm, HSS-MEoH discovers a **Pareto set of algorithms** with different trade-offs.

------

## ✨ Key Contributions

- 🔥 First LLM-based framework for **HSS algorithm design**

- 🌱 

  Diverse Warm-Start Initialization (DWS-Init)

  - Uses multiple classical HSS algorithms as seeds

- 🔁 

  Reflective Evolution Mechanism (REM)

  - Enables LLM to learn from past generated heuristics

- 🎲 

  Novelty-Weighted Parent Selection (NWPS)

  - Balances exploitation and exploration using novelty

------

## ⚙️ Framework Pipeline

The HSS-MEoH framework follows an evolutionary process:

```
Initialization → Selection → Reflection → Crossover → Mutation → Evaluation → Population Update
```

Key characteristics:

- Individuals are represented as **LLM-generated code**
- Evolution is guided by **performance + diversity**
- Reflection provides **high-level feedback for refinement**

------

## 🧪 Experimental Setup

- Candidate set size: `n = 200`
- Objectives: `m = 3, 4, 5`
- Subset size: `k = 5, 10, 15`
- Population size: `10`
- Generations: `15`
- LLM: `DeepSeek-V3.2`

------

## 📊 Results

### ✅ Performance Highlights

- Achieves **near-TPOSS-level hypervolume**
- Provides **~100× speedup** in runtime
- Produces **denser and more diverse Pareto fronts** than MEoH

### ⚖️ Trade-off Advantage

| Method       | Quality | Runtime | Balance |
| ------------ | ------- | ------- | ------- |
| TPOSS        | ⭐⭐⭐⭐    | ❌ Slow  | Poor    |
| MEoH         | ⭐⭐      | ⭐⭐⭐⭐    | Medium  |
| **HSS-MEoH** | ⭐⭐⭐⭐    | ⭐⭐⭐⭐    | ✅ Best  |

------

## 📁 Repository Structure (Example)

```
.
├── llm4ad/             # Core framework
│   ├── base/           # Base classes and shared abstractions
│   ├── method/         # Method implementations
│   └── tools/          # Utility tools
├── HV_cal/             # HSS baselines and HV evaluation
│   ├── benchmark_hss_train.py
│   ├── benchmark_hss_test.py
│   └── GAHSS/GHSS/GL_HSS/GSI_LS/... baseline implementations
├── data/               # Input datasets
│   ├── train_data/     # Training instances
│   └── test_data/      # Test instances
├── example/            # Reproducible experiment scripts
│   └── method_MEoH/    # MEoH/HSS-MEoH running and plotting examples
├── LICENSE             # License file
└── README.md           # Documentation
```

------

## 🚀 Getting Started

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run algorithm design

```
python run_evolution.py
```

### 3. Evaluate heuristics

```
python evaluate.py
```

------

## 📈 Supported Baselines

- TPOSS
- SPESS
- GHSS / GAHSS
- GSI_LS
- GL_HSS
- MEoH

------

## 🔬 Key Components

### 1. DWS-Init

- Initializes population using diverse classical algorithms
- Improves starting quality and diversity

### 2. REM (Reflection)

- Summarizes useful patterns from previous generations
- Guides future generation of heuristics

### 3. NWPS

- Selection probability:
  - Combines **fitness** and **novelty (offspring count)**
  - Encourages exploration of underused heuristics

------

## 🤝 Acknowledgements

This project is inspired by and built upon the excellent open-source project:

👉 https://github.com/Optima-CityU/LLM4AD

We sincerely thank the authors for their contributions to LLM-based automatic algorithm design, which greatly supported and inspired this work.

------

## 📜 Citation

If you find this work useful, please cite:

```
@article{hss_meoh,
  title={Designing Hypervolume Subset Selection Algorithms with Large Language Models},
  author={Anonymous Authors},
  journal={},
  year={2025}
}
```

------

## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or contact us.