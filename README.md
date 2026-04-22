\\textbf{Due to the large size of the original paper, a compressed version was used when submitting it. The complete version can be downloaded from the current website - "Designing Hypervolume Subset Selection Algorithms with Large Language Models"}

# 🚀 HSS-MEoH: Designing Hypervolume Subset Selection Algorithms with Large Language Models

This repository contains the official implementation of **HSS-MEoH**, the first LLM-based framework for automatically designing Hypervolume Subset Selection (HSS) algorithms.

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

Instead of designing a single algorithm, HSS-MEoH discovers a **well-distributed set of non-dominated HSS algorithms** with different performance and runtime trade-offs.

------

## ✨ Key Contributions

- 🔥 We propose the first LLM-based framework for automatic design of HSS algorithms by formulating the problem as a multi-objective optimization task over algorithm performance and runtime.


* ⚙️ We incorporate three mechanisms, namely **Diverse Warm-Start Initialization (DWS-Init)**, **Reflective Evolution Mechanism (REM)**, and **Novelty-Weighted Parent Selection (NWPS)**, into the proposed framework, which together improve the quality, diversity, and search efficiency of the discovered non-dominated HSS algorithms.

* 🧪 We conduct extensive experiments to show that the algorithms designed by **HSS-MEoH** consistently outperform those designed by **MEoH** and most human-designed HSS algorithms in terms of performance and runtime trade-offs.

------

## ⚙️ Framework Pipeline

The HSS-MEoH framework follows an evolutionary process:

```text
Initialization → Selection → Reflection → Crossover → Mutation → Evaluation → Population Update
```

Key characteristics:

- Individuals are represented as **LLM-generated code**
- Evolution is guided by **performance + diversity**
- Reflection provides **high-level feedback for refinement**

------

## 🧪 Experimental Setup

### Design stage
- Training candidate set size: `n = 200`
- Objectives: `m = 4`
- Subset size: `k = 8`
- Population size: `10`
- Generations: `15`
- LLM: `DeepSeek-V3.2`

### Generalization stage
- Candidate set sizes: `n = 100, 200, 300`
- Objectives: `m = 3, 4, 5`
- Subset sizes: `k = 5, 10, 15`
- Total test settings: `27`

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

## 📁 Repository Structure

```text
.
├── data/
├── HSS_benchmark/          # Baselines and benchmark scripts for HSS
├── llm4ad/                 # Trimmed LLM4AD core used by MEoH
│   ├── base/
│   ├── method/
│   │   └── meoh/
│   └── tools/
├── meoh_hss/
│   ├── core/               # Evaluation/template helpers
│   ├── scripts/            # Main experiment and evaluation scripts
│   ├── plots/              # Plotting scripts
│   └── results/
│       ├── image/          # Generated figures
│       ├── logs/           # Search logs and archives
│       ├── result_code/    # Extracted discovered algorithms
│       ├── test_result/    # Generalization results on test settings
│       └── train_result/   # Design-stage / training-stage results
├── LICENSE
└── README.md
```

------

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run algorithm design

```bash
python meoh_hss/scripts/run_meoh_hss.py
```

### 3. Extract discovered algorithms

```bash
python meoh_hss/scripts/extract_elitist_code.py
```

### 4. Evaluate heuristics on training instances

```bash
python HSS_benchmark/benchmark_hss_train.py
```

### 5. Evaluate heuristics on test instances

```bash
python HSS_benchmark/benchmark_hss_test.py
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

```bibtex
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
