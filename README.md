
# Belief Space Planning (BSP)

This repository contains an implementation of algorithms for motion planning under uncertainty in belief space. The focus is on using sampling-based techniques like Belief Roadmap (BRM) and Rapidly-exploring Random Belief Trees (RRBT), coupled with stabilizing controllers, to compute dynamically feasible and uncertainty-aware trajectories.
This implementation is very basic, and is not modular in the sense that you can plug in whatever motion/measurement model you want. Even the implementation of the controller is quite simplistic so do not trust this implementation wholeheartedly. Even the implementation of the algorithms themselves is not one-to-one with the papers so make sure to check my work if you intend on using this! 

This code was produced for a course project at Queen's University for the Path Planning and Robotic Algorithms course, ELEC 844.

## Table of Contents
- [Belief Space Planning (BSP)](#belief-space-planning-bsp)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Results and Visualizations](#results-and-visualizations)
  - [Acknowledgements](#acknowledgements)

## Introduction

Motion planning in the presence of uncertainty requires navigating both physical constraints (e.g., obstacles) and probabilistic ones (e.g., state estimation errors). Traditional methods like RRTs or PRMs assume deterministic state knowledge, which is unrealistic in many real-world applications. To address this, we implement:
	1.	Belief Roadmap (BRM): Extends the Probabilistic Roadmap approach to belief space, leveraging Kalman filtering for uncertainty prediction.
	2.	Rapidly-exploring Random Belief Trees (RRBT): Builds on RRTs by incorporating chance constraints to ensure safe and feasible trajectories under uncertainty.

These methods rely on nominal trajectories stabilized using Linear Quadratic Regulators (LQR) or similar controllers. The repository provides implementations, evaluations, and visualizations for these approaches.

## Project Structure
```
.
├── src/                 # Source code for various components
│   ├── Controller/      # Contains LQR-based controllers
│   ├── RandomTrees/     # Implementations of RRBT and RRT
│   ├── Roadmaps/        # BRM and related roadmap algorithms
│   ├── utils/           # Utility functions
│   └── main.py          # Entry point for the project
├── figures/             # Debug and result visualization images
├── results/             # Resulting data from experiments
├── papers/              # Relevant research papers
├── tex/                 # LaTeX files for documentation
├── run_main.sh          # Shell script for running the main file
└── README.md            # Project documentation
```

## Usage
1.	**Run the Main Script:**
    ```bash
    python src/main.py
    ```
    This script sets up the environment, initializes the planner, and executes the planning algorithms. Results will be saved in the `results/` directory.

2. **Visualization:**
Results and trajectories are visualized using Matplotlib and stored in the figures/ directory. To debug or fine-tune parameters, modify the configuration in main.py.

3.	**Configuration:**
	•	Adjust dynamics models, uncertainty parameters, or planner settings in the respective modules within src/.

## Results and Visualizations
- **Figures:** The `figures/` directory contains visualizations of the computed belief-space trajectories. These include:
  - Covariance ellipses showing uncertainty propagation.
  - Paths avoiding high-uncertainty regions while ensuring collision-free trajectories.
- **Results:** Detailed performance metrics, including computation times, path costs, and uncertainty measures, are saved as .csv files in the `results/` directory.

## Acknowledgements

This project was built as part of coursework at Queen’s University under the Electrical and Computer Engineering department. The implementation was inspired by and extends algorithms from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics).

Citations

If you find this work helpful, please consider citing the following references:
1.	**Belief Roadmap (BRM):**
    - Prentice, S., & Roy, N. (2009). The Belief Roadmap: Efficient Planning in Belief Space by Factoring the Covariance. The International Journal of Robotics Research, 28(11-12), 1448–1465. [DOI:10.1177/0278364909341659]([0278364909341659](http://dx.doi.org/10.1177/0278364909341659))
2.	**Rapidly-exploring Random Belief Trees (RRBT):**
	- Bry, A., & Roy, N. (2011). Rapidly-exploring Random Belief Trees for Motion Planning Under Uncertainty. IEEE International Conference on Robotics and Automation. [DOI:10.1109/ICRA.2011.5980480](https://doi.org/10.1109/ICRA.2011.5980480)
3.	**PythonRobotics:**
	- Atsushi Sakai et al. PythonRobotics: A Python code collection of robotics algorithms. [GitHub Repository](https://github.com/AtsushiSakai/PythonRobotics)