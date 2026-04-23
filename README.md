# Tredence AI Engineering - The Self-Pruning Neural Network

This repository contains the completed **AI Engineering Intern Case Study** submission.

## Objective
The task requires building a neural network that dynamically identifies and removes its own weakest connections during the training phase, bypassing standard post-training ad-hoc manual pruning limits. 

The approach involves appending an explicitly bounded continuous mathematical Gate to each parameter and applying Sparsity L1 Regularization to automatically drive obsolete variables to 0.

## Project Structure
- `self_pruning_model.py`: The core standalone codebase featuring the custom `PrunableLinear` PyTorch layer, dataset orchestration, standard training logic, sparsity penalty math, and result evaluations.
- `requirements.txt`: Package dependencies.
- `REPORT.md`: Evaluative theory breakdown of how L1 norms influence Sigmoid outputs to produce localized mathematical Zero states, plus statistical summaries of Lambda variation.
- `gate_distribution.png`: Empirically generated histogram chart visually graphing the polarized effect of the Gates on standard PyTorch architecture.
- `results/result_table.md`: Generated table mapping Accuracy to Lambda constraints.

## Execution Requirements
1. **Python 3.8+**
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the core system:
```bash
python self_pruning_model.py
```
> The script will automatically trigger a multi-param loop, download the CIFAR-10 data, execute classification epochs for each lambda, save evaluations, drop distribution pngs, and establish validation tables in standard output structure.
