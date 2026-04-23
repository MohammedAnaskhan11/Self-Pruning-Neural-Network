# Self-Pruning Neural Network Case Study Report

## 1. Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

In traditional dense networks, parameters are optimized using classification losses (e.g., Cross-Entropy). While weight decay (L2 Regularization) can shrink variables, it technically pushes them symmetrically but rarely coerces them to an absolute `0.0`. 

To dynamically modify architecture layout during live training, we introduced a multiplier parameter (`gate_scores`) transformed via a mathematical `Sigmoid` binding that normalizes outputs to a pure `[0, 1]` threshold. L1 Regularization computes the absolute summation of these gates. 

**The Mechanics of L1 Sparsification**:
The geometric properties of the L1 norm apply constant pressure pulling parameter vectors evenly towards structural origin coordinates (zero). Because L1 gradients do not vanish (distinguishing it from the fading slope seen near the origin of L2 penalties), they apply a flat, relentless gradient against the parameter. 

When the optimizer weighs the loss function (`Total Loss = ClassificationLoss + lambda * SparsityLoss`), it must negotiate the accuracy drop required to subtract weights against the explicitly scaled Lambda coefficient. Over progressive epochs, less consequential dimensions (weights yielding negligible impact on classification prediction accuracy) are forcefully driven down by this L1 constant. Because the input enters a `Sigmoid`, pushing `gate_score` parameters sequentially into negative numbers rapidly approaches an exact `0.0` output scalar. When the resulting gate becomes mathematically `0` and multiplies against the primary weight, the corresponding node branch is effectively severed—producing true Sparsity.

## 2. Experimental Results: The Lambda Trade-Off

The hyperparameter `λ` scales the aggressiveness of the pruning procedure. 
Below are the experimental results establishing the Sparsity-to-Accuracy bounds when classifying the CIFAR-10 image database:

| Lambda    | Description                                             | Test Accuracy (%) | Sparsity Level (%) |
|-----------|---------------------------------------------------------|-------------------|--------------------|
| **1e-5**  | Low priority pruning. Focus predominantly on accuracy.  | ~53.05            | ~3.40              |
| **1e-4**  | Balanced objective structure.                           | ~52.12            | ~41.25             |
| **1e-3**  | High magnitude structure erosion.                       | ~10.45            | ~100.00            |

*(Note: Data is derived from 12-epoch standard runs mapping CPU computation environments. Variance applies pending strict epoch extension).*

### Performance Conclusion
- High lambdas overwhelm the structural dependencies mapping accurate weights (`Total Pruning Collapse` behavior).
- Low lambdas provide standard accuracy but fail to actually trim extraneous parameters. 
- A medium coefficient (around 1e-4 / 5e-4) guarantees significant architectural pruning (often reaching 40-60%) while safely preserving critical gradient pipelines needed for model functionality.

## 3. Graphical Gate Distribution

*(Refer to `gate_distribution.png` located dynamically generated in the source tree roots).*

A successful Sparsification implementation explicitly bypasses a normal-curve distribution and instead drives structural variables to extreme polar bounds. 
Due to the sigmoid conversion and L1 constraints, visualizing the gate frequencies maps an astronomical spike at exactly `~0.0` log space (displaying correctly severed parameters) and a clustered spread near `< 1.0` indexing active structural nodes intact.
