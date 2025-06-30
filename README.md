# Support Vector Machine (SVM) from Scratch

An educational implementation of Support Vector Machines using only NumPy and Matplotlib, designed to help you understand SVM theory deeply.

## What You'll Learn

This implementation covers all the fundamental SVM concepts:

### 1. **SVM Theory & Math**
- **Dual Formulation**: The optimization problem that SVM solves
- **Lagrange Multipliers (α)**: How they determine support vectors
- **Decision Function**: `f(x) = Σ(αᵢ * yᵢ * K(xᵢ, x)) + b`
- **KKT Conditions**: Optimality conditions for the solution

### 2. **Kernel Functions**
- **Linear Kernel**: `K(xᵢ, xⱼ) = xᵢᵀ * xⱼ`
- **Polynomial Kernel**: `K(xᵢ, xⱼ) = (γ * xᵢᵀ * xⱼ + c₀)^d`
- **RBF (Gaussian) Kernel**: `K(xᵢ, xⱼ) = exp(-γ * ||xᵢ - xⱼ||²)`

### 3. **SMO Algorithm**
- Sequential Minimal Optimization for solving the quadratic programming problem
- How to select and optimize pairs of Lagrange multipliers
- Bias term computation and convergence criteria

### 4. **Support Vectors**
- Points that lie exactly on the margin boundaries
- Why only these points matter for the final decision boundary
- How sparsity emerges naturally from the optimization

## Key Results from the Demo

### Linear SVM (Linearly Separable Data)
- ✅ **100% accuracy** with only **11% support vectors**
- Demonstrates how SVM finds the optimal hyperplane with maximum margin

### Non-Linear Data (Concentric Circles)
- **Linear Kernel**: 68% accuracy (expected failure on non-linear data)
- **Polynomial Kernel**: 65% accuracy 
- **RBF Kernel**: 100% accuracy (perfect for circular patterns)

### Hyperparameter Effects (C parameter)
- **Low C (0.1)**: More regularization → More support vectors → Simpler boundary
- **High C (100.0)**: Less regularization → Fewer support vectors → Complex boundary

## Running the Code

```bash
# Install dependencies
pip3 install --break-system-packages -r requirements.txt

# Run the full demonstration
python3 main.py
```

## Code Structure

```python
class SVM:
    def _kernel_function()     # Implements different kernel types
    def _simplified_smo()      # SMO algorithm for optimization
    def _decision_function()   # Compute decision values
    def fit()                  # Train the SVM
    def predict()              # Make predictions
```

## Educational Value

This implementation helps you understand:

1. **Why SVM works**: Maximum margin principle and geometric intuition
2. **How kernels work**: Mapping to higher-dimensional spaces
3. **What support vectors are**: Critical points that define the boundary
4. **Optimization details**: How SMO solves the dual problem
5. **Hyperparameter effects**: Bias-variance tradeoff with C parameter

## Theory Summary

- **Objective**: Find hyperplane that maximizes margin between classes
- **Primal Problem**: Minimize `½||w||² + C*Σξᵢ` subject to constraints
- **Dual Problem**: Maximize `Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)` subject to `0 ≤ αᵢ ≤ C`
- **Support Vectors**: Points with `αᵢ > 0` (lie on or inside margin)
- **Decision**: `sign(f(x))` where `f(x) = Σ(αᵢyᵢK(xᵢ,x)) + b`

Perfect for learning SVM theory without the abstractions of scikit-learn! 