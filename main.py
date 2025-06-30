"""
Support Vector Machine (SVM) Implementation from Scratch
Educational implementation to understand SVM theory deeply
Uses only NumPy and Matplotlib - no scikit-learn or other ML libraries
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class SVM:

    def __init__(self, C: float = 1.0, kernel: str = 'linear', 
                 gamma: float = 1.0, degree: int = 3, coef0: float = 0.0,
                 tol: float = 1e-3, max_iter: int = 1000):

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
       kernel function'ları data'yı lineer bir ayrışmanın olabileceği yüksek bir dimensiona çıkarıyor
        """
        if self.kernel == 'linear':
            # K(x_i, x_j) = x_i^T * x_j
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'polynomial':
            # K(x_i, x_j) = (gamma * x_i^T * x_j + coef0)^degree
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            # K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
            # Gaussian/RBF kernel
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
                
            sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                     np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dist)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _objective_function(self, alphas: np.ndarray) -> float:
        """
        SVM dual objective function to maximize:
        
        W(α) = Σα_i - (1/2)ΣΣα_i*α_j*y_i*y_j*K(x_i,x_j)

        optimizasyonu kolaylaştırmak için dual form'a dönüştürüyoruz
        
        """
        K = self._kernel_function(self.X_train, self.X_train)
        return np.sum(alphas) - 0.5 * np.sum(
            alphas[:, None] * alphas[None, :] * 
            self.y_train[:, None] * self.y_train[None, :] * K
        )
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for samples X
        
        The decision function is:
        f(x) = Σ(α_i * y_i * K(x_i, x)) + b
        
        where α_i are the Lagrange multipliers, y_i are the labels,
        K is the kernel function, and b is the bias term.
        """
        if self.support_vectors is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        K = self._kernel_function(X, self.support_vectors)
        return np.sum(
            self.support_vector_alphas[:, None] * 
            self.support_vector_labels[:, None] * K.T, 
            axis=0
        ) + self.b
    
    def _simplified_smo(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Simplified Sequential Minimal Optimization (SMO) algorithm
        
        SMO solves the SVM optimization problem by iteratively optimizing
        pairs of Lagrange multipliers while keeping all others fixed.
        
        The optimization problem is:
        maximize: W(α) = Σα_i - (1/2)ΣΣα_i*α_j*y_i*y_j*K(x_i,x_j)
        subject to: 0 ≤ α_i ≤ C and Σα_i*y_i = 0
        """
        n_samples = X.shape[0]
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # Precompute kernel matrix for efficiency
        K = self._kernel_function(X, X)
        
        # SMO main loop
        for iteration in range(self.max_iter):
            alpha_prev = alphas.copy()
            
            for i in range(n_samples):
                # Calculate error for example i
                f_i = np.sum(alphas * y * K[i, :]) + b
                E_i = f_i - y[i]
                
                # Check KKT conditions
                # If alpha_i = 0, then y_i * f_i >= 1
                # If 0 < alpha_i < C, then y_i * f_i = 1
                # If alpha_i = C, then y_i * f_i <= 1
                
                if ((y[i] * E_i < -self.tol and alphas[i] < self.C) or
                    (y[i] * E_i > self.tol and alphas[i] > 0)):
                    
                    # Select second alpha randomly (simplified heuristic)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for example j
                    f_j = np.sum(alphas * y * K[j, :]) + b
                    E_j = f_j - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Compute bounds L and H for alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative of objective function)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alphas[j] = alphas[j] - (y[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha_j to [L, H]
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    # If change in alpha_j is too small, skip
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute bias b
                    b1 = (b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - 
                          y[j] * (alphas[j] - alpha_j_old) * K[i, j])
                    
                    b2 = (b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - 
                          y[j] * (alphas[j] - alpha_j_old) * K[j, j])
                    
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            
            # Check for convergence
            if np.allclose(alphas, alpha_prev, atol=self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return alphas, b
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM classifier
        
        The SVM optimization problem in dual form:
        maximize: W(α) = Σα_i - (1/2)ΣΣα_i*α_j*y_i*y_j*K(x_i,x_j)
        subject to: 0 ≤ α_i ≤ C and Σα_i*y_i = 0

        """
        # Validate input
        X = np.array(X)
        y = np.array(y)
        
        # Ensure binary classification with labels -1, 1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM supports only binary classification")
        
        # Convert labels to -1, 1 if necessary
        if not np.array_equal(np.sort(unique_labels), [-1, 1]):
            y = np.where(y == unique_labels[0], -1, 1)
        
        self.X_train = X
        self.y_train = y
        
        print("Training SVM...")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Kernel: {self.kernel}, C: {self.C}")
        
        # Solve optimization problem using simplified SMO
        self.alphas, self.b = self._simplified_smo(X, y)
        
        # Extract support vectors (points with alpha > 0)
        support_indices = self.alphas > 1e-8
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        self.support_vector_alphas = self.alphas[support_indices]
        
        print(f"Number of support vectors: {len(self.support_vectors)}")
        print(f"Support vector ratio: {len(self.support_vectors)/len(X):.2%}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X
        
        Uses the decision function: sign(f(x)) where
        f(x) = Σ(α_i * y_i * K(x_i, x)) + b
        """
        decision_values = self._decision_function(X)
        return np.sign(decision_values)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return the decision function value for samples in X
        
        The distance from the separating hyperplane
        """
        return self._decision_function(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_linearly_separable_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable 2D data for testing"""
    np.random.seed(42)
    
    # Generate two clusters
    X1 = np.random.normal([2, 2], [1, 1], (n_samples//2, 2))
    X2 = np.random.normal([-2, -2], [1, 1], (n_samples//2, 2))
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
    
    # Add some noise
    X += np.random.normal(0, noise, X.shape)
    
    return X, y


def generate_nonlinear_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate non-linearly separable data (concentric circles)"""
    np.random.seed(42)
    
    # Inner circle (class 1)
    theta1 = np.random.uniform(0, 2*np.pi, n_samples//2)
    r1 = np.random.uniform(0, 1, n_samples//2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    
    # Outer circle (class -1)
    theta2 = np.random.uniform(0, 2*np.pi, n_samples//2)
    r2 = np.random.uniform(2, 3, n_samples//2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
    
    return X, y


def plot_svm_decision_boundary(svm: SVM, X: np.ndarray, y: np.ndarray, 
                              title: str = "SVM Decision Boundary"):
    """
    Visualize SVM decision boundary and support vectors
    """
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get decision function values for the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.75, 
                linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), 
                 alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
    
    # Highlight support vectors
    if svm.support_vectors is not None:
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=200, linewidth=2, facecolors='none', edgecolors='black',
                   label=f'Support Vectors ({len(svm.support_vectors)})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title}\nKernel: {svm.kernel}, C: {svm.C}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def demonstrate_linear_svm():
    """Demonstrate SVM on linearly separable data"""
    print("=" * 60)
    print("DEMONSTRATION 1: LINEAR SVM ON LINEARLY SEPARABLE DATA")
    print("=" * 60)
    
    # Generate linearly separable data
    X, y = generate_linearly_separable_data(n_samples=100, noise=0.1)
    
    # Train SVM with linear kernel
    svm_linear = SVM(C=1.0, kernel='linear')
    svm_linear.fit(X, y)
    
    # Calculate accuracy
    accuracy = svm_linear.score(X, y)
    print(f"Training Accuracy: {accuracy:.2%}")
    
    # Plot results
    plot_svm_decision_boundary(svm_linear, X, y, "Linear SVM")
    plt.show()
    
    return svm_linear, X, y


def demonstrate_nonlinear_svm():
    """Demonstrate SVM with different kernels on non-linear data"""
    print("=" * 60)
    print("DEMONSTRATION 2: NONLINEAR SVM ON CIRCULAR DATA")
    print("=" * 60)
    
    # Generate non-linearly separable data
    X, y = generate_nonlinear_data(n_samples=200)
    
    # Test different kernels
    kernels = [
        ('linear', {}),
        ('polynomial', {'degree': 3, 'gamma': 1.0}),
        ('rbf', {'gamma': 1.0})
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, (kernel_name, kernel_params) in enumerate(kernels):
        plt.subplot(1, 3, i+1)
        
        print(f"\nTesting {kernel_name.upper()} kernel:")
        
        # Train SVM
        svm = SVM(C=1.0, kernel=kernel_name, **kernel_params)
        svm.fit(X, y)
        
        # Calculate accuracy
        accuracy = svm.score(X, y)
        print(f"Training Accuracy: {accuracy:.2%}")
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        
        colors = ['red' if label == -1 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=30)
        
        if svm.support_vectors is not None:
            plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                       s=100, linewidth=1, facecolors='none', edgecolors='black')
        
        plt.title(f'{kernel_name.title()} Kernel\nAccuracy: {accuracy:.2%}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()


def demonstrate_hyperparameter_effects():
    """Demonstrate the effect of different hyperparameters"""
    print("=" * 60)
    print("DEMONSTRATION 3: HYPERPARAMETER EFFECTS")
    print("=" * 60)
    
    # Generate data with some noise
    X, y = generate_linearly_separable_data(n_samples=100, noise=0.3)
    
    # Test different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    plt.figure(figsize=(20, 5))
    
    for i, C in enumerate(C_values):
        plt.subplot(1, 4, i+1)
        
        print(f"\nTesting C = {C}:")
        
        svm = SVM(C=C, kernel='linear')
        svm.fit(X, y)
        
        accuracy = svm.score(X, y)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Support vectors: {len(svm.support_vectors)}")
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'], 
                   linestyles=['--', '-', '--'], linewidths=1)
        
        colors = ['red' if label == -1 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=30)
        
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=100, linewidth=1, facecolors='none', edgecolors='black')
        
        plt.title(f'C = {C}\nSV: {len(svm.support_vectors)}, Acc: {accuracy:.2%}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run all demonstrations"""
    print("Support Vector Machine (SVM) Implementation from Scratch")
    print("=" * 60)
    print("This implementation demonstrates:")
    print("1. Linear SVM for linearly separable data")
    print("2. Kernel SVM for non-linearly separable data")
    print("3. Effect of hyperparameters")
    print("4. Support vector identification")
    print("5. Decision boundary visualization")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_linear_svm()
    demonstrate_nonlinear_svm()
    demonstrate_hyperparameter_effects()
    
    print("\n" + "=" * 60)
    print("SVM THEORY SUMMARY:")
    print("=" * 60)
    print("1. SVM finds the optimal hyperplane that maximizes margin")
    print("2. Support vectors are the points closest to the decision boundary")
    print("3. Kernel trick allows non-linear classification")
    print("4. C parameter controls regularization (bias-variance tradeoff)")
    print("5. Higher C: Less regularization, more complex decision boundary")
    print("6. Lower C: More regularization, simpler decision boundary")
    print("=" * 60)


if __name__ == "__main__":
    main()