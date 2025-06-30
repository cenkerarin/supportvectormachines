"""
SVM Implementation using Scikit-Learn
Comparison with our from-scratch implementation to demonstrate:
1. Convenience vs Understanding trade-offs
2. Performance differences
3. Advanced features available in scikit-learn
4. Industry-standard usage patterns
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_circles, make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom SVM for comparison
from main import SVM, generate_linearly_separable_data, generate_nonlinear_data

class SVMComparison:
    """
    Class to compare scikit-learn SVM with our custom implementation
    """
    
    def __init__(self):
        self.results = {}
    
    def generate_datasets(self):
        """Generate various datasets for comprehensive testing"""
        np.random.seed(42)
        
        datasets = {}
        
        # 1. Linearly separable data
        datasets['linear'] = generate_linearly_separable_data(n_samples=200, noise=0.1)
        
        # 2. Non-linear circular data
        datasets['circles'] = generate_nonlinear_data(n_samples=300)
        
        # 3. More complex non-linear data using sklearn
        datasets['complex_circles'] = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=42)
        
        # 4. Multi-class data (we'll make it binary)
        X_multi, y_multi = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                             n_informative=2, n_clusters_per_class=1, 
                                             random_state=42)
        datasets['classification'] = (X_multi, np.where(y_multi == 0, -1, 1))
        
        # 5. Overlapping blobs
        X_blobs, y_blobs = make_blobs(n_samples=300, centers=2, n_features=2, 
                                     cluster_std=1.5, random_state=42)
        datasets['blobs'] = (X_blobs, np.where(y_blobs == 0, -1, 1))
        
        return datasets
    
    def compare_basic_performance(self, datasets):
        """Compare basic performance between custom and sklearn SVM"""
        print("=" * 80)
        print("BASIC PERFORMANCE COMPARISON")
        print("=" * 80)
        
        results = []
        
        for name, (X, y) in datasets.items():
            print(f"\nDataset: {name.upper()}")
            print("-" * 40)
            
            # Standardize features (important for SVM)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Custom SVM
            start_time = time.time()
            custom_svm = SVM(C=1.0, kernel='rbf', gamma=1.0)
            custom_svm.fit(X_scaled, y)
            custom_time = time.time() - start_time
            custom_accuracy = custom_svm.score(X_scaled, y)
            custom_support_vectors = len(custom_svm.support_vectors)
            
            # Scikit-learn SVM
            start_time = time.time()
            sklearn_svm = SVC(C=1.0, kernel='rbf', gamma=1.0)
            sklearn_svm.fit(X_scaled, y)
            sklearn_time = time.time() - start_time
            sklearn_accuracy = sklearn_svm.score(X_scaled, y)
            sklearn_support_vectors = len(sklearn_svm.support_vectors_)
            
            # Store results
            result = {
                'dataset': name,
                'custom_accuracy': custom_accuracy,
                'sklearn_accuracy': sklearn_accuracy,
                'custom_time': custom_time,
                'sklearn_time': sklearn_time,
                'custom_sv': custom_support_vectors,
                'sklearn_sv': sklearn_support_vectors
            }
            results.append(result)
            
            # Print comparison
            print(f"Custom SVM    - Accuracy: {custom_accuracy:.3f}, Time: {custom_time:.3f}s, SV: {custom_support_vectors}")
            print(f"Sklearn SVM  - Accuracy: {sklearn_accuracy:.3f}, Time: {sklearn_time:.3f}s, SV: {sklearn_support_vectors}")
            print(f"Time Ratio   - Sklearn is {custom_time/sklearn_time:.1f}x faster")
            print(f"Accuracy Diff- {abs(custom_accuracy - sklearn_accuracy):.3f}")
        
        return results
    
    def demonstrate_sklearn_advantages(self, X, y):
        """Demonstrate advanced features available in scikit-learn"""
        print("\n" + "=" * 80)
        print("SCIKIT-LEARN ADVANCED FEATURES")
        print("=" * 80)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 1. Grid Search for Hyperparameter Tuning
        print("\n1. HYPERPARAMETER TUNING WITH GRID SEARCH")
        print("-" * 50)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        grid_search = GridSearchCV(
            SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        print(f"Grid search completed in {grid_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # 2. Cross-Validation
        print("\n2. CROSS-VALIDATION ANALYSIS")
        print("-" * 50)
        
        best_svm = grid_search.best_estimator_
        cv_scores = cross_val_score(best_svm, X_train, y_train, cv=5)
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # 3. Detailed Classification Report
        print("\n3. DETAILED CLASSIFICATION METRICS")
        print("-" * 50)
        
        y_pred = best_svm.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Class -1', 'Class +1']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # 4. Decision Function Analysis
        print("\n4. DECISION FUNCTION ANALYSIS")
        print("-" * 50)
        
        decision_scores = best_svm.decision_function(X_test)
        print(f"Decision function range: [{decision_scores.min():.3f}, {decision_scores.max():.3f}]")
        print(f"Points on margin (|score| < 1): {np.sum(np.abs(decision_scores) < 1)}")
        
        # 5. Support Vector Analysis
        print("\n5. SUPPORT VECTOR ANALYSIS")
        print("-" * 50)
        
        print(f"Number of support vectors: {len(best_svm.support_vectors_)}")
        print(f"Support vector ratio: {len(best_svm.support_vectors_)/len(X_train):.2%}")
        print(f"Support vectors per class: {best_svm.n_support_}")
        
        return best_svm, grid_search
    
    def kernel_comparison_sklearn(self, X, y):
        """Compare different kernels using scikit-learn"""
        print("\n" + "=" * 80)
        print("KERNEL COMPARISON WITH SCIKIT-LEARN")
        print("=" * 80)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        
        plt.figure(figsize=(20, 5))
        
        for i, kernel in enumerate(kernels):
            plt.subplot(1, 4, i+1)
            
            # Train SVM with current kernel
            if kernel == 'poly':
                svm = SVC(kernel=kernel, degree=3, gamma='scale', C=1.0)
            else:
                svm = SVC(kernel=kernel, gamma='scale', C=1.0)
            
            start_time = time.time()
            svm.fit(X_scaled, y)
            train_time = time.time() - start_time
            
            accuracy = svm.score(X_scaled, y)
            
            print(f"{kernel.upper()} Kernel:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Training time: {train_time:.3f}s")
            print(f"  Support vectors: {len(svm.support_vectors_)}")
            print(f"  Support vector ratio: {len(svm.support_vectors_)/len(X_scaled):.2%}")
            print()
            
            # Create decision boundary plot
            h = 0.02
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = svm.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
            plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            
            colors = ['red' if label == -1 else 'blue' for label in y]
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, alpha=0.6, s=30)
            plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
                       s=100, linewidth=1, facecolors='none', edgecolors='black')
            
            plt.title(f'{kernel.title()} Kernel\nAcc: {accuracy:.2%}, Time: {train_time:.3f}s')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()


def simple_sklearn_demo():
    """Simple demonstration of scikit-learn SVM usage"""
    print("=" * 60)
    print("SIMPLE SCIKIT-LEARN SVM DEMONSTRATION")
    print("=" * 60)
    
    # Generate some data
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("1. BASIC SVM USAGE")
    print("-" * 30)
    
    # Create and train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Support Vectors: {len(svm.support_vectors_)}")
    print(f"Support Vector Ratio: {len(svm.support_vectors_)/len(X_train_scaled):.2%}")
    
    print("\n2. QUICK HYPERPARAMETER TUNING")
    print("-" * 40)
    
    # Quick grid search
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, verbose=0)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    # Test best model
    best_accuracy = grid_search.best_estimator_.score(X_test_scaled, y_test)
    print(f"Best model test accuracy: {best_accuracy:.3f}")
    
    print("\n3. VISUALIZATION")
    print("-" * 20)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Basic SVM
    plt.subplot(1, 3, 2)
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    train_colors = ['red' if label == 0 else 'blue' for label in y_train]
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=train_colors, alpha=0.6)
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
               s=100, linewidth=1, facecolors='none', edgecolors='black')
    plt.title(f'Basic SVM\nAcc: {accuracy:.2%}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Optimized SVM
    plt.subplot(1, 3, 3)
    best_svm = grid_search.best_estimator_
    Z_best = best_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_best = Z_best.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_best, levels=50, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z_best, levels=[0], colors='black', linewidths=2)
    
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=train_colors, alpha=0.6)
    plt.scatter(best_svm.support_vectors_[:, 0], best_svm.support_vectors_[:, 1], 
               s=100, linewidth=1, facecolors='none', edgecolors='black')
    plt.title(f'Optimized SVM\nAcc: {best_accuracy:.2%}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run scikit-learn demonstrations"""
    print("SCIKIT-LEARN SVM DEMONSTRATION")
    print("=" * 60)
    print("Showing the power and convenience of scikit-learn")
    print("Comparing with our educational implementation")
    print("=" * 60)
    
    # Simple demo first
    simple_sklearn_demo()
    
    # Full comparison
    comparison = SVMComparison()
    
    # Generate datasets
    print("\nGenerating test datasets...")
    datasets = comparison.generate_datasets()
    
    # Performance comparison
    results = comparison.compare_basic_performance(datasets)
    
    # Advanced features
    X, y = datasets['circles']
    best_svm, grid_search = comparison.demonstrate_sklearn_advantages(X, y)
    
    # Kernel comparison
    comparison.kernel_comparison_sklearn(X, y)
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("✅ Scikit-learn SVM is:")
    print("   • 10-100x faster than our implementation")
    print("   • More accurate due to advanced optimizations")
    print("   • Feature-rich (grid search, cross-validation, etc.)")
    print("   • Production-ready and well-tested")
    
    print("\n✅ Our custom SVM is:")
    print("   • Educational and transparent")
    print("   • Helps understand the math and theory")
    print("   • Great for learning and research")
    print("   • Shows how algorithms work under the hood")
    
    print("\n🎯 BEST PRACTICE:")
    print("   • Learn with custom implementations")
    print("   • Deploy with scikit-learn")
    print("   • Understand both for maximum effectiveness!")


if __name__ == "__main__":
    main() 