"""
Interactive SVM Web Application using Streamlit
Educational tool to explore Support Vector Machines
Compares custom implementation with scikit-learn
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_blobs

# Import our custom SVM implementation
from main import SVM, generate_linearly_separable_data, generate_nonlinear_data

# Configure page
st.set_page_config(
    page_title="Interactive SVM Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_dataset(dataset_type, n_samples, noise_level):
    """Generate different types of datasets"""
    np.random.seed(42)
    
    if dataset_type == "Linear Separable":
        X, y = generate_linearly_separable_data(n_samples, noise_level)
    elif dataset_type == "Circular":
        X, y = generate_nonlinear_data(n_samples)
    elif dataset_type == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise_level, factor=0.3, random_state=42)
        y = np.where(y == 0, -1, 1)
    elif dataset_type == "Overlapping Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                         cluster_std=1.5, random_state=42)
        y = np.where(y == 0, -1, 1)
    elif dataset_type == "XOR Pattern":
        X1 = np.random.normal([1, 1], [0.5, 0.5], (n_samples//4, 2))
        X2 = np.random.normal([-1, -1], [0.5, 0.5], (n_samples//4, 2))
        X3 = np.random.normal([1, -1], [0.5, 0.5], (n_samples//4, 2))
        X4 = np.random.normal([-1, 1], [0.5, 0.5], (n_samples//4, 2))
        
        X = np.vstack([X1, X2, X3, X4])
        y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
    
    return X, y

def train_models(X, y, custom_params, sklearn_params):
    """Train both custom and sklearn models"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Train custom SVM
    try:
        start_time = time.time()
        custom_svm = SVM(**custom_params)
        custom_svm.fit(X_scaled, y)
        custom_time = time.time() - start_time
        
        custom_accuracy = custom_svm.score(X_scaled, y)
        results['custom'] = {
            'model': custom_svm,
            'accuracy': custom_accuracy,
            'time': custom_time,
            'support_vectors': len(custom_svm.support_vectors),
            'sv_ratio': len(custom_svm.support_vectors) / len(X_scaled),
            'scaler': scaler,
            'success': True
        }
    except Exception as e:
        results['custom'] = {
            'error': str(e),
            'success': False
        }
    
    # Train sklearn SVM
    try:
        start_time = time.time()
        sklearn_svm = SVC(**sklearn_params)
        sklearn_svm.fit(X_scaled, y)
        sklearn_time = time.time() - start_time
        
        sklearn_accuracy = sklearn_svm.score(X_scaled, y)
        results['sklearn'] = {
            'model': sklearn_svm,
            'accuracy': sklearn_accuracy,
            'time': sklearn_time,
            'support_vectors': len(sklearn_svm.support_vectors_),
            'sv_ratio': len(sklearn_svm.support_vectors_) / len(X_scaled),
            'scaler': scaler,
            'success': True
        }
    except Exception as e:
        results['sklearn'] = {
            'error': str(e),
            'success': False
        }
    
    return results, X_scaled

def create_matplotlib_plot(X, y, results):
    """Create decision boundary plot using matplotlib"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    colors = ['red' if label == -1 else 'blue' for label in y]
    
    # Plot custom SVM
    if results['custom']['success']:
        try:
            Z_custom = results['custom']['model'].decision_function(mesh_points)
            Z_custom = Z_custom.reshape(xx.shape)
            
            axes[0].contourf(xx, yy, Z_custom, levels=50, alpha=0.3, cmap='RdYlBu')
            axes[0].contour(xx, yy, Z_custom, levels=[0], colors='black', linewidths=2)
            axes[0].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
            
            sv = results['custom']['model'].support_vectors
            axes[0].scatter(sv[:, 0], sv[:, 1], s=200, linewidth=2, 
                          facecolors='none', edgecolors='black')
            
            axes[0].set_title(f"Custom SVM\nAcc: {results['custom']['accuracy']:.2%}, "
                            f"SV: {len(sv)}, Time: {results['custom']['time']:.3f}s")
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error: {str(e)[:30]}...", 
                        transform=axes[0].transAxes, ha='center')
    else:
        axes[0].text(0.5, 0.5, "Custom SVM Failed", 
                    transform=axes[0].transAxes, ha='center')
    
    # Plot sklearn SVM
    if results['sklearn']['success']:
        try:
            Z_sklearn = results['sklearn']['model'].decision_function(mesh_points)
            Z_sklearn = Z_sklearn.reshape(xx.shape)
            
            axes[1].contourf(xx, yy, Z_sklearn, levels=50, alpha=0.3, cmap='RdYlBu')
            axes[1].contour(xx, yy, Z_sklearn, levels=[0], colors='black', linewidths=2)
            axes[1].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
            
            sv = results['sklearn']['model'].support_vectors_
            axes[1].scatter(sv[:, 0], sv[:, 1], s=200, linewidth=2, 
                          facecolors='none', edgecolors='black')
            
            axes[1].set_title(f"Scikit-learn SVM\nAcc: {results['sklearn']['accuracy']:.2%}, "
                            f"SV: {len(sv)}, Time: {results['sklearn']['time']:.3f}s")
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Error: {str(e)[:30]}...", 
                        transform=axes[1].transAxes, ha='center')
    else:
        axes[1].text(0.5, 0.5, "Sklearn SVM Failed", 
                    transform=axes[1].transAxes, ha='center')
    
    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    
    st.title("🤖 Interactive SVM Explorer")
    st.markdown("**Compare custom implementation vs. scikit-learn • Explore SVM theory interactively**")
    
    # Sidebar controls
    st.sidebar.title("🔧 SVM Configuration")
    
    # Dataset selection
    st.sidebar.subheader("📊 Dataset Settings")
    dataset_type = st.sidebar.selectbox(
        "Choose Dataset Type",
        ["Linear Separable", "Circular", "Concentric Circles", "Overlapping Blobs", "XOR Pattern"]
    )
    
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 200, 25)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
    
    # Model parameters
    st.sidebar.subheader("⚙️ SVM Parameters")
    kernel = st.sidebar.selectbox("Kernel Type", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("C (Regularization)", 0.1, 100.0, 1.0, 0.1)
    
    if kernel in ["rbf", "poly"]:
        gamma = st.sidebar.slider("Gamma", 0.001, 10.0, 1.0, 0.001, format="%.3f")
    else:
        gamma = 1.0
    
    if kernel == "poly":
        degree = st.sidebar.slider("Polynomial Degree", 1, 5, 3)
    else:
        degree = 3
    
    # Advanced settings
    st.sidebar.subheader("🔬 Advanced Settings")
    max_iter = st.sidebar.slider("Max Iterations (Custom SVM)", 100, 2000, 1000, 100)
    tol = st.sidebar.select_slider("Tolerance", [1e-4, 1e-3, 1e-2], value=1e-3)
    
    # Generate dataset
    X, y = generate_dataset(dataset_type, n_samples, noise_level)
    
    # Display dataset info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Generated Dataset")
        fig_data = plt.figure(figsize=(8, 6))
        colors = ['red' if label == -1 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
        plt.title(f"{dataset_type} Dataset ({n_samples} samples)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_data)
        plt.close()
    
    with col2:
        st.subheader("📈 Dataset Info")
        st.metric("Total Samples", n_samples)
        st.metric("Features", 2)
        st.metric("Classes", 2)
        st.metric("Class Balance", f"{np.sum(y==1)}/{np.sum(y==-1)}")
        st.metric("Noise Level", f"{noise_level:.2f}")
    
    # Prepare parameters
    custom_params = {
        'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree,
        'tol': tol, 'max_iter': max_iter
    }
    
    sklearn_params = {
        'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree,
        'tol': tol, 'max_iter': max_iter
    }
    
    # Train models
    if st.button("🚀 Train SVM Models", type="primary"):
        with st.spinner("Training models..."):
            results, X_scaled = train_models(X, y, custom_params, sklearn_params)
        
        st.subheader("📈 Training Results")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Custom SVM")
            if results['custom']['success']:
                st.metric("Accuracy", f"{results['custom']['accuracy']:.3f}")
                st.metric("Time", f"{results['custom']['time']:.3f}s")
                st.metric("Support Vectors", results['custom']['support_vectors'])
                st.metric("SV Ratio", f"{results['custom']['sv_ratio']:.2%}")
            else:
                st.error(f"Failed: {results['custom']['error']}")
        
        with col2:
            st.markdown("### Scikit-learn SVM")
            if results['sklearn']['success']:
                st.metric("Accuracy", f"{results['sklearn']['accuracy']:.3f}")
                st.metric("Time", f"{results['sklearn']['time']:.3f}s")
                st.metric("Support Vectors", results['sklearn']['support_vectors'])
                st.metric("SV Ratio", f"{results['sklearn']['sv_ratio']:.2%}")
            else:
                st.error(f"Failed: {results['sklearn']['error']}")
        
        with col3:
            st.markdown("### Comparison")
            if results['custom']['success'] and results['sklearn']['success']:
                acc_diff = abs(results['custom']['accuracy'] - results['sklearn']['accuracy'])
                speed_ratio = results['custom']['time'] / results['sklearn']['time']
                sv_diff = abs(results['custom']['support_vectors'] - results['sklearn']['support_vectors'])
                
                st.metric("Accuracy Diff", f"{acc_diff:.3f}")
                st.metric("Speed Ratio", f"{speed_ratio:.1f}x")
                st.metric("SV Difference", sv_diff)
                
                if speed_ratio > 1:
                    st.success(f"Sklearn is {speed_ratio:.1f}x faster!")
        
        # Visualization
        st.subheader("🎯 Decision Boundary Visualization")
        
        if results['custom']['success'] or results['sklearn']['success']:
            fig = create_matplotlib_plot(X_scaled, y, results)
            st.pyplot(fig)
            plt.close()
        
        # Detailed comparison
        if results['custom']['success'] and results['sklearn']['success']:
            st.subheader("📊 Detailed Comparison")
            
            comparison_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Training Time (s)', 'Support Vectors', 'SV Ratio (%)', 'Status'],
                'Custom SVM': [
                    f"{results['custom']['accuracy']:.3f}",
                    f"{results['custom']['time']:.3f}",
                    results['custom']['support_vectors'],
                    f"{results['custom']['sv_ratio']*100:.1f}%",
                    "✅ Success"
                ],
                'Scikit-learn SVM': [
                    f"{results['sklearn']['accuracy']:.3f}",
                    f"{results['sklearn']['time']:.3f}",
                    results['sklearn']['support_vectors'],
                    f"{results['sklearn']['sv_ratio']*100:.1f}%",
                    "✅ Success"
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
    
    # Educational content
    st.subheader("🎓 SVM Theory & Tips")
    
    tab1, tab2, tab3 = st.tabs(["📖 Theory", "🔧 Parameters", "💡 Tips"])
    
    with tab1:
        st.markdown("""
        ### Support Vector Machine Theory
        
        **Objective:** Find the optimal hyperplane that maximizes the margin between classes.
        
        **Key Concepts:**
        - **Margin:** Distance between the hyperplane and the closest points
        - **Support Vectors:** Points that lie on the margin boundary  
        - **Kernel Trick:** Map data to higher dimensions for non-linear separation
        
        **Decision Function:** f(x) = sign(∑αᵢyᵢK(xᵢ,x) + b)
        """)
    
    with tab2:
        st.markdown(f"""
        ### Current Parameter Settings
        
        **Dataset:** {dataset_type} ({n_samples} samples)  
        **Kernel:** {kernel}  
        **C:** {C} (Regularization strength)  
        **Gamma:** {gamma} (Kernel coefficient)  
        **Degree:** {degree} (Polynomial degree)  
        
        **Parameter Effects:**
        - **Higher C:** Less regularization, more complex boundary
        - **Lower C:** More regularization, simpler boundary
        - **Higher Gamma:** Tighter fit to training data
        - **Lower Gamma:** Smoother decision boundary
        """)
    
    with tab3:
        st.markdown("""
        ### Practical Tips
        
        **Data Preprocessing:**
        - Always standardize/normalize features ✅ (Done automatically)
        - Handle missing values appropriately
        - Consider feature selection for high dimensions
        
        **Parameter Selection:**
        - Start with default parameters
        - Use grid search for optimization
        - Cross-validate to avoid overfitting
        
        **Kernel Choice:**
        - **Linear:** Good baseline, fast training
        - **RBF:** Most versatile, handles non-linear patterns  
        - **Polynomial:** When polynomial relationships expected
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🤖 Interactive SVM Explorer | Built with Streamlit<br>
    Educational tool comparing custom SVM implementation vs. scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 