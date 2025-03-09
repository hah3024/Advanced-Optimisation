# 8. Multi-class Classification for Dataset 4

# Dataset 4 consists of three sets of points corresponding to three classes
# We need to find classifiers for this multi-class case

# First, let's visualize the data
plt.figure(figsize=(10, 8))
plt.scatter(X_4[:, 0], X_4[:, 1], c='blue', label='Class X')
plt.scatter(Y_4[:, 0], Y_4[:, 1], c='red', label='Class Y')
plt.scatter(Z_4[:, 0], Z_4[:, 1], c='green', label='Class Z')
plt.title('Dataset 4 - Three Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# There are several approaches to multi-class classification:
# 1. One-vs-Rest (OVR): Train K binary classifiers, each separating one class from all others
# 2. One-vs-One (OVO): Train K(K-1)/2 binary classifiers, each separating one class from another
# 3. Direct multi-class formulation: Extend the binary SVM to handle multiple classes directly

# Let's implement the One-vs-Rest approach, which is the most common
def one_vs_rest_svm(X_list, gamma=0.1, kernel_func=None):
    """
    Implement a One-vs-Rest SVM classifier for multi-class classification
    
    Parameters:
    X_list: List of data points for each class [X_1, X_2, ..., X_K]
    gamma: Parameter controlling the trade-off between margin and misclassification
    kernel_func: Kernel function to use (if None, use linear SVM)
    
    Returns:
    classifiers: List of classifiers, each separating one class from the rest
    """
    K = len(X_list)  # Number of classes
    classifiers = []
    
    for i in range(K):
        # Separate class i (positive) from all other classes (negative)
        X_i = X_list[i]
        Y_i = np.vstack([X_list[j] for j in range(K) if j != i])
        
        if kernel_func is None:
            # Use linear SVM
            a, b = standard_svm(X_i, Y_i, gamma=gamma)
            classifiers.append((a, b))
        else:
            # Use kernel SVM
            alpha_x, alpha_y, b = kernel_svm(X_i, Y_i, kernel_func, gamma=gamma)
            classifiers.append((alpha_x, alpha_y, b, X_i, Y_i, kernel_func))
    
    return classifiers

# Function to make predictions with One-vs-Rest SVM
def predict_one_vs_rest(x, classifiers, kernel_func=None):
    """
    Make a prediction for a single point using One-vs-Rest SVM
    
    Parameters:
    x: The point to classify
    classifiers: List of classifiers from one_vs_rest_svm
    kernel_func: Kernel function used (if None, classifiers are linear)
    
    Returns:
    class_idx: Predicted class index
    """
    K = len(classifiers)
    scores = np.zeros(K)
    
    for i in range(K):
        if kernel_func is None:
            # Linear SVM
            a, b = classifiers[i]
            scores[i] = np.dot(x, a) - b
        else:
            # Kernel SVM
            alpha_x, alpha_y, b, X_i, Y_i, kernel_func = classifiers[i]
            
            # Compute the decision function
            f_x = 0
            # Contribution from X points
            for j in range(len(X_i)):
                f_x += alpha_x[j] * kernel_func(x, X_i[j])
            # Contribution from Y points
            for j in range(len(Y_i)):
                f_x -= alpha_y[j] * kernel_func(x, Y_i[j])
            # Add the bias term
            f_x += b
            scores[i] = f_x
    
    # Return the class with the highest score
    return np.argmax(scores)

# Function to evaluate One-vs-Rest SVM
def evaluate_one_vs_rest(X_list, classifiers, kernel_func=None):
    """
    Evaluate One-vs-Rest SVM on data
    
    Parameters:
    X_list: List of data points for each class [X_1, X_2, ..., X_K]
    classifiers: List of classifiers from one_vs_rest_svm
    kernel_func: Kernel function used (if None, classifiers are linear)
    
    Returns:
    accuracy: Overall accuracy
    confusion_matrix: Confusion matrix
    """
    K = len(X_list)
    total_samples = sum(len(X_i) for X_i in X_list)
    correct = 0
    confusion_matrix = np.zeros((K, K), dtype=int)
    
    for true_class in range(K):
        for x in X_list[true_class]:
            pred_class = predict_one_vs_rest(x, classifiers, kernel_func)
            confusion_matrix[true_class, pred_class] += 1
            if pred_class == true_class:
                correct += 1
    
    accuracy = correct / total_samples
    return accuracy, confusion_matrix

# Function to plot the decision boundaries for One-vs-Rest SVM
def plot_one_vs_rest(X_list, classifiers, kernel_func=None, title="One-vs-Rest SVM Classifier"):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    colors = ['blue', 'red', 'green']
    labels = ['Class X', 'Class Y', 'Class Z']
    
    for i, X_i in enumerate(X_list):
        plt.scatter(X_i[:, 0], X_i[:, 1], c=colors[i], label=labels[i])
    
    # Plot the decision boundaries
    x_min, x_max = min(np.min(X_i[:, 0]) for X_i in X_list) - 0.5, max(np.max(X_i[:, 0]) for X_i in X_list) + 0.5
    y_min, y_max = min(np.min(X_i[:, 1]) for X_i in X_list) - 0.5, max(np.max(X_i[:, 1]) for X_i in X_list) + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the class for each point in the grid
    Z = np.zeros(grid.shape[0])
    for i, x in enumerate(grid):
        Z[i] = predict_one_vs_rest(x, classifiers, kernel_func)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Train and evaluate linear One-vs-Rest SVM
print("Training linear One-vs-Rest SVM for Dataset 4...")
linear_classifiers = one_vs_rest_svm([X_4, Y_4, Z_4], gamma=0.1)
linear_accuracy, linear_confusion = evaluate_one_vs_rest([X_4, Y_4, Z_4], linear_classifiers)

print(f"Linear One-vs-Rest SVM - Accuracy: {linear_accuracy:.4f}")
print("Confusion Matrix:")
print(linear_confusion)

# Plot the linear One-vs-Rest SVM
plot_one_vs_rest([X_4, Y_4, Z_4], linear_classifiers, title="Linear One-vs-Rest SVM for Dataset 4")

# Train and evaluate kernel One-vs-Rest SVM with different kernels
kernel_functions = {
    'Polynomial (degree=2)': lambda x, y: polynomial_kernel(x, y, degree=2),
    'RBF (sigma=1.0)': lambda x, y: rbf_kernel(x, y, sigma=1.0)
}

kernel_results_4 = {}

for kernel_name, kernel_func in kernel_functions.items():
    print(f"\nTraining {kernel_name} One-vs-Rest SVM for Dataset 4...")
    kernel_classifiers = one_vs_rest_svm([X_4, Y_4, Z_4], gamma=0.1, kernel_func=kernel_func)
    kernel_accuracy, kernel_confusion = evaluate_one_vs_rest([X_4, Y_4, Z_4], kernel_classifiers, kernel_func=kernel_func)
    
    kernel_results_4[kernel_name] = {
        'classifiers': kernel_classifiers,
        'accuracy': kernel_accuracy,
        'confusion_matrix': kernel_confusion
    }
    
    print(f"{kernel_name} One-vs-Rest SVM - Accuracy: {kernel_accuracy:.4f}")
    print("Confusion Matrix:")
    print(kernel_confusion)
    
    # Plot the kernel One-vs-Rest SVM
    plot_one_vs_rest([X_4, Y_4, Z_4], kernel_classifiers, kernel_func=kernel_func, 
                    title=f"{kernel_name} One-vs-Rest SVM for Dataset 4")

# Find the best kernel based on accuracy
best_kernel_4 = max(kernel_results_4.items(), key=lambda x: x[1]['accuracy'])
best_kernel_name_4 = best_kernel_4[0]
best_accuracy_4 = best_kernel_4[1]['accuracy']

print(f"\nBest Kernel for Dataset 4: {best_kernel_name_4}")
print(f"Best Accuracy: {best_accuracy_4:.4f}")

# Compare with linear SVM
print("\nComparison with Linear One-vs-Rest SVM:")
print(f"Linear One-vs-Rest SVM - Accuracy: {linear_accuracy:.4f}")
print(f"Best Kernel One-vs-Rest SVM - Accuracy: {best_accuracy_4:.4f}")

# Discussion on multi-class classification
"""
In this question, I've addressed the challenge of multi-class classification for dataset 4, which consists of three classes (X, Y, and Z) instead of two.

Approach to Multi-class Classification
-------------------------------------
There are several approaches to extending binary classifiers like SVM to handle multiple classes:

1. **One-vs-Rest (OVR)**: Train K binary classifiers, each separating one class from all others. For each new point, we compute the decision function for all K classifiers and assign the point to the class with the highest score.

2. **One-vs-One (OVO)**: Train K(K-1)/2 binary classifiers, each separating one pair of classes. For each new point, we use a voting scheme where each classifier votes for one class, and the class with the most votes wins.

3. **Direct Multi-class Formulation**: Extend the binary SVM optimization problem to handle multiple classes directly. This approach is more complex but can be more efficient for large numbers of classes.

I chose to implement the One-vs-Rest approach because:
- It's conceptually simple and easy to implement
- It requires training only K classifiers (compared to K(K-1)/2 for One-vs-One)
- It works well in practice for many problems
- It can be easily extended to use different kernels

Implementation
-------------
For dataset 4 with three classes (X, Y, and Z), I trained three binary classifiers:
1. Classifier 1: X vs. (Y and Z)
2. Classifier 2: Y vs. (X and Z)
3. Classifier 3: Z vs. (X and Y)

For a new point, I compute the decision function for all three classifiers and assign the point to the class with the highest score.

I implemented both linear and kernel-based One-vs-Rest SVMs:
- Linear One-vs-Rest SVM achieved an accuracy of {linear_accuracy:.4f}
- {best_kernel_name_4} One-vs-Rest SVM achieved an accuracy of {best_accuracy_4:.4f}

The visualization of the decision boundaries shows how the classifiers partition the feature space into regions corresponding to each class. The kernel-based approach can create more complex boundaries that better separate the classes.

Analysis of Results
------------------
The confusion matrices provide insight into which classes are more difficult to separate:
- Linear One-vs-Rest SVM confusion matrix:
{linear_confusion}

- Best Kernel One-vs-Rest SVM confusion matrix:
{best_kernel_4[1]['confusion_matrix']}

From these matrices, we can see that {'all classes are well-separated' if np.sum(np.diag(best_kernel_4[1]['confusion_matrix'])) > 0.9 * np.sum(best_kernel_4[1]['confusion_matrix']) else 'some classes are more difficult to separate than others'}.

The improvement from linear to kernel-based approaches ({(best_accuracy_4 - linear_accuracy) * 100:.2f}% increase in accuracy) indicates that {'the classes in dataset 4 are not linearly separable' if best_accuracy_4 - linear_accuracy > 0.1 else 'even a linear approach works reasonably well for dataset 4'}.

Comparison with Binary Classification
------------------------------------
Multi-class classification is inherently more challenging than binary classification because:
- The decision boundaries are more complex
- The classifiers need to distinguish between multiple classes simultaneously
- Errors in one binary classifier can affect the overall performance

Despite these challenges, the One-vs-Rest approach with the {best_kernel_name_4} kernel achieved a high accuracy of {best_accuracy_4:.4f}, demonstrating the effectiveness of this approach for dataset 4.

Conclusion
---------
The One-vs-Rest approach with kernel methods provides an effective solution for multi-class classification problems. By training separate binary classifiers and combining their outputs, we can extend the powerful binary SVM framework to handle multiple classes.

For dataset 4, the {best_kernel_name_4} kernel provided the best performance, suggesting that this kernel captures the underlying structure of the data most effectively. The visualization of the decision boundaries helps us understand how the classifier partitions the feature space and makes predictions for new points.

This approach can be extended to datasets with even more classes, although the computational complexity increases linearly with the number of classes. For very large numbers of classes, more efficient approaches like hierarchical classification or direct multi-class formulations might be more appropriate.
""" 