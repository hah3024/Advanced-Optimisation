# 4. Linear Classifier via Logistic Modeling

# Implement logistic regression classifier
def logistic_regression(X, Y, max_iter=1000, learning_rate=0.01):
    """
    Implement a logistic regression classifier using gradient descent
    
    Parameters:
    X: Training data points of class X (n_samples, n_features)
    Y: Training data points of class Y (m_samples, n_features)
    max_iter: Maximum number of iterations for gradient descent
    learning_rate: Learning rate for gradient descent
    
    Returns:
    w: Weight vector
    b: Bias term
    """
    # Combine data and create labels
    X_combined = np.vstack((X, Y))
    y_labels = np.hstack((np.ones(len(X)), np.zeros(len(Y))))
    
    n_samples, n_features = X_combined.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0
    
    # Gradient descent
    for i in range(max_iter):
        # Linear model
        z = X_combined @ w + b
        
        # Sigmoid function
        y_pred = 1 / (1 + np.exp(-z))
        
        # Compute gradients
        dw = (1/n_samples) * X_combined.T @ (y_pred - y_labels)
        db = (1/n_samples) * np.sum(y_pred - y_labels)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b

# Apply logistic regression to dataset 1
w_1, b_1_log = logistic_regression(X_train_1, Y_train_1)

print("Weight vector w:", w_1)
print("Bias term b:", b_1_log)

# Function to evaluate logistic regression classifier
def evaluate_logistic(X_test, Y_test, w, b):
    """
    Evaluate the logistic regression classifier on test data
    
    Parameters:
    X_test: Test data points of class X
    Y_test: Test data points of class Y
    w: Weight vector
    b: Bias term
    
    Returns:
    metrics: Dictionary containing various performance metrics
    """
    # Combine test data
    X_scores = X_test @ w + b
    Y_scores = Y_test @ w + b
    
    # Apply sigmoid function
    X_probs = 1 / (1 + np.exp(-X_scores))
    Y_probs = 1 / (1 + np.exp(-Y_scores))
    
    # Predict classes (threshold at 0.5)
    X_predictions = X_probs > 0.5
    Y_predictions = Y_probs > 0.5
    
    # Count correct predictions
    X_correct = np.sum(X_predictions)
    Y_correct = np.sum(~Y_predictions)
    
    # Calculate metrics
    total_samples = len(X_test) + len(Y_test)
    
    # True positives: X points correctly classified as X
    TP = X_correct
    # False positives: Y points incorrectly classified as X
    FP = np.sum(Y_predictions)
    # True negatives: Y points correctly classified as Y
    TN = Y_correct
    # False negatives: X points incorrectly classified as Y
    FN = len(X_test) - X_correct
    
    # Accuracy: Overall proportion of correct predictions
    accuracy = (TP + TN) / total_samples
    
    # Precision: Proportion of true positives among all positive predictions
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall (Sensitivity): Proportion of true positives among all actual positives
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity: Proportion of true negatives among all actual negatives
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # F1 Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Balanced Accuracy: Average of recall and specificity
    balanced_accuracy = (recall + specificity) / 2
    
    # Misclassification Rate: Proportion of incorrect predictions
    misclassification_rate = 1 - accuracy
    
    # Geometric Mean: Square root of the product of recall and specificity
    g_mean = np.sqrt(recall * specificity)
    
    # Store all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy,
        'misclassification_rate': misclassification_rate,
        'g_mean': g_mean,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }
    
    return metrics

# Evaluate logistic regression on the test set
metrics_log = evaluate_logistic(X_test_1, Y_test_1, w_1, b_1_log)

# Print the metrics
print("\nPerformance Metrics for Logistic Regression Classifier on Test Set:")
print(f"Accuracy: {metrics_log['accuracy']:.4f}")
print(f"Precision: {metrics_log['precision']:.4f}")
print(f"Recall: {metrics_log['recall']:.4f}")
print(f"Specificity: {metrics_log['specificity']:.4f}")
print(f"F1 Score: {metrics_log['f1_score']:.4f}")
print(f"Balanced Accuracy: {metrics_log['balanced_accuracy']:.4f}")
print(f"Misclassification Rate: {metrics_log['misclassification_rate']:.4f}")
print(f"Geometric Mean: {metrics_log['g_mean']:.4f}")
print(f"Confusion Matrix:")
print(f"TP: {metrics_log['TP']}, FP: {metrics_log['FP']}")
print(f"FN: {metrics_log['FN']}, TN: {metrics_log['TN']}")

# Plot the logistic regression classifier
def plot_logistic_classifier(X, Y, w, b, title="Logistic Regression Classifier"):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Class X')
    plt.scatter(Y[:, 0], Y[:, 1], c='red', label='Class Y')
    
    # Plot the decision boundary
    x_min, x_max = min(np.min(X[:, 0]), np.min(Y[:, 0])) - 0.5, max(np.max(X[:, 0]), np.max(Y[:, 0])) + 0.5
    y_min, y_max = min(np.min(X[:, 1]), np.min(Y[:, 1])) - 0.5, max(np.max(X[:, 1]), np.max(Y[:, 1])) + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute the decision function values
    Z = grid @ w + b
    Z = 1 / (1 + np.exp(-Z))  # Apply sigmoid
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary (probability = 0.5, which is Z = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors=['k'], linestyles=['-'])
    
    # Fill the regions
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the logistic regression classifier
plot_logistic_classifier(X_train_1, Y_train_1, w_1, b_1_log, title="Logistic Regression Classifier for Dataset 1 - Training Data")
plot_logistic_classifier(X_test_1, Y_test_1, w_1, b_1_log, title="Logistic Regression Classifier for Dataset 1 - Test Data")

# Compare SVM and logistic regression
# Get the best SVM classifier from question 3
best_gamma = gamma_values[best_gamma_f1_idx]
best_svm_metrics = results[best_gamma_f1_idx][3]

print("\nComparison of SVM and Logistic Regression:")
print(f"SVM (γ={best_gamma}) - Accuracy: {best_svm_metrics['accuracy']:.4f}")
print(f"Logistic Regression - Accuracy: {metrics_log['accuracy']:.4f}")
print(f"SVM (γ={best_gamma}) - F1 Score: {best_svm_metrics['f1_score']:.4f}")
print(f"Logistic Regression - F1 Score: {metrics_log['f1_score']:.4f}")
print(f"SVM (γ={best_gamma}) - Balanced Accuracy: {best_svm_metrics['balanced_accuracy']:.4f}")
print(f"Logistic Regression - Balanced Accuracy: {metrics_log['balanced_accuracy']:.4f}")
print(f"SVM (γ={best_gamma}) - G-Mean: {best_svm_metrics['g_mean']:.4f}")
print(f"Logistic Regression - G-Mean: {metrics_log['g_mean']:.4f}")

# Discussion on the comparison
"""
Comparison of SVM and Logistic Regression Classifiers:

1. **Mathematical Formulation**:
   - SVM aims to find the hyperplane that maximizes the margin between classes
   - Logistic regression models the probability of a point belonging to a class using the sigmoid function

2. **Decision Boundary**:
   - Both classifiers find a linear decision boundary for this dataset
   - SVM's decision boundary is determined by the support vectors (points closest to the boundary)
   - Logistic regression's decision boundary is where the probability equals 0.5

3. **Objective Function**:
   - SVM minimizes hinge loss plus a regularization term
   - Logistic regression minimizes logistic loss (cross-entropy)

4. **Robustness**:
   - SVM is generally more robust to outliers due to its focus on the margin
   - Logistic regression can be more sensitive to outliers

5. **Probabilistic Interpretation**:
   - SVM does not provide probability estimates directly
   - Logistic regression provides probability estimates, which can be useful for decision-making

6. **Performance Comparison**:
   - Accuracy: SVM with γ={best_gamma} achieves {best_svm_metrics['accuracy']:.4f}, while logistic regression achieves {metrics_log['accuracy']:.4f}
   - F1 Score: SVM achieves {best_svm_metrics['f1_score']:.4f}, while logistic regression achieves {metrics_log['f1_score']:.4f}
   - Balanced Accuracy: SVM achieves {best_svm_metrics['balanced_accuracy']:.4f}, while logistic regression achieves {metrics_log['balanced_accuracy']:.4f}
   - G-Mean: SVM achieves {best_svm_metrics['g_mean']:.4f}, while logistic regression achieves {metrics_log['g_mean']:.4f}

7. **Visual Comparison**:
   - The decision boundaries of both classifiers are similar for this dataset
   - The margin concept is explicit in SVM but not in logistic regression
   - The probability contours in logistic regression provide additional information about the confidence of predictions

8. **Hyperparameter Tuning**:
   - SVM has the γ parameter that needs to be tuned
   - Logistic regression has fewer hyperparameters to tune (mainly regularization strength if added)

9. **Computational Efficiency**:
   - For large datasets, logistic regression can be more computationally efficient
   - SVM's complexity increases with the number of support vectors

Based on the performance metrics, the SVM classifier with γ={best_gamma} performs {'better' if best_svm_metrics['f1_score'] > metrics_log['f1_score'] else 'worse'} than the logistic regression classifier on this dataset in terms of F1 score.

The choice between SVM and logistic regression depends on the specific requirements of the application:
- If interpretability and probability estimates are important, logistic regression might be preferred
- If maximizing the margin and robustness to outliers are important, SVM might be preferred
- If computational efficiency is a concern, the size of the dataset and the number of features should be considered

For this specific dataset, {'SVM' if best_svm_metrics['f1_score'] > metrics_log['f1_score'] else 'logistic regression'} provides better overall performance based on the F1 score.
""" 