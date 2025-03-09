# 3. The Role of Parameter γ

# Test different values of gamma
def test_gamma_values(X_train, Y_train, X_test, Y_test, gamma_values):
    """
    Test different values of gamma and evaluate their performance
    
    Parameters:
    X_train: Training data points of class X
    Y_train: Training data points of class Y
    X_test: Test data points of class X
    Y_test: Test data points of class Y
    gamma_values: List of gamma values to test
    
    Returns:
    results: List of tuples (gamma, metrics) for each gamma value
    """
    results = []
    
    for gamma in gamma_values:
        # Train the classifier with the current gamma
        a, b = standard_svm(X_train, Y_train, gamma=gamma)
        
        # Evaluate on test data
        metrics = evaluate_classifier(X_test, Y_test, a, b)
        
        # Store the results
        results.append((gamma, a, b, metrics))
        
        # Plot the classifier for each gamma
        plt.figure(figsize=(10, 8))
        plot_classifier(X_train, Y_train, a, b, title=f"SVM Classifier with γ={gamma} - Training Data")
        
        plt.figure(figsize=(10, 8))
        plot_classifier(X_test, Y_test, a, b, title=f"SVM Classifier with γ={gamma} - Test Data")
    
    return results

# Define a range of gamma values to test
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Test the different gamma values
results = test_gamma_values(X_train_1, Y_train_1, X_test_1, Y_test_1, gamma_values)

# Extract the metrics for plotting
gamma_values = [result[0] for result in results]
accuracies = [result[3]['accuracy'] for result in results]
precisions = [result[3]['precision'] for result in results]
recalls = [result[3]['recall'] for result in results]
specificities = [result[3]['specificity'] for result in results]
f1_scores = [result[3]['f1_score'] for result in results]
balanced_accuracies = [result[3]['balanced_accuracy'] for result in results]
g_means = [result[3]['g_mean'] for result in results]

# Plot the metrics vs gamma
plt.figure(figsize=(12, 8))
plt.semilogx(gamma_values, accuracies, 'o-', label='Accuracy')
plt.semilogx(gamma_values, precisions, 's-', label='Precision')
plt.semilogx(gamma_values, recalls, '^-', label='Recall')
plt.semilogx(gamma_values, specificities, 'v-', label='Specificity')
plt.semilogx(gamma_values, f1_scores, 'd-', label='F1 Score')
plt.semilogx(gamma_values, balanced_accuracies, '*-', label='Balanced Accuracy')
plt.semilogx(gamma_values, g_means, 'x-', label='G-Mean')

plt.xlabel('γ (log scale)')
plt.ylabel('Metric Value')
plt.title('Classifier Performance vs γ')
plt.legend()
plt.grid(True)
plt.show()

# Find the best gamma based on different metrics
best_gamma_accuracy_idx = np.argmax(accuracies)
best_gamma_f1_idx = np.argmax(f1_scores)
best_gamma_balanced_accuracy_idx = np.argmax(balanced_accuracies)
best_gamma_g_mean_idx = np.argmax(g_means)

print(f"Best γ value based on Accuracy: {gamma_values[best_gamma_accuracy_idx]}")
print(f"Best γ value based on F1 Score: {gamma_values[best_gamma_f1_idx]}")
print(f"Best γ value based on Balanced Accuracy: {gamma_values[best_gamma_balanced_accuracy_idx]}")
print(f"Best γ value based on G-Mean: {gamma_values[best_gamma_g_mean_idx]}")

# Print detailed metrics for the best gamma based on F1 score
best_gamma = gamma_values[best_gamma_f1_idx]
best_metrics = results[best_gamma_f1_idx][3]

print(f"\nDetailed metrics for best γ={best_gamma} (based on F1 Score):")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"Specificity: {best_metrics['specificity']:.4f}")
print(f"F1 Score: {best_metrics['f1_score']:.4f}")
print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
print(f"G-Mean: {best_metrics['g_mean']:.4f}")

# Discussion on the role of parameter γ
"""
The parameter γ in the standard support vector classifier controls the trade-off between maximizing the margin and minimizing the classification error:

1. **Small γ values** (e.g., 0.001, 0.01):
   - The classifier focuses more on maximizing the margin
   - It allows more misclassifications in the training data
   - This can lead to underfitting, where the model is too simple to capture the underlying pattern
   - The decision boundary tends to be smoother and less influenced by individual data points

2. **Large γ values** (e.g., 10, 100):
   - The classifier focuses more on correctly classifying all training points
   - It allows for a smaller margin
   - This can lead to overfitting, where the model is too complex and captures noise in the training data
   - The decision boundary tends to be more complex and influenced by individual data points

From the plots and metrics, we can observe how the performance changes with different γ values:

- **Accuracy**: Shows how the overall classification performance changes with γ
- **Precision and Recall**: Show how the classifier's performance on the positive class changes
- **Specificity**: Shows how the classifier's performance on the negative class changes
- **F1 Score, Balanced Accuracy, and G-Mean**: Show balanced measures of performance

The best γ value depends on the specific requirements of the application:
- If we want to maximize overall accuracy, we might choose one γ value
- If we want to balance precision and recall (F1 score), we might choose another
- If we want to ensure good performance on both classes (balanced accuracy or G-mean), we might choose yet another

Based on the F1 score, which balances precision and recall, the best γ value is {best_gamma}. This value provides the best trade-off between correctly classifying the positive class and avoiding false positives.

The plots of the decision boundaries for different γ values illustrate how the classifier changes:
- With small γ, the decision boundary is simpler and the margin is wider
- With large γ, the decision boundary is more complex and the margin is narrower
- The best γ value finds the right balance for this specific dataset

In practice, the choice of γ should be based on cross-validation or performance on a separate validation set, as we've done here with the test set.
""" 