# 2. Evaluating the Classifier on Test Data

# Define a more comprehensive evaluation function
def evaluate_classifier(X_test, Y_test, a, b):
    """
    Evaluate the classifier on test data with multiple metrics
    
    Parameters:
    X_test: Test data points of class X
    Y_test: Test data points of class Y
    a: Normal vector to the separating hyperplane
    b: Offset of the hyperplane
    
    Returns:
    metrics: Dictionary containing various performance metrics
    """
    # Combine test data
    X_predictions = X_test @ a - b
    Y_predictions = Y_test @ a - b
    
    # Count correct predictions
    X_correct = np.sum(X_predictions > 0)
    Y_correct = np.sum(Y_predictions < 0)
    
    # Calculate metrics
    total_samples = len(X_test) + len(Y_test)
    
    # True positives: X points correctly classified as X
    TP = X_correct
    # False positives: Y points incorrectly classified as X
    FP = len(Y_test) - Y_correct
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

# Evaluate the classifier on the test set
metrics_1 = evaluate_classifier(X_test_1, Y_test_1, a_1, b_1)

# Print the metrics
print("Performance Metrics for Standard SVM Classifier (γ=0.1) on Test Set:")
print(f"Accuracy: {metrics_1['accuracy']:.4f}")
print(f"Precision: {metrics_1['precision']:.4f}")
print(f"Recall: {metrics_1['recall']:.4f}")
print(f"Specificity: {metrics_1['specificity']:.4f}")
print(f"F1 Score: {metrics_1['f1_score']:.4f}")
print(f"Balanced Accuracy: {metrics_1['balanced_accuracy']:.4f}")
print(f"Misclassification Rate: {metrics_1['misclassification_rate']:.4f}")
print(f"Geometric Mean: {metrics_1['g_mean']:.4f}")
print(f"Confusion Matrix:")
print(f"TP: {metrics_1['TP']}, FP: {metrics_1['FP']}")
print(f"FN: {metrics_1['FN']}, TN: {metrics_1['TN']}")

# Explanation of the chosen metrics
"""
I've chosen several metrics to evaluate the classifier:

1. Accuracy: The proportion of correctly classified samples out of all samples. It gives an overall measure of the classifier's performance.
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Pros: Simple and intuitive
   - Cons: Can be misleading with imbalanced classes

2. Precision: The proportion of true positives out of all predicted positives. It measures how many of the points classified as X are actually X.
   - Formula: TP / (TP + FP)
   - Pros: Important when the cost of false positives is high
   - Cons: Doesn't consider false negatives

3. Recall (Sensitivity): The proportion of true positives out of all actual positives. It measures how many of the actual X points are correctly classified.
   - Formula: TP / (TP + FN)
   - Pros: Important when the cost of false negatives is high
   - Cons: Doesn't consider false positives

4. Specificity: The proportion of true negatives out of all actual negatives. It measures how many of the actual Y points are correctly classified.
   - Formula: TN / (TN + FP)
   - Pros: Complements recall by focusing on the negative class
   - Cons: Doesn't consider false negatives

5. F1 Score: The harmonic mean of precision and recall. It provides a balance between precision and recall.
   - Formula: 2 * (precision * recall) / (precision + recall)
   - Pros: Combines precision and recall into a single metric
   - Cons: Doesn't consider true negatives

6. Balanced Accuracy: The average of recall and specificity. It's useful when classes are imbalanced.
   - Formula: (recall + specificity) / 2
   - Pros: Accounts for both classes equally, regardless of their sizes
   - Cons: May not reflect overall performance if one class is more important

7. Misclassification Rate: The proportion of incorrect predictions. It's the complement of accuracy.
   - Formula: 1 - accuracy
   - Pros: Directly measures the error rate
   - Cons: Same limitations as accuracy

8. Geometric Mean: The square root of the product of recall and specificity. It's another way to balance the performance on both classes.
   - Formula: sqrt(recall * specificity)
   - Pros: Penalizes poor performance on either class
   - Cons: Less intuitive than some other metrics

These metrics were chosen because they provide a comprehensive evaluation of the classifier's performance. Different metrics focus on different aspects of classification performance:
- Accuracy gives an overall view but can be misleading with imbalanced classes.
- Precision and recall focus on the positive class from different angles.
- Specificity focuses on the negative class.
- F1 score, balanced accuracy, and geometric mean provide balanced measures that consider both classes.

By examining all these metrics together, we get a more complete picture of how well the classifier is performing on both classes, which is especially important when the classes are imbalanced or when the costs of different types of errors vary.
""" 