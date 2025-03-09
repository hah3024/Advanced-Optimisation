# 5. Standard Support Vector Classifier for Dataset 2

# Apply the standard SVM to dataset 2
a_2, b_2 = standard_svm(X_train_2, Y_train_2, gamma=0.1)

print("Normal vector a:", a_2)
print("Offset b:", b_2)

# Plot the classifier for dataset 2 (training data)
plot_classifier(X_train_2, Y_train_2, a_2, b_2, title="Standard SVM Classifier for Dataset 2 (γ=0.1) - Training Data")

# Plot the classifier for dataset 2 (test data)
plot_classifier(X_test_2, Y_test_2, a_2, b_2, title="Standard SVM Classifier for Dataset 2 (γ=0.1) - Test Data")

# Evaluate the classifier on the test set using the metrics defined in question 2
metrics_2 = evaluate_classifier(X_test_2, Y_test_2, a_2, b_2)

# Print the metrics
print("\nPerformance Metrics for Standard SVM Classifier (γ=0.1) on Dataset 2 Test Set:")
print(f"Accuracy: {metrics_2['accuracy']:.4f}")
print(f"Precision: {metrics_2['precision']:.4f}")
print(f"Recall: {metrics_2['recall']:.4f}")
print(f"Specificity: {metrics_2['specificity']:.4f}")
print(f"F1 Score: {metrics_2['f1_score']:.4f}")
print(f"Balanced Accuracy: {metrics_2['balanced_accuracy']:.4f}")
print(f"Misclassification Rate: {metrics_2['misclassification_rate']:.4f}")
print(f"Geometric Mean: {metrics_2['g_mean']:.4f}")
print(f"Confusion Matrix:")
print(f"TP: {metrics_2['TP']}, FP: {metrics_2['FP']}")
print(f"FN: {metrics_2['FN']}, TN: {metrics_2['TN']}")

# Discussion of the results
"""
For dataset 2, I've applied a standard support vector classifier with γ=0.1, just as I did for dataset 1. The classifier finds a linear decision boundary that separates the two classes.

The plot shows:
- Blue points: Class X (positive class)
- Red points: Class Y (negative class)
- Black line: Decision boundary (hyperplane)
- Dashed lines: Margins
- Colored regions: Classification regions (blue for Class X, red for Class Y)

The performance metrics on the test set indicate how well the classifier generalizes to unseen data. The accuracy, precision, recall, F1 score, and other metrics provide a comprehensive evaluation of the classifier's performance.

Looking at the plots and metrics, we can observe:
1. How well the linear boundary separates the classes in the training data
2. How well this boundary generalizes to the test data
3. The specific strengths and weaknesses of the classifier in terms of different metrics

The slab (region between the dashed lines) represents the margin of the classifier. Points within this margin or on the wrong side of the boundary are either support vectors or misclassified points.

Based on the metrics, we can assess whether a linear classifier is appropriate for this dataset or if we need to explore nonlinear classifiers, which is the focus of the next question.
""" 