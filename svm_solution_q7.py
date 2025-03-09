# 7. Repeat parts 5 and 6 on dataset 3

# Part 5: Standard Support Vector Classifier for Dataset 3
print("Part 5: Standard Support Vector Classifier for Dataset 3")
print("=" * 60)

# Apply the standard SVM to dataset 3
a_3, b_3 = standard_svm(X_train_3, Y_train_3, gamma=0.1)

print("Normal vector a:", a_3)
print("Offset b:", b_3)

# Plot the classifier for dataset 3 (training data)
plot_classifier(X_train_3, Y_train_3, a_3, b_3, title="Standard SVM Classifier for Dataset 3 (γ=0.1) - Training Data")

# Plot the classifier for dataset 3 (test data)
plot_classifier(X_test_3, Y_test_3, a_3, b_3, title="Standard SVM Classifier for Dataset 3 (γ=0.1) - Test Data")

# Evaluate the classifier on the test set using the metrics defined in question 2
metrics_3 = evaluate_classifier(X_test_3, Y_test_3, a_3, b_3)

# Print the metrics
print("\nPerformance Metrics for Standard SVM Classifier (γ=0.1) on Dataset 3 Test Set:")
print(f"Accuracy: {metrics_3['accuracy']:.4f}")
print(f"Precision: {metrics_3['precision']:.4f}")
print(f"Recall: {metrics_3['recall']:.4f}")
print(f"Specificity: {metrics_3['specificity']:.4f}")
print(f"F1 Score: {metrics_3['f1_score']:.4f}")
print(f"Balanced Accuracy: {metrics_3['balanced_accuracy']:.4f}")
print(f"Misclassification Rate: {metrics_3['misclassification_rate']:.4f}")
print(f"Geometric Mean: {metrics_3['g_mean']:.4f}")
print(f"Confusion Matrix:")
print(f"TP: {metrics_3['TP']}, FP: {metrics_3['FP']}")
print(f"FN: {metrics_3['FN']}, TN: {metrics_3['TN']}")

# Part 6: Nonlinear Classifiers for Dataset 3
print("\nPart 6: Nonlinear Classifiers for Dataset 3")
print("=" * 60)

# Test different kernel functions on dataset 3
kernel_results_3 = {}

# Test each kernel
for kernel_name, kernel_func in kernels.items():
    print(f"\nTraining kernel SVM with {kernel_name} kernel...")
    
    # Train the kernel SVM
    alpha_x, alpha_y, b = kernel_svm(X_train_3, Y_train_3, kernel_func)
    
    # Evaluate on test data
    metrics = evaluate_kernel_svm(X_test_3, Y_test_3, X_train_3, Y_train_3, alpha_x, alpha_y, b, kernel_func)
    
    # Store the results
    kernel_results_3[kernel_name] = {
        'alpha_x': alpha_x,
        'alpha_y': alpha_y,
        'b': b,
        'metrics': metrics
    }
    
    # Print the metrics
    print(f"Performance Metrics for {kernel_name} Kernel SVM on Dataset 3 Test Set:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    
    # Plot the classifier
    plot_kernel_svm(X_train_3, Y_train_3, X_train_3, Y_train_3, alpha_x, alpha_y, b, kernel_func, 
                   title=f"{kernel_name} Kernel SVM for Dataset 3 - Training Data")
    plot_kernel_svm(X_test_3, Y_test_3, X_train_3, Y_train_3, alpha_x, alpha_y, b, kernel_func, 
                   title=f"{kernel_name} Kernel SVM for Dataset 3 - Test Data")

# Find the best kernel based on F1 score
best_kernel_3 = max(kernel_results_3.items(), key=lambda x: x[1]['metrics']['f1_score'])
best_kernel_name_3 = best_kernel_3[0]
best_metrics_3 = best_kernel_3[1]['metrics']

print(f"\nBest Kernel for Dataset 3: {best_kernel_name_3}")
print(f"Best F1 Score: {best_metrics_3['f1_score']:.4f}")
print(f"Corresponding Accuracy: {best_metrics_3['accuracy']:.4f}")
print(f"Corresponding Balanced Accuracy: {best_metrics_3['balanced_accuracy']:.4f}")
print(f"Corresponding G-Mean: {best_metrics_3['g_mean']:.4f}")

# Compare with linear SVM
print("\nComparison with Linear SVM for Dataset 3:")
print(f"Linear SVM - Accuracy: {metrics_3['accuracy']:.4f}")
print(f"Best Kernel SVM - Accuracy: {best_metrics_3['accuracy']:.4f}")
print(f"Linear SVM - F1 Score: {metrics_3['f1_score']:.4f}")
print(f"Best Kernel SVM - F1 Score: {best_metrics_3['f1_score']:.4f}")
print(f"Linear SVM - Balanced Accuracy: {metrics_3['balanced_accuracy']:.4f}")
print(f"Best Kernel SVM - Balanced Accuracy: {best_metrics_3['balanced_accuracy']:.4f}")
print(f"Linear SVM - G-Mean: {metrics_3['g_mean']:.4f}")
print(f"Best Kernel SVM - G-Mean: {best_metrics_3['g_mean']:.4f}")

# Discussion on the results for dataset 3
"""
In this question, I've repeated the analysis from questions 5 and 6 on dataset 3. This allows us to compare how different classifiers perform on different datasets and to see if the conclusions from dataset 2 generalize to dataset 3.

Part 5: Standard Support Vector Classifier
------------------------------------------
I applied a standard support vector classifier with γ=0.1 to dataset 3. The linear decision boundary achieved an accuracy of {metrics_3['accuracy']:.4f} and an F1 score of {metrics_3['f1_score']:.4f} on the test set.

The visualization of the decision boundary shows how well a linear classifier can separate the classes in dataset 3. Compared to dataset 2, the linear classifier for dataset 3 achieved {'better' if metrics_3['accuracy'] > metrics_2['accuracy'] else 'worse'} accuracy, suggesting that the classes in dataset 3 are {'more' if metrics_3['accuracy'] > metrics_2['accuracy'] else 'less'} linearly separable.

Part 6: Nonlinear Classifiers
-----------------------------
I tested the same set of kernel functions on dataset 3 as I did on dataset 2:
- Linear kernel
- Polynomial kernels (degrees 2 and 3)
- RBF kernels (σ values of 0.5, 1.0, and 2.0)
- Sigmoid kernels (α values of 0.5 and 1.0)

The best kernel for dataset 3 was the {best_kernel_name_3}, which achieved an accuracy of {best_metrics_3['accuracy']:.4f} and an F1 score of {best_metrics_3['f1_score']:.4f} on the test set.

Compared to the linear classifier, the best nonlinear classifier improved accuracy by {(best_metrics_3['accuracy'] - metrics_3['accuracy']) * 100:.2f}% and F1 score by {(best_metrics_3['f1_score'] - metrics_3['f1_score']) * 100:.2f}%.

Comparison with Dataset 2
-------------------------
Comparing the results for datasets 2 and 3:

1. Linear Classifiers:
   - Dataset 2: Accuracy = {metrics_2['accuracy']:.4f}, F1 Score = {metrics_2['f1_score']:.4f}
   - Dataset 3: Accuracy = {metrics_3['accuracy']:.4f}, F1 Score = {metrics_3['f1_score']:.4f}
   - The linear classifier performed {'better' if metrics_3['accuracy'] > metrics_2['accuracy'] else 'worse'} on dataset 3 than on dataset 2.

2. Best Nonlinear Classifiers:
   - Dataset 2: {best_kernel_name} with Accuracy = {best_metrics['accuracy']:.4f}, F1 Score = {best_metrics['f1_score']:.4f}
   - Dataset 3: {best_kernel_name_3} with Accuracy = {best_metrics_3['accuracy']:.4f}, F1 Score = {best_metrics_3['f1_score']:.4f}
   - The best nonlinear classifier for dataset 3 was {'the same as' if best_kernel_name == best_kernel_name_3 else 'different from'} the best for dataset 2.
   - The improvement from linear to nonlinear was {'greater' if (best_metrics_3['accuracy'] - metrics_3['accuracy']) > (best_metrics['accuracy'] - metrics_2['accuracy']) else 'smaller'} for dataset 3 than for dataset 2.

3. Decision Boundaries:
   - The visualization of the decision boundaries shows that dataset 3 requires {'a more complex' if (best_metrics_3['accuracy'] - metrics_3['accuracy']) > (best_metrics['accuracy'] - metrics_2['accuracy']) else 'a simpler'} decision boundary than dataset 2.
   - The {best_kernel_name_3} kernel was able to capture the underlying structure of dataset 3 better than the linear classifier.

Conclusions
-----------
The analysis of dataset 3 reinforces the importance of choosing the right classifier for each specific dataset. The fact that {'the same' if best_kernel_name == best_kernel_name_3 else 'a different'} kernel performed best on dataset 3 compared to dataset 2 highlights the need to test multiple classifiers and select the one that best fits the data.

The improvement from linear to nonlinear classifiers was {'significant' if (best_metrics_3['accuracy'] - metrics_3['accuracy']) > 0.1 else 'modest'} for dataset 3, indicating that {'nonlinear methods are essential' if (best_metrics_3['accuracy'] - metrics_3['accuracy']) > 0.1 else 'even simple linear methods can perform reasonably well'} for this dataset.

Overall, the results demonstrate the flexibility of kernel methods in adapting to different data distributions and the importance of a systematic approach to classifier selection and evaluation.
""" 