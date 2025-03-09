# Function to plot the data points and the classifier
def plot_classifier(X, Y, a, b, title="Support Vector Classifier"):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Class X')
    plt.scatter(Y[:, 0], Y[:, 1], c='red', label='Class Y')
    
    # Plot the decision boundary (hyperplane)
    x_min, x_max = min(np.min(X[:, 0]), np.min(Y[:, 0])) - 0.5, max(np.max(X[:, 0]), np.max(Y[:, 0])) + 0.5
    y_min, y_max = min(np.min(X[:, 1]), np.min(Y[:, 1])) - 0.5, max(np.max(X[:, 1]), np.max(Y[:, 1])) + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute the decision function values
    Z = grid @ a - b
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['r', 'k', 'b'], linestyles=['--', '-', '--'])
    
    # Fill the regions
    plt.contourf(xx, yy, Z, levels=[-100, -1, 1, 100], colors=['#FFAAAA', '#FFFFFF', '#AAAAFF'], alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the classifier for dataset 1 (training data)
plot_classifier(X_train_1, Y_train_1, a_1, b_1, title="Standard SVM Classifier for Dataset 1 (γ=0.1) - Training Data")

# Plot the classifier for dataset 1 (test data)
plot_classifier(X_test_1, Y_test_1, a_1, b_1, title="Standard SVM Classifier for Dataset 1 (γ=0.1) - Test Data")

# Evaluate the classifier on training data
def evaluate_points(X, Y, a, b):
    """
    Evaluate how many points are correctly classified
    
    Parameters:
    X: Data points expected to be on the positive side
    Y: Data points expected to be on the negative side
    a: Normal vector to the separating hyperplane
    b: Offset of the hyperplane
    
    Returns:
    X_correct: Number of correctly classified X points
    Y_correct: Number of correctly classified Y points
    """
    X_predictions = X @ a - b
    Y_predictions = Y @ a - b
    
    X_correct = np.sum(X_predictions > 0)
    Y_correct = np.sum(Y_predictions < 0)
    
    return X_correct, Y_correct

# Evaluate on training data
X_correct_train, Y_correct_train = evaluate_points(X_train_1, Y_train_1, a_1, b_1)
print(f"Training data - Correctly classified X points: {X_correct_train}/{len(X_train_1)} ({X_correct_train/len(X_train_1)*100:.2f}%)")
print(f"Training data - Correctly classified Y points: {Y_correct_train}/{len(Y_train_1)} ({Y_correct_train/len(Y_train_1)*100:.2f}%)")
print(f"Training data - Overall accuracy: {(X_correct_train + Y_correct_train)/(len(X_train_1) + len(Y_train_1))*100:.2f}%")

# Evaluate on test data
X_correct_test, Y_correct_test = evaluate_points(X_test_1, Y_test_1, a_1, b_1)
print(f"Test data - Correctly classified X points: {X_correct_test}/{len(X_test_1)} ({X_correct_test/len(X_test_1)*100:.2f}%)")
print(f"Test data - Correctly classified Y points: {Y_correct_test}/{len(Y_test_1)} ({Y_correct_test/len(Y_test_1)*100:.2f}%)")
print(f"Test data - Overall accuracy: {(X_correct_test + Y_correct_test)/(len(X_test_1) + len(Y_test_1))*100:.2f}%") 