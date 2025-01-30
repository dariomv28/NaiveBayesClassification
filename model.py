import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.priors = {}
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        :param X: Feature matrix (numpy array)
        :param y: Target vector (numpy array)
        """
        self.classes = np.unique(y)  # Get unique class labels
        for cls in self.classes:
            # Extract data for each class
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)  # Mean of features for class
            self.variance[cls] = np.var(X_cls, axis=0)  # Variance of features for class
            self.priors[cls] = X_cls.shape[0] / X.shape[0]  # Prior probability
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian probability density function.
        :param x: Data point
        :param mean: Mean of the distribution
        :param var: Variance of the distribution
        """
        eps = 1e-6  # To prevent division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent
    
    def _class_posterior(self, x):
        """
        Compute posterior probabilities for all classes.
        :param x: Single data point
        """
        posteriors = {}
        for cls in self.classes:
            # Start with the prior probability
            prior = np.log(self.priors[cls])
            # Add the log of the likelihood for each feature
            likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[cls], self.variance[cls])))
            posteriors[cls] = prior + likelihood
        return posteriors
    
    def predict(self, X):
        """
        Predict the class labels for a dataset.
        :param X: Feature matrix (numpy array)
        """
        y_pred = []
        for x in X:
            # Get posteriors for each class
            posteriors = self._class_posterior(x)
            # Choose the class with the highest posterior
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)



class MultiNB:
    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        self.class_log_prior = None
        self.feature_log_prob = None
        self.classes_ = None
        self.vocab_size = 0

    def fit(self, X, y):
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_log_prior = np.log(class_counts / len(y))   # P(C) = N(C) / samples
        self.vocab_size = X.shape[1]
        feature_counts = np.zeros((len(self.classes_), self.vocab_size))
        for index, cls in enumerate(self.classes_):
            feature_counts[index, :] = X[y == cls].sum(axis = 0)
        smoothed_counts = feature_counts + self.alpha
        smoothed_total = feature_counts.sum(axis = 1, keepdims = True)
        self.feature_log_prob = np.log(smoothed_counts / smoothed_total)

    def predict(self, X):
        log_probs = X @ self.feature_log_prob.T + self.class_log_prior
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


        


# Example usage
if __name__ == "__main__":
    # Sample dataset: word frequency representation (term-document matrix)
    X_train = np.array([
        [3, 0, 1],  # Example 1
        [2, 1, 0],  # Example 2
        [0, 2, 4],  # Example 3
        [1, 0, 3]   # Example 4
    ])
    y_train = np.array([0, 0, 1, 1])  # Labels (two classes: 0 and 1)

    model = MultiNB(alpha=1.0)
    model.fit(X_train, y_train)

    X_test = np.array([
        [2, 0, 1],  # Test example 1
        [1, 1, 2]   # Test example 2
    ])
    y_test = np.array([0, 1])

    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    print("Accuracy:", model.score(X_test, y_test))
