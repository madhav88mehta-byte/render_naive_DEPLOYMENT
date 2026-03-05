from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__)

# ======================================
# MANUAL TRAIN TEST SPLIT
# ======================================

def manual_train_test_split(X, y, test_size=0.2):

    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ======================================
# GAUSSIAN NAIVE BAYES
# ======================================

def gaussian_pdf(x, mean, var):

    eps = 1e-6
    coeff = 1.0 / np.sqrt(2 * np.pi * var + eps)
    exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))

    return coeff * exponent


def run_classification(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size)

    classes = np.unique(y_train)

    mean = {}
    var = {}
    prior = {}

    for c in classes:
        X_c = X_train[y_train == c]

        mean[c] = X_c.mean()
        var[c] = X_c.var()

        prior[c] = len(X_c) / len(X_train)

    predictions = []

    for i in range(len(X_test)):

        posteriors = []

        for c in classes:

            prior_prob = np.log(prior[c])

            likelihood = np.sum(
                np.log(gaussian_pdf(X_test.iloc[i], mean[c], var[c]))
            )

            posterior = prior_prob + likelihood

            posteriors.append(posterior)

        predictions.append(classes[np.argmax(posteriors)])

    predictions = np.array(predictions)

    accuracy = np.mean(predictions == y_test.values)

    unique_classes = np.unique(y)

    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

    for true, pred in zip(y_test, predictions):

        i = list(unique_classes).index(true)
        j = list(unique_classes).index(pred)

        cm[i][j] += 1

    return accuracy, cm.tolist()


# ======================================
# LINEAR REGRESSION
# ======================================

def run_regression(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    if y.dtype == "object":
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size)

    X_train = np.c_[np.ones(len(X_train)), X_train]
    X_test = np.c_[np.ones(len(X_test)), X_test]

    y_train = y_train.values
    y_test = y_test.values

    theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    y_pred = X_test @ theta

    mse = np.mean((y_test - y_pred) ** 2)

    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_pred) ** 2)

    r2 = 1 - (ss_residual / ss_total)

    return mse, r2


# ======================================
# ROUTES
# ======================================

@app.route("/")
def home():
    return send_from_directory(".", "app.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    dataset = data["data"]
    target = data["target"]
    features = data["features"]
    problem_type = data["type"]

    df = pd.DataFrame(dataset)

    test_size = 0.2

    if problem_type == "classification":

        acc, cm = run_classification(df, target, features, test_size)

        return jsonify({
            "model": "Gaussian Naive Bayes",
            "accuracy": acc,
            "confusion_matrix": cm
        })

    else:

        mse, r2 = run_regression(df, target, features, test_size)

        return jsonify({
            "model": "Linear Regression",
            "mse": mse,
            "r2_score": r2
        })


# ======================================
# MAIN
# ======================================

if __name__ == "__main__":
    app.run(debug=True)