import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("Starting Hand-written Digits Recognition Analysis")

digits = datasets.load_digits()

print("\nDigits Dataset Loaded Successfully!")
print(f"Number of samples: {len(digits.images)}")
print(f"Image shape: {digits.images[0].shape} (8x8 pixels)\n")

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Training: {label}")
plt.suptitle("Sample Digits from the Dataset")
plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

print(f"\nOriginal image shape: {digits.images.shape}")
print(f"Flattened data shape: {data.shape} (n_samples, n_features)\n")

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

print(f"Training data shape (X_train): {X_train.shape}")
print(f"Test data shape (X_test): {X_test.shape}")
print(f"Training target shape (y_train): {y_train.shape}")
print(f"Test target shape (y_test): {y_test.shape}\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully using StandardScaler.\n")

print("Performing Digit Classification using Logistic Regression")

logistic_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

print("Training Logistic Regression classifier...")
logistic_classifier.fit(X_train_scaled, y_train)
print("Logistic Regression classifier trained successfully.\n")

predicted = logistic_classifier.predict(X_test_scaled)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction, actual in zip(axes, X_test, predicted, y_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    color = "green" if prediction == actual else "red"
    ax.set_title(f"Pred: {prediction}\nActual: {actual}", color=color)
plt.suptitle("Sample Test Predictions")
plt.show()

print("Evaluating Logistic Regression Model Performance")

print(
    f"Classification report for classifier {logistic_classifier}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix for Digits Classification")
plt.show()

print(f"Confusion matrix:\n{disp.confusion_matrix}\n")
