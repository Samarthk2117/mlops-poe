
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Load the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
print("Iris Flower Classification")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Prepare reference and current datasets for monitoring
reference_data = X_train.copy()
current_data = X_test.copy()

# Use feature names as columns in both datasets
report = Report([DataDriftPreset()])
result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# 6. Save the monitoring report
result.save_html("iris_drift_report.html")
print("Evidently monitoring report generated: iris_drift_report.html")
