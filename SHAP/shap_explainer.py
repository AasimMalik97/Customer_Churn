import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('data/processed/dataset.csv')
print(data.columns)

# Convert categorical columns to numerical using one-hot encoding
categorical_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
data = pd.get_dummies(data, columns=categorical_columns)

X = data.drop('Churn', axis=1)  # 'Churn' is the target column
y = data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Step 1: Visualize SHAP explanations for a specific data point
shap.initjs()  # Load JS visualization library
index = 0  # Index of the data point to visualize

# Check if the model is binary or multi-class
if isinstance(explainer.expected_value, list):
    # Multi-class case
    shap.force_plot(explainer.expected_value[1], shap_values[1][index], X_test.iloc[index])
else:
    # Binary case
    shap.force_plot(explainer.expected_value, shap_values[index], X_test.iloc[index])

# Step 2: Visualize explanations for all points
shap.summary_plot(shap_values, X_test)

# Step 3: Visualize class-specific summary plots (for multi-class models)
if isinstance(shap_values, list):
    for i, class_shap_values in enumerate(shap_values):
        print(f"Summary plot for class {i}")
        shap.summary_plot(class_shap_values, X_test)
else:
    shap.summary_plot(shap_values, X_test)

# Advanced SHAP plots
# Waterfall plot for a specific data point
if isinstance(explainer.expected_value, list):
    shap.plots.waterfall(shap.Explanation(values=shap_values[1][index], base_values=explainer.expected_value[1], data=X_test.iloc[index]))
else:
    shap.plots.waterfall(shap.Explanation(values=shap_values[index], base_values=explainer.expected_value, data=X_test.iloc[index]))

# Force plot for the same data point
if isinstance(explainer.expected_value, list):
    shap.force_plot(explainer.expected_value[1], shap_values[1][index], X_test.iloc[index])
else:
    shap.force_plot(explainer.expected_value, shap_values[index], X_test.iloc[index])

# Mean SHAP plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Beeswarm plot
shap.plots.beeswarm(shap_values)

# Dependence plots
for feature in X_test.columns:
    shap.dependence_plot(feature, shap_values, X_test)