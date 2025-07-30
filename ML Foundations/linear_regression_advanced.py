#Importing necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

#Loading the data
scaler = MinMaxScaler()
data = pd.read_csv("ML Foundations/linear_regression_dataset.csv")
X = data.drop("target", axis=1)
X += np.random.normal(0, 0.1, size=X.shape)
y = data["target"]
scaler = MinMaxScaler()

cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_best_models = {}

# Defining the models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

# Defining the hyperparameters
hyperparameters = {
    "Linear Regression": {},
    "Lasso": {"model__alpha": [0.01, 0.1, 1, 10]},
    "Ridge": {"model__alpha": [0.01, 0.1, 1, 10]},
    "ElasticNet": {"model__alpha": [0.01, 0.1, 1, 10], "model__l1_ratio": [0.1, 0.5, 0.9]}
}

# Training and evaluating the models using K-Fold + GridSearchCV for best hyperparameter tuning + model selection
for model in models:
    print(f"\nModel: {model}")
    pipe = Pipeline([("scaler", scaler), ("model", models[model])])

    if model == "Linear Regression":
        pipe.fit(X, y)
        mse = -np.mean(cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=cv))
        cv_best_models[model] = {'best_mse': mse, 'best_params': {}}
        print(f"Best MSE: {mse:.4f}")

    else:
        grid_search = GridSearchCV(pipe, hyperparameters[model], cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best MSE: {-grid_search.best_score_:.4f}")
        cv_best_models[model] = {
            'best_mse': -grid_search.best_score_,
            'best_params': grid_search.best_params_
        }

print("\nModel Wise Performance:")
for model, details in cv_best_models.items():
    print(f"{model}: MSE = {details['best_mse']:.4f}, Params = {details['best_params']}")

best_model = min(cv_best_models.items(), key=lambda x: x[1]['best_mse'])[0]
model_map = {
    'Linear Regression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet
}

best_params = {
    k.replace("model__", ""): v
    for k, v in cv_best_models[best_model]['best_params'].items()
}
best_model = model_map[best_model](**best_params)

# Training the best model on the entire dataset
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)


# Printing the evaluation metrics and plotting the actual vs predicted prices
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()