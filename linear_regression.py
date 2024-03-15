import pandas

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from content import df

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Cat Boost': CatBoostRegressor(verbose=False),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

models_PCA = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Cat Boost': CatBoostRegressor(verbose=False),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

res = pandas.DataFrame(columns=['Model', 'R2 score', 'Mean Squared Error'])

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

for name, model in models.items():

    pca_model = models_PCA[name]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    res.loc[len(res.index)] = [name, r2, mse]

    pca_model.fit(X_train_pca, y_train)
    y_pred = pca_model.predict(X_test_pca)

    mse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    res.loc[len(res.index)] = [name + '_PCA', r2, mse]


# model.score(X_test, y_test) - зачем это

print(res)

# 1
best_model = models['Cat Boost']
feature_importance = best_model.feature_importances_
feature_importance_dict = dict(zip(X.columns, feature_importance))

sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nFeature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


selected_features = [feature for feature, importance in sorted_feature_importance[:len(sorted_feature_importance)//2]]

X_selected = X[selected_features]

X_train_df = pandas.DataFrame(data=X_train, columns=X.columns)
X_test_df = pandas.DataFrame(data=X_test, columns=X.columns)

X_train_selected, X_test_selected = X_train_df[selected_features], X_test_df[selected_features]

best_model.fit(X_train_selected, y_train)
y_pred_selected = best_model.predict(X_test_selected)

mse_selected = mean_squared_error(y_test, y_pred_selected)
r2_selected = r2_score(y_test, y_pred_selected)

print(f"Model with selected features: MSE = {mse_selected**0.5}, R^2 = {r2_selected}")

# 3
cb = CatBoostRegressor(verbose=False)
cb.fit(X_train, y_train)
yp = cb.predict(X_test)

param_grid = {
    'iterations': [200],
    'learning_rate': [0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
    'depth': [2, 3, 4, 5]
}
grid_search = GridSearchCV(CatBoostRegressor(verbose=False), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

y_pred_best_params = best_model.predict(X_test)
mse_best_params = mean_squared_error(y_test, y_pred_best_params)
r2_best_params = r2_score(y_test, y_pred_best_params)

print(f"Model with best parameters: MSE = {mse_best_params**0.5}, R^2 = {r2_best_params}")

cv_scores = cross_val_score(best_model, X_selected, y, cv=5)
print(f"Cross-validated R^2 scores: {cv_scores}")

train_sizes, train_scores, test_scores = learning_curve(grid_search, X_selected, y, cv=5)

shift = 0

plt.figure()
plt.plot(train_sizes[shift:], train_scores.mean(axis=1)[shift:], label='Train')

plt.plot(train_sizes[shift:], test_scores.mean(axis=1)[shift:], label='Test')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.show()
