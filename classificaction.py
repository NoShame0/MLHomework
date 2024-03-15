import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA

from content import df

res = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Предположим, что в качестве целевой переменной выберем "Price"
# Разделим данные на признаки и целевую переменную
X = df.drop(columns=['Price'])
y = df['Price']

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создадим пайплайн для предобработки данных
numeric_features = [
    'Release date',
    'Max resolution',
    'Low resolution',
    'Effective pixels',
    'Zoom wide (W)',
    'Zoom tele (T)',
    'Normal focus range',
    'Macro focus range',
    'Storage included',
    'Weight (inc. batteries)',
    'Dimensions'
]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Обучим модели и оценим их производительность
for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    res.loc[len(res.index)] = [name, accuracy, precision, recall, f1]

    # С уменьшением размерности
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=2)),
            ('classifier', model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    res.loc[len(res.index)] = [name + '_PCA', accuracy, precision, recall, f1]

print(res)

# Оптимизация лучшей модели
best_model = RandomForestClassifier()
best_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ]
)

# Подберем оптимальные параметры с помощью GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 125, 150],
    'classifier__max_depth': [None, 10, 20, 25, 30, 35, 40],
    'classifier__min_samples_split': [2, 3, 4, 5, 8, 10, 15]
}

grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("\n")

# Оценка модели с лучшими параметрами
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Best Model Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

