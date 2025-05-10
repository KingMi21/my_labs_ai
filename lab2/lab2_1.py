import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка и подготовка данных Iris
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

# Вывод основной информации о данных
print("Первые 5 строк датасета:")
print(iris_df.head())
print("\nИнформация о датасете:")
print(iris_df.info())
print("\nНазвания классов:", iris_data.target_names)


# 1. Визуализация данных Iris
def plot_features(ax, feature_name: str) -> None:
    """Функция для отрисовки зависимости длины от ширины для заданного признака"""
    for target_class in np.unique(iris_data.target):
        ax.scatter(
            iris_df[iris_df['target'] == target_class][f'{feature_name} length (cm)'],
            iris_df[iris_df['target'] == target_class][f'{feature_name} width (cm)'],
            c=['red', 'green', 'blue'][target_class],
            label=iris_data.target_names[target_class]
        )
    ax.set_xlabel(f'Длина {feature_name} (см)')
    ax.set_ylabel(f'Ширина {feature_name} (см)')
    ax.set_title(f'Зависимость длины и ширины {feature_name}')
    ax.legend()


# Создаем графики для сепалов и лепестков
fig, (ax_sepal, ax_petal) = plt.subplots(1, 2, figsize=(12, 5))
plot_features(ax_sepal, 'sepal')
plot_features(ax_petal, 'petal')
plt.tight_layout()
plt.show()

# 2. Pairplot для визуализации всех взаимосвязей
sns.pairplot(iris_df, hue='target', palette='viridis',
             plot_kws={'alpha': 0.7, 's': 80})
plt.suptitle('Парные зависимости признаков Iris dataset', y=1.02)
plt.show()

# 3. Подготовка двух датасетов для бинарной классификации
# Датесет 1: setosa (0) vs versicolor (1)
df_setosa_versicolor = iris_df[iris_df['target'].isin([0, 1])].copy()
# Датесет 2: versicolor (1) vs virginica (2)
df_versicolor_virginica = iris_df[iris_df['target'].isin([1, 2])].copy()


def train_and_evaluate(X, y, dataset_name: str) -> float:
    """Функция для обучения и оценки модели логистической регрессии"""
    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Создание и обучение модели
    model = LogisticRegression(random_state=0, max_iter=200)
    model.fit(X_train, y_train)  # 6. Обучение модели

    # 7. Предсказание и оценка точности
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # 8. Оценка точности

    print(f"\nРезультаты для датасета {dataset_name}:")
    print(f"Точность модели: {accuracy:.2f}")
    print(f"Коэффициенты модели: {model.coef_}")
    print(f"Свободный член: {model.intercept_}")

    return accuracy


# Обучаем и оцениваем модели для обоих датасетов
accuracy1 = train_and_evaluate(
    df_setosa_versicolor.drop('target', axis=1),
    df_setosa_versicolor['target'],
    "Setosa vs Versicolor"
)

accuracy2 = train_and_evaluate(
    df_versicolor_virginica.drop('target', axis=1),
    df_versicolor_virginica['target'],
    "Versicolor vs Virginica"
)

# 9. Генерация синтетического датасета и классификация
X_synth, y_synth = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1
)

# Визуализация синтетических данных
plt.figure(figsize=(10, 6))
plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap='viridis', alpha=0.6)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Сгенерированный датасет для бинарной классификации')
plt.colorbar(label='Класс')
plt.grid(True)
plt.show()

# Обучение и оценка на синтетических данных
accuracy_synth = train_and_evaluate(
    X_synth, y_synth,
    "Синтетические данные"
)

# Сравнение точности всех моделей
print("\nСравнение точности моделей:")
print(f"1. Setosa vs Versicolor: {accuracy1:.2f}")
print(f"2. Versicolor vs Virginica: {accuracy2:.2f}")
print(f"3. Синтетические данные: {accuracy_synth:.2f}")