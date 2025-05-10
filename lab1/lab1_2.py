import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Загрузка данных о диабете
diabetes_data = datasets.load_diabetes()
diabetes_df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
diabetes_df['target'] = diabetes_data.target

# Выбор признака BMI (индекс массы тела) и целевой переменной
X = diabetes_df[['bmi']].values  # Признак
y = diabetes_df['target'].values  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки (60%/40%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=45
)

# 1. Реализация с использованием Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)  # Обучение модели
sklearn_predictions = sklearn_model.predict(X_test)  # Прогнозирование
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)  # Оценка ошибки

# 2. Кастомная реализация линейной регрессии
class CustomLinearRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        self.learning_rate = learning_rate  # Скорость обучения
        self.iterations = iterations  # Количество итераций
        self.weights = None  # Веса модели
        self.bias = 0  # Смещение

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]  # Количество образцов
        self.weights = np.zeros(X.shape[1])  # Инициализация весов нулями

        # Градиентный спуск
        for _ in range(self.iterations):
            predictions = np.dot(X, self.weights) + self.bias
            # Вычисление градиентов
            dw = (2 / n_samples) * np.dot(X.T, (predictions - y))
            db = (2 / n_samples) * np.sum(predictions - y)
            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

# Обучение и прогнозирование кастомной модели
custom_model = CustomLinearRegression(learning_rate=0.01, iterations=1000)
custom_model.fit(X_train, y_train)
custom_predictions = custom_model.predict(X_test)
custom_mse = mean_squared_error(y_test, custom_predictions)

# Вывод результатов сравнения моделей
print("\nРезультаты сравнения моделей:")
print(f"Sklearn MSE: {sklearn_mse:.2f}")
print(f"Коэффициенты Sklearn: w={sklearn_model.coef_[0]:.2f}, b={sklearn_model.intercept_:.2f}")
print(f"\nCustom MSE: {custom_mse:.2f}")
print(f"Коэффициенты Custom: w={custom_model.weights[0]:.2f}, b={custom_model.bias:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Реальные значения')
plt.plot(X_test, sklearn_predictions, color='red', linewidth=2,
         linestyle='dashed', label='Sklearn модель')
plt.plot(X_test, custom_predictions, color='blue', linestyle='dashed',
         label='Кастомная модель')
plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Уровень глюкозы')
plt.title('Сравнение моделей линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()

# Таблица с результатами прогнозирования
results_df = pd.DataFrame({
    'Реальные значения': y_test,
    'Прогноз Sklearn': sklearn_predictions,
    'Прогноз Custom': custom_predictions
})

print("\nПервые 20 прогнозов:")
print(results_df.head(20))