import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.svm import SVC


class Perceptron:
    """Реализация перцептрона Розенблатта с возможностью выбора функции активации"""

    def __init__(self, random_weights=True, epochs=100, activation='step'):
        """
        Инициализация перцептрона
        :param random_weights: Инициализация весов случайным образом (True) или нулями (False)
        :param epochs: Количество эпох обучения
        :param activation: Функция активации ('step', 'sigmoid', 'relu')
        """
        self.random_weights = random_weights
        self.epochs = epochs
        self.activation = activation
        self.weights = None
        self.bias = None
        self.errors_history = []

    def _initialize_parameters(self, n_features):
        """Инициализация весов и смещения"""
        if self.random_weights:
            self.weights = np.random.rand(n_features)
            self.bias = np.random.rand()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

    def _apply_activation(self, x):
        """Применение выбранной функции активации"""
        if self.activation == 'step':
            return 1 if x >= 0 else 0
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return max(0, x)
        else:
            raise ValueError("Неподдерживаемая функция активации")

    def fit(self, X, y):
        """
        Обучение перцептрона
        :param X: Матрица признаков
        :param y: Вектор меток
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                output = np.dot(xi, self.weights) + self.bias
                prediction = self._apply_activation(output)
                update = (target - prediction)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)

            self.errors_history.append(errors)
            if errors == 0:
                break

    def predict(self, X):
        """Предсказание меток для новых данных"""
        linear_output = np.dot(X, self.weights) + self.bias

        if self.activation == 'step':
            return np.where(linear_output >= 0, 1, 0)
        elif self.activation == 'sigmoid':
            return np.where(linear_output >= 0.5, 1, 0)
        elif self.activation == 'relu':
            return np.where(linear_output >= 0, 1, 0)


def visualize_data(X, y, title="Данные для классификации"):
    """Визуализация данных с цветовой маркировкой классов"""
    colors = ('green', 'orange')
    plt.figure(figsize=(8, 6))
    for class_idx in range(2):
        plt.scatter(X[y == class_idx][:, 0],
                    X[y == class_idx][:, 1],
                    c=colors[class_idx],
                    s=50,
                    label=f'Класс {class_idx}')
    plt.title(title)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X, y, title="Разделяющая граница"):
    """Визуализация разделяющей границы модели"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    colors = ('green', 'orange')
    for class_idx in range(2):
        plt.scatter(X[y == class_idx][:, 0],
                    X[y == class_idx][:, 1],
                    c=colors[class_idx],
                    s=50,
                    label=f'Класс {class_idx}')
    plt.title(title)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def part1_synthetic_data():
    """Часть 1: Работа с синтетическими данными"""
    # Генерация данных
    X, y = make_blobs(n_samples=500,
                      centers=([1.1, 3], [4.5, 6.9]),
                      cluster_std=1.3,
                      random_state=0)

    # Визуализация данных
    visualize_data(X, y)

    # Создание и обучение нашего перцептрона
    custom_perceptron = Perceptron(random_weights=True, epochs=100, activation='step')
    custom_perceptron.fit(X, y)
    custom_pred = custom_perceptron.predict(X)
    custom_acc = accuracy_score(y, custom_pred)
    print(f"\nРезультаты нашего перцептрона:")
    print(f"Точность: {custom_acc:.4f}")
    print(f"Количество ошибок на каждой эпохе: {custom_perceptron.errors_history}")

    # Визуализация разделяющей границы
    plot_decision_boundary(custom_perceptron, X, y,
                           "Разделяющая граница нашего перцептрона")

    # Сравнение с перцептроном из sklearn
    sklearn_perceptron = SkPerceptron(max_iter=100, random_state=0)
    sklearn_perceptron.fit(X, y)
    sklearn_pred = sklearn_perceptron.predict(X)
    sklearn_acc = accuracy_score(y, sklearn_pred)
    print(f"\nРезультаты перцептрона из sklearn:")
    print(f"Точность: {sklearn_acc:.4f}")
    print(f"Разница в точности: {abs(custom_acc - sklearn_acc):.4f}")


def part2_iris_data():
    """Часть 2: Работа с данными Iris"""
    # Загрузка данных Iris (последние 100 записей)
    iris = load_iris()
    X = iris.data[-100:]  # Берем только последние 100 записей (классы 1 и 2)
    y = iris.target[-100:]

    # Перцептрон из sklearn
    perceptron = SkPerceptron(max_iter=100, random_state=0)
    perceptron.fit(X, y)
    perceptron_pred = perceptron.predict(X)
    perceptron_acc = accuracy_score(y, perceptron_pred)

    # SVM для сравнения
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    svm_pred = svm.predict(X)
    svm_acc = accuracy_score(y, svm_pred)

    print("\nРезультаты для датасета Iris:")
    print(f"Точность перцептрона: {perceptron_acc:.4f}")
    print(f"Точность SVM: {svm_acc:.4f}")
    print(f"Разница в точности: {abs(perceptron_acc - svm_acc):.4f}")


print("Часть 1: Работа с синтетическими данными")
part1_synthetic_data()
print("\nЧасть 2: Работа с данными Iris")
part2_iris_data()
