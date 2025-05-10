import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# Создаем сетку графиков 2x2
fig, ((ax_initial, ax_regression), (ax_squares, ax_empty)) = plt.subplots(2, 2)
ax_empty.axis('off')  # Отключаем четвертый график (не используется)

# Устанавливаем заголовки для графиков
ax_initial.set_title('Исходные данные')
ax_regression.set_title('Линейная регрессия')
ax_squares.set_title('Квадраты ошибок')


def plot_initial_data(x: list, y: list, selected_columns: tuple[str, str]) -> None:
    """
    Отображает исходные данные в виде точечного графика.

    Параметры:
        x: список значений по оси X
        y: список значений по оси Y
        selected_columns: кортеж с названиями колонок для осей X и Y
    """
    ax_initial.scatter(x, y, color='blue')
    ax_initial.set_title(f"{selected_columns[0]} vs {selected_columns[1]}")
    ax_initial.set_xlabel(selected_columns[0])
    ax_initial.set_ylabel(selected_columns[1])


def calculate_regression_params(x: list, y: list) -> tuple[float, float]:
    """
    Вычисляет параметры линейной регрессии (w0, w1) методом наименьших квадратов.

    Параметры:
        x: список значений по оси X
        y: список значений по оси Y

    Возвращает:
        Кортеж (w0, w1) - коэффициенты линейной регрессии y = w0 + w1*x
    """
    n = len(x)
    # Вычисляем коэффициент w1 (наклон)
    w1 = (1 / n * sum([xi * sum(y) for xi in x]) - sum([y[i] * x[i] for i in range(n)])) / \
         (1 / n * sum([xi * sum(x) for xi in x]) - sum([xi ** 2 for xi in x]))
    # Вычисляем коэффициент w0 (свободный член)
    w0 = sum([y[i] - w1 * x[i] for i in range(n)]) / n
    return w0, w1


def plot_regression_line(x: list, y: list, w0: int, w1: int, selected_columns: tuple[str, str]) -> None:
    """
    Отображает исходные данные и линию регрессии.

    Параметры:
        x: список значений по оси X
        y: список значений по оси Y
        w0: свободный член уравнения регрессии
        w1: коэффициент наклона уравнения регрессии
        selected_columns: кортеж с названиями колонок для осей X и Y
    """
    ax_regression.scatter(x, y, color='blue')
    ax_regression.set_xlabel(selected_columns[0])
    ax_regression.set_ylabel(selected_columns[1])
    # Рисуем линию регрессии
    ax_regression.axline(xy1=(0, w0), slope=w1, color='red')


def plot_error_squares(x: list, y: list, w0: int, w1: int, selected_columns: tuple[str, str]) -> None:
    """
    Отображает квадраты ошибок между фактическими значениями и предсказаниями.

    Параметры:
        x: список значений по оси X
        y: список значений по оси Y
        w0: свободный член уравнения регрессии
        w1: коэффициент наклона уравнения регрессии
        selected_columns: кортеж с названиями колонок для осей X и Y
    """
    ax_squares.scatter(x, y, color='blue')
    ax_squares.set_xlabel(selected_columns[0])
    ax_squares.set_ylabel(selected_columns[1])
    # Вычисляем предсказанные значения
    y_pred = [w0 + w1 * xi for xi in x]

    # Рисуем линию регрессии
    ax_squares.axline(xy1=(0, w0), slope=w1, color='red')

    # Получаем размеры графика для корректного отображения квадратов
    fig_bbox = ax_squares.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Вычисляем разницу между масштабами осей
    scale_ratio = (ax_squares.get_xlim()[1] - ax_squares.get_xlim()[0]) / \
                  (ax_squares.get_ylim()[1] - ax_squares.get_ylim()[0]) / \
                  (fig_bbox.width / fig_bbox.height)

    # Удаляем старые квадраты (если есть)
    for patch in ax_squares.patches:
        patch.remove()

    # Рисуем квадраты ошибок
    for i in range(len(x)):
        if y[i] < y_pred[i]:
            # Если точка ниже линии регрессии
            rect = patches.Rectangle(
                (x[i], y[i]),
                -abs(y_pred[i] - y[i]) * scale_ratio,
                abs(y_pred[i] - y[i]),
                color='green', alpha=0.4)
        else:
            # Если точка выше линии регрессии
            rect = patches.Rectangle(
                (x[i], y_pred[i]),
                abs(y_pred[i] - y[i]) * scale_ratio,
                abs(y_pred[i] - y[i]),
                color='green', alpha=0.4)
        ax_squares.add_patch(rect)


# Чтение данных из CSV файла
data = pd.read_csv(r'student_scores.csv')
# Альтернативный вариант с вводом пути от пользователя:
# data = pd.read_csv(input("Введите путь к файлу относительно текущей директории:"))
# Вывод статистической информации о данных
print(data.describe())
# Выбор варианта отображения данных
print("Выберите вариант отображения графика (введите номер):")
column_options = [
    (data.columns[0], data.columns[1]),
    (data.columns[1], data.columns[0])
]
for idx, option in enumerate(column_options):
    print(f"{idx + 1}. " + ' : '.join(option))
selected_option = int(input())
# Выбор данных в соответствии с выбранным вариантом
match selected_option:
    case 1:
        x, y = data[data.columns[0]], data[data.columns[1]]
    case 2:
        x, y = data[data.columns[1]], data[data.columns[0]]
    case _:
        raise Exception("Выбран несуществующий вариант")
# Выполнение всех этапов лабораторной работы
plot_initial_data(x, y, column_options[selected_option - 1])
w0, w1 = calculate_regression_params(x, y)
plot_regression_line(x, y, w0, w1, column_options[selected_option - 1])
plot_error_squares(x, y, w0, w1, column_options[selected_option - 1])
# Отображение всех графиков
plt.tight_layout()
plt.show()
