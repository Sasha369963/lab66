import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Завдання 1: Генерація даних
np.random.seed(42)
true_k = 2.5  # Справжній нахил лінії
true_b = 1.0  # Справжній перехоплення
n_points = 100
x = np.random.uniform(-10, 10, n_points)
y = true_k * x + true_b + np.random.normal(0, 5, n_points)  # Додаємо шум

# Метод найменших квадратів
def least_squares(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    k_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b_hat = y_mean - k_hat * x_mean
    return k_hat, b_hat

# Параметри лінії методом найменших квадратів
k_hat, b_hat = least_squares(x, y)

# Параметри за допомогою np.polyfit
polyfit_params = np.polyfit(x, y, 1)
k_polyfit, b_polyfit = polyfit_params

# Побудова графіка для Завдання 1
plt.scatter(x, y, color='gray', label="Дані", alpha=0.6)
plt.plot(x, true_k * x + true_b, label="Справжня лінія", linestyle='dashed')
plt.plot(x, k_hat * x + b_hat, label=f"Найменші квадрати (k={k_hat:.2f}, b={b_hat:.2f})")
plt.plot(x, k_polyfit * x + b_polyfit, label=f"np.polyfit (k={k_polyfit:.2f}, b={b_polyfit:.2f})")

plt.legend()
plt.title("Порівняння методів регресії")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# Завдання 2: Градієнтний спуск
def gradient_descent(x, y, learning_rate=0.01, n_iter=1000):
    n = len(x)
    b = 0
    k = 0
    errors = []
    for i in range(n_iter):
        y_pred = k * x + b
        error = np.mean((y - y_pred) ** 2)
        errors.append(error)
        b_grad = -2 * np.mean(y - y_pred)
        k_grad = -2 * np.mean(x * (y - y_pred))
        b -= learning_rate * b_grad
        k -= learning_rate * k_grad
    return k, b, errors

# Виконання градієнтного спуску
learning_rate = 0.01
n_iter = 1000
k_gd, b_gd, errors = gradient_descent(x, y, learning_rate, n_iter)

# Додавання лінії регресії методом градієнтного спуску
plt.scatter(x, y, color='gray', label="Дані", alpha=0.6)
plt.plot(x, true_k * x + true_b, label="Справжня лінія", linestyle='dashed')
plt.plot(x, k_hat * x + b_hat, label=f"Найменші квадрати (k={k_hat:.2f}, b={b_hat:.2f})")
plt.plot(x, k_polyfit * x + b_polyfit, label=f"np.polyfit (k={k_polyfit:.2f}, b={b_polyfit:.2f})")
plt.plot(x, k_gd * x + b_gd, label=f"Градієнтний спуск (k={k_gd:.2f}, b={b_gd:.2f})")

plt.legend()
plt.title("Порівняння ліній регресії")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# Побудова графіка похибки
plt.plot(range(n_iter), errors, label="Похибка")
plt.title("Залежність похибки від кількості ітерацій")
plt.xlabel("Кількість ітерацій")
plt.ylabel("Середньоквадратична похибка")
plt.grid()
plt.legend()
plt.show()

# Таблиця з результатами
def results_table(true_k, true_b, k_ls, b_ls, k_poly, b_poly, k_gd, b_gd):
    data = {
        "Метод": ["Справжня лінія", "Найменші квадрати", "np.polyfit", "Градієнтний спуск"],
        "k (нахил)": [true_k, k_ls, k_poly, k_gd],
        "b (перехоплення)": [true_b, b_ls, b_poly, b_gd],
    }
    df = pd.DataFrame(data)
    return df

# Створення таблиці з результатами
results_df = results_table(true_k, true_b, k_hat, b_hat, k_polyfit, b_polyfit, k_gd, b_gd)

# Відображення таблиці
from IPython.display import display
display(results_df)
