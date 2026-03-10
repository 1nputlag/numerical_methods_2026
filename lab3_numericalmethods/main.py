import os
import csv
import math
import matplotlib.pyplot as plt


# 1. Автоматичне створення файлу data.csv згідно з прикладом у методичці
def create_sample_csv(filename):
    data = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Month", "Temp"])
            writer.writerows(data)


# 2. Зчитування даних з CSV
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаємо заголовок
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


# 3. Формування матриці та вектора системи лінійних алгебраїчних рівнянь
def form_matrix(x, m):
    a = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            a[i][j] = sum(xi ** (i + j) for xi in x)
    return a


def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b


# 4. Розв'язок системи методом Гауса з вибором головного елемента по стовпцях
def gauss_solve(a, b):
    n = len(a)
    a_copy = [row[:] for row in a]
    b_copy = b[:]

    # Прямий хід
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(a_copy[i][k]) > abs(a_copy[max_row][k]):
                max_row = i

        # Перестановка рядків місцями
        a_copy[k], a_copy[max_row] = a_copy[max_row], a_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        if a_copy[k][k] == 0:
            continue

        for i in range(k + 1, n):
            factor = a_copy[i][k] / a_copy[k][k]
            for j in range(k, n):
                a_copy[i][j] -= factor * a_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    # Зворотній хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        if a_copy[i][i] == 0:
            x_sol[i] = 0
        else:
            x_sol[i] = (b_copy[i] - s) / a_copy[i][i]
    return x_sol


# Обчислення значень алгебраїчного многочлена
def polynomial(x_vals, coef):
    return [sum(coef[i] * (xv ** i) for i in range(len(coef))) for xv in x_vals]


# Функція обчислення дисперсії
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return math.sqrt(sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n)


# Лінійна інтерполяція для знаходження дійсного y між вузлами (для похибки)
def get_y_true(x_val, x_nodes, y_nodes):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            return y_nodes[i] + (y_nodes[i + 1] - y_nodes[i]) * (x_val - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    return y_nodes[-1]


# --- Головний блок програми ---
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data.csv')

    # Ініціалізація даних
    create_sample_csv(data_path)
    x, y = read_data(data_path)

    variances = []
    max_degree = 10
    n_nodes = len(x)

    # Завдання 3: Знайти многочлен і дисперсію для m = 1...10
    for m in range(1, max_degree + 1):
        a_mat = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(a_mat, b_vec)
        y_approx = polynomial(x, coef)
        var = calculate_variance(y, y_approx)
        variances.append(var)

    # Вибір оптимального значення m за мінімумом дисперсії
    optimal_m = variances.index(min(variances)) + 1
    print(f"Оптимальний ступінь полінома: m = {optimal_m}")

    # Побудова апроксимації для оптимального полінома
    a_opt = form_matrix(x, optimal_m)
    b_opt = form_vector(x, y, optimal_m)
    coef_opt = gauss_solve(a_opt, b_opt)

    # Точки для плавного графіка
    x_smooth = [x[0] + i * (x[-1] - x[0]) / 200 for i in range(201)]
    y_smooth = polynomial(x_smooth, coef_opt)

    # Завдання 6: Екстраполяція (прогноз) на наступні 3 місяці
    x_future = [25, 26, 27]
    y_future = polynomial(x_future, coef_opt)
    print("Прогноз температур на 25, 26, 27 місяці:", [round(temp, 2) for temp in y_future])

    # Завдання 4: Табулювання похибки з кроком h1
    h1 = (x[-1] - x[0]) / (20 * n_nodes)
    x_err = []
    curr_x = x[0]
    while curr_x <= x[-1]:
        x_err.append(curr_x)
        curr_x += h1

    # --- Побудова графіків ---
    plt.figure(figsize=(16, 5))

    # Графік 1: Дисперсія від степеня
    plt.subplot(1, 3, 1)
    plt.plot(range(1, max_degree + 1), variances, 'b-o')
    plt.axvline(x=optimal_m, color='r', linestyle='--', label=f'Оптимальне m={optimal_m}')
    plt.title("Залежність дисперсії від степеня")
    plt.xlabel("Степінь полінома (m)")
    plt.ylabel("Дисперсія")
    plt.legend()
    plt.grid(True)

    # Графік 2: Фактичні дані, апроксимація та прогноз
    plt.subplot(1, 3, 2)
    plt.plot(x, y, 'ko', label='Фактичні дані')
    plt.plot(x_smooth, y_smooth, 'b-', label=f'Апроксимація (m={optimal_m})')
    plt.plot(x_future, y_future, 'rx--', label='Прогноз (3 міс.)')
    plt.title("Апроксимація та прогноз")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.legend()
    plt.grid(True)

    # Графік 3: Похибки для різних m
    plt.subplot(1, 3, 3)
    for m in range(1, max_degree + 1):
        a_mat = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        c_m = gauss_solve(a_mat, b_vec)

        y_approx_err = polynomial(x_err, c_m)
        y_true_err = [get_y_true(xv, x, y) for xv in x_err]
        error_vals = [abs(y_true_err[i] - y_approx_err[i]) for i in range(len(x_err))]

        if m == optimal_m:
            plt.plot(x_err, error_vals, 'r-', linewidth=2, label=f'm={m} (оптимальний)')
        else:
            plt.plot(x_err, error_vals, alpha=0.3)

    plt.title("Похибка апроксимації на відрізку")
    plt.xlabel("Місяць")
    plt.ylabel("Абсолютна похибка")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()