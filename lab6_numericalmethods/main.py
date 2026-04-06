import numpy as np


# ==========================================
# 1. Генерація та збереження даних
# ==========================================
def generate_and_save_data(n=100, x_val=2.5, file_A="matrix_A.txt", file_B="vector_B.txt"):
    """
    Генерує матрицю А, задає точний розв'язок та обчислює вектор B.
    Зберігає згенеровані дані у текстові файли.
    """
    # Генеруємо випадкову матрицю A (n x n)
    A = np.random.rand(n, n) * 10

    # Задаємо точний розв'язок системи, де всі x_j = 2.5
    X_exact = np.full(n, x_val)

    # Обчислюємо вектор вільних членів B = A * X
    B = np.dot(A, X_exact)

    # Записуємо матрицю A та вектор B у текстові файли
    np.savetxt(file_A, A)
    np.savetxt(file_B, B)

    return A, B, X_exact


# ==========================================
# 2. Опис необхідних функцій (Завдання 2)
# ==========================================
def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)


def lu_decomposition(A):
    """
    Знаходження LU-розкладу матриці А.
    L - нижня трикутна, U - верхня трикутна (з 1 на діагоналі).
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Задаємо значення діагональних елементів матриці U рівними 1
    for i in range(n):
        U[i][i] = 1.0

    # Почергово знаходимо стовпці L та рядки U
    for k in range(n):
        # Елементи k-го стовпця матриці L
        for i in range(k, n):
            sum_L = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_L

        # Елементи k-го рядка матриці U
        for i in range(k + 1, n):
            sum_U = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_U) / L[k][k]

    return L, U


def write_lu(L, U, filename="matrix_LU.txt"):
    """Запис LU-розкладу в текстовий файл."""
    with open(filename, 'w') as f:
        f.write("Matrix L:\n")
        np.savetxt(f, L)
        f.write("\nMatrix U:\n")
        np.savetxt(f, U)


def solve_lu(L, U, B):
    """
    Розв'язок системи рівнянь AX=B за допомогою LU-розкладу.
    Спочатку розв'язуємо LZ = B, потім UX = Z.
    """
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)

    # Прямий хід: LZ = B
    for i in range(n):
        sum_Z = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - sum_Z) / L[i][i]

    # Зворотний хід: UX = Z
    for i in range(n - 1, -1, -1):
        sum_X = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - sum_X

    return X


def matrix_vector_mult(A, X):
    """Обчислення добутку матриці на вектор."""
    return np.dot(A, X)


def vector_norm(V):
    """Обчислення норми вектора (максимальне по модулю значення)."""
    return np.max(np.abs(V))


# ==========================================
# 3-5. Виконання алгоритмів та ітераційне уточнення
# ==========================================
def main():
    n = 100

    # 1. Генеруємо та зберігаємо дані
    generate_and_save_data(n)

    # 2. Зчитуємо дані з файлів
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    # Знаходимо LU-розклад та записуємо у файл
    L, U = lu_decomposition(A)
    write_lu(L, U)

    # 3. Розв'язуємо систему AX = B
    X_calc = solve_lu(L, U, B)

    # 4. Оцінюємо точність знайденого розв'язку
    B_calc = matrix_vector_mult(A, X_calc)
    residual_vector = B_calc - B
    eps = vector_norm(residual_vector)
    print(f"Початкова точність (максимальна похибка розв'язку): {eps}")

    # 5. Ітераційне уточнення розв'язку СЛАР
    eps_0 = 1e-14
    X_iter = X_calc.copy()
    iterations = 0

    print(f"Починаємо ітераційне уточнення (цільова точність: {eps_0})...")

    while True:
        # Обчислюємо вектор нев'язки R = B - A * X^(k)
        B_curr = matrix_vector_mult(A, X_iter)
        R = B - B_curr
        current_eps = vector_norm(R)

        # Перевірка умови виходу
        if current_eps <= eps_0:
            break

        # Розв'язуємо систему відносно похибки: A * dX = R (використовуючи існуючі L та U)
        dX = solve_lu(L, U, R)

        # Уточнюємо розв'язок
        X_iter = X_iter + dX
        iterations += 1

        # Захист від нескінченного циклу через обмеження типу float
        if iterations >= 50:
            print("Увага: Досягнуто машинної межі точності для типу float64.")
            break

    print("-" * 40)
    print(f"Кількість ітерацій для уточнення: {iterations}")
    print(f"Кінцева точність (норма нев'язки): {current_eps}")
    print(f"Фрагмент уточненого розв'язку: {X_iter[:5]} ...")


if __name__ == '__main__':
    main()