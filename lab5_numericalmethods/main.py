import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. Задана функція навантаження на сервер
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24

# 2. Знаходження "точного" значення інтегралу I0 за допомогою scipy.integrate
I0, _ = quad(f, a, b)
print(f"2. Точне значення інтегралу I0: {I0:.10f}")


# 3. Функція для обчислення інтегралу методом Сімпсона
def simpson_composite(f, a, b, N):
    if N % 2 != 0:
        N += 1  # N має бути парним для формули Сімпсона
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # Застосування складової квадратурної формули Сімпсона
    I = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return I


# 4. Дослідження залежності точності від кількості розбиттів N
N_vals = np.arange(10, 1002, 2)
errors = []

for N in N_vals:
    I_N = simpson_composite(f, a, b, N)
    errors.append(abs(I_N - I0))

# Побудова графіка похибки
plt.figure(figsize=(10, 5))
plt.plot(N_vals, errors, label='Похибка $\epsilon(N)$')
plt.yscale('log')
plt.axhline(1e-12, color='r', linestyle='--', label='Задана точність $\epsilon=10^{-12}$')
plt.title('Залежність похибки складової формули Сімпсона від $N$')
plt.xlabel('Кількість розбиттів, $N$')
plt.ylabel('Абсолютна похибка, $\epsilon(N)$')
plt.grid(True)
plt.legend()
plt.show()

# Знаходження N_opt (може перевищувати 1000 для досягнення такої високої точності)
N_opt = 10
while True:
    err = abs(simpson_composite(f, a, b, N_opt) - I0)
    if err <= 1e-12:
        break
    N_opt += 2

epsopt = abs(simpson_composite(f, a, b, N_opt) - I0)
print(f"4. N_opt для досягнення точності 1e-12: {N_opt}")
print(f"   Похибка epsopt при N_opt: {epsopt:.2e}")

# 5. Обчислення похибки для N0 = N_opt / 10 (кратне 8)
N0 = max(8, int(round((N_opt / 10) / 8) * 8))
I_N0 = simpson_composite(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"5. Базове N0: {N0}")
print(f"   Похибка eps0 при N0: {eps0:.2e}")

# 6. Уточнення методом Рунге-Ромберга
I_N0_2 = simpson_composite(f, a, b, N0 // 2)
# Метод Рунге-Ромберга для формули Сімпсона (порядок 4, тому знаменник 2^4 - 1 = 15)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)
print(f"6. Значення за методом Рунге-Ромберга: {I_R:.10f}")
print(f"   Похибка epsR: {epsR:.2e}")

# 7. Уточнення методом Ейткена
I_N0_4 = simpson_composite(f, a, b, N0 // 4)

# Оцінка порядку методу
numerator_p = abs(I_N0_4 - I_N0_2)
denominator_p = abs(I_N0_2 - I_N0)
if denominator_p != 0:
    p = (1 / np.log(2)) * np.log(numerator_p / denominator_p)
else:
    p = float('inf')

# Обчислення уточненого значення
numerator_E = I_N0_2 ** 2 - I_N0 * I_N0_4
denominator_E = 2 * I_N0_2 - (I_N0 + I_N0_4)

if denominator_E != 0:
    I_E = numerator_E / denominator_E
else:
    I_E = I_N0  # Fallback, якщо знаменник рівний 0

epsE = abs(I_E - I0)
print(f"7. Значення за методом Ейткена: {I_E:.10f}")
print(f"   Оцінений порядок методу p: {p:.4f}")
print(f"   Похибка epsE: {epsE:.2e}")

# 8. Аналіз результатів
print(f"\n8. Порівняння похибок:")
print(f"   Базова похибка (eps0): {eps0:.2e}")
print(f"   Після Рунге-Ромберга (epsR): {epsR:.2e}")
print(f"   Після Ейткена (epsE): {epsE:.2e}")


# 9. Адаптивний алгоритм інтегрування
def adaptive_simpson(f, a, b, tol, eval_count):
    m = (a + b) / 2
    h = b - a

    # Інтеграл на відрізку без дроблення
    I1 = (h / 6) * (f(a) + 4 * f(m) + f(b))
    eval_count[0] += 3

    # Інтеграл з розбиттям навпіл
    m1 = (a + m) / 2
    m2 = (m + b) / 2
    I2 = (h / 12) * (f(a) + 4 * f(m1) + f(m)) + (h / 12) * (f(m) + 4 * f(m2) + f(b))
    eval_count[0] += 4  # f(a), f(m), f(b) вже обчислені, додаємо лише f(m1), f(m2)

    # Умова збіжності
    if abs(I1 - I2) <= tol:
        return I2
    else:
        # Рекурсивне розбиття
        return (adaptive_simpson(f, a, m, tol / 2, eval_count) +
                adaptive_simpson(f, m, b, tol / 2, eval_count))


print("\n9. Дослідження адаптивного алгоритму:")
tolerances = [1e-3, 1e-6, 1e-9, 1e-12]
for tol in tolerances:
    eval_count = [0]  # Використовуємо список як mutable object для підрахунку викликів
    I_adapt = adaptive_simpson(f, a, b, tol, eval_count)
    err_adapt = abs(I_adapt - I0)
    print(f"   Tolerence: {tol:.1e} | Похибка: {err_adapt:.2e} | Кількість обчислень функції: {eval_count[0]}")