import requests
import numpy as np
import matplotlib.pyplot as plt


def get_data_from_url():
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,%2024.536044|48.164983,%2024.534836|48.165605,%2024.534068|48.166228,%2024.532915|48.166777,%2024.531927|48.167326,%2024.530884|48.167011,%2024.530061|48.166053,%2024.528039|48.166655,%2024.526064|48.166497,%2024.523574|48.166128,%2024.520214|48.165416,%2024.517170|48.164546,%2024.514640|48.163412,%2024.512980|48.162331,%2024.511715|48.162015,%2024.509462|48.162147,%2024.506932|48.161751,%2024.504244|48.161197,%2024.501793|48.160580,%2024.500537|48.160250,%2024.500106"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()['results']
        else:
            return []
    except Exception:
        return []


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_spline_coefficients(x, y):
    n = len(x)
    h = np.diff(x)
    dim = n - 2

    main_diag = np.zeros(dim)
    upper_diag = np.zeros(dim - 1)
    lower_diag = np.zeros(dim - 1)
    rhs = np.zeros(dim)

    for i in range(dim):
        main_diag[i] = 2 * (h[i] + h[i + 1])
        rhs[i] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
        if i < dim - 1:
            upper_diag[i] = h[i + 1]
            lower_diag[i] = h[i + 1]

    P = np.zeros(dim)
    Q = np.zeros(dim)
    P[0] = -upper_diag[0] / main_diag[0]
    Q[0] = rhs[0] / main_diag[0]

    for i in range(1, dim):
        denom = main_diag[i] + lower_diag[i - 1] * P[i - 1]
        if i < dim - 1:
            P[i] = -upper_diag[i] / denom
        Q[i] = (rhs[i] - lower_diag[i - 1] * Q[i - 1]) / denom

    c_inner = np.zeros(dim)
    c_inner[-1] = Q[-1]
    for i in range(dim - 2, -1, -1):
        c_inner[i] = P[i] * c_inner[i + 1] + Q[i]

    c = np.zeros(n)
    c[1:-1] = c_inner

    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])

    return a, b, c[:-1], d


def spline_eval_array(x_vals, x_nodes, a, b, c, d):
    y_vals = []
    for val in x_vals:
        found = False
        for i in range(len(x_nodes) - 1):
            if x_nodes[i] <= val <= x_nodes[i + 1]:
                dx = val - x_nodes[i]
                y = a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)
                y_vals.append(y)
                found = True
                break
        if not found:
            y_vals.append(x_nodes[-1])
    return np.array(y_vals)


data = get_data_from_url()
if not data:
    exit()

lats = [p['latitude'] for p in data]
lons = [p['longitude'] for p in data]
elevs = [p['elevation'] for p in data]

dist = [0.0]
for i in range(1, len(data)):
    d = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
    dist.append(dist[-1] + d)

x_full = np.array(dist)
y_full = np.array(elevs)

indices_subset = np.arange(0, len(x_full), 2)
if indices_subset[-1] != len(x_full) - 1:
    indices_subset = np.append(indices_subset, len(x_full) - 1)
x_sub = x_full[indices_subset]
y_sub = y_full[indices_subset]

a1, b1, c1, d1 = compute_spline_coefficients(x_full, y_full)
a2, b2, c2, d2 = compute_spline_coefficients(x_sub, y_sub)

x_smooth = np.linspace(x_full[0], x_full[-1], 500)
y_smooth_full = spline_eval_array(x_smooth, x_full, a1, b1, c1, d1)
y_smooth_sub = spline_eval_array(x_smooth, x_sub, a2, b2, c2, d2)

error_smooth = np.abs(y_smooth_full - y_smooth_sub)

y_sub_at_full_nodes = spline_eval_array(x_full, x_sub, a2, b2, c2, d2)
error_at_nodes = np.abs(y_full - y_sub_at_full_nodes)

print(f"1. Загальна довжина: {x_full[-1]:.2f} м")

total_ascent = 0
total_descent = 0
for i in range(1, len(y_full)):
    diff = y_full[i] - y_full[i - 1]
    if diff > 0:
        total_ascent += diff
    else:
        total_descent += abs(diff)

print(f"2. Сумарний набір висоти: {total_ascent:.2f} м")
print(f"3. Сумарний спуск: {total_descent:.2f} м")

grad_full = np.gradient(y_smooth_full, x_smooth) * 100
print(f"4. Максимальний підйом: {np.max(grad_full):.2f} %")
print(f"5. Максимальний спуск: {np.min(grad_full):.2f} %")
print(f"6. Середній градієнт: {np.mean(np.abs(grad_full)):.2f} %")

energy_kcal = (80 * 9.81 * total_ascent) / 4184
print(f"7. Витрачена енергія (80 кг): {energy_kcal:.2f} ккал")

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x_full, y_full, 'ko', markersize=5, label='Вузли')
axes[0].plot(x_smooth, y_smooth_full, 'b-', linewidth=2, label='Сплайн')
axes[0].set_title('Профіль висоти')
axes[0].set_ylabel('Висота (м)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(x_smooth, y_smooth_full, 'b-', alpha=0.3, label='Еталон')
axes[1].plot(x_sub, y_sub, 'ro', markersize=5, label='Кожен 2-й вузол')
axes[1].plot(x_smooth, y_smooth_sub, 'r--', label='Спрощений сплайн')
axes[1].set_title('Порівняння точності')
axes[1].set_ylabel('Висота (м)')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(x_smooth, error_smooth, 'g-', label='Лінія похибки')
axes[2].fill_between(x_smooth, error_smooth, color='green', alpha=0.1)
axes[2].plot(x_full, error_at_nodes, 'ko', markersize=4, label='Похибка у вузлах')
axes[2].set_title('Графік похибок')
axes[2].set_xlabel('Відстань (м)')
axes[2].set_ylabel('Похибка (м)')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()