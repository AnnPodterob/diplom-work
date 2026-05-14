import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- Параметры модели (Sharma et al. 2024) ---
Ad = np.array([[0.017, 0], [1.85e-6, 0.90]])
Bd = np.array([[5.78], [0]])
Cd = np.array([[0, 1]])

# --- Параметры MPC ---
Np = 14
Nc = 7
max_delta_dose = 25.0
max_total_dose = 200.0
target_ft4 = 13.7
initial_ft4 = 8.5

# Веса (попробуем увеличить вес на ошибку, чтобы сделать модель "активнее")
Q_weight = 10.0
R_weight = 0.1

simulation_days = 30
history_ft4 = [initial_ft4]
history_dose = [25.0]

x_k = np.array([[0.0], [initial_ft4]])
current_dose = 25.0

print("Запуск симуляции MPC...")

for day in range(1, simulation_days):
    # Переменная: приращение дозы на каждом шаге горизонта управления
    delta_d = cp.Variable((Nc, 1))

    cost = 0
    constraints = []
    x_curr = x_k

    # Формируем текущую дозу на горизонте
    for i in range(Np):
        idx = min(i, Nc - 1)
        # Доза в момент i — это текущая доза + сумма всех изменений до этого момента
        fut_dose = current_dose + cp.sum(delta_d[:idx + 1])

        # Прогноз состояния
        x_curr = Ad @ x_curr + Bd * (fut_dose / 100.0)
        y_curr = Cd @ x_curr

        # Функция стоимости
        cost += Q_weight * cp.square(y_curr - target_ft4)
        if i < Nc:
            cost += R_weight * cp.square(delta_d[i])
            # Ограничения
            constraints += [cp.abs(delta_d[i]) <= max_delta_dose]
            constraints += [current_dose + cp.sum(delta_d[:i + 1]) <= max_total_dose]
            constraints += [current_dose + cp.sum(delta_d[:i + 1]) >= 0]

    # Решение
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # Используем 'OSQP' как более стабильный решатель для таких задач
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"День {day}: Решение не найдено (Статус: {prob.status}). Проверьте ограничения.")
        applied_delta = 0.0
    else:
        applied_delta = delta_d.value[0, 0]

    # Обновляем состояние
    current_dose += applied_delta
    x_k = Ad @ x_k + Bd * (current_dose / 100.0)

    history_ft4.append(x_k[1, 0])
    history_dose.append(current_dose)

    # print(f"День {day}: FT4={x_k[1, 0]:.2f}, Доза={current_dose:.2f}")

# --- Графики ---
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(history_ft4, 'b-o', label='FT4 Level')
plt.axhline(y=target_ft4, color='r', linestyle='--', label='Target')
plt.ylabel('FT4 (ng/L)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.step(range(simulation_days), history_dose, where='post', color='k', label='Dose')
plt.ylabel('Dose (mcg/day)')
plt.xlabel('Days')
plt.grid(True)
plt.show()