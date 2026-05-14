import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.signal import cont2discrete
import cvxpy as cp

from thyrosim_patient2 import ThyrosimPatient

# ================== 1. Генерация идентификационных данных ==================
def generate_identification_data(patient, days=60, seed=42):
    """
    Генерирует почасовые данные: время (ч), FT4 (нг/л), TSH (мЕд/л), скорость дозы (мкг/ч)
    """
    np.random.seed(seed)
    # Случайные ежедневные дозы (мкг/сут)
    daily_doses = 50 + 30 * np.random.randn(days)
    daily_doses = np.clip(daily_doses, 10, 200)

    # Временная сетка с шагом 1 час
    t = np.linspace(0, days*24, days*24 + 1)
    n_steps = len(t) - 1

    ft4_rec = np.zeros(n_steps + 1)
    tsh_rec = np.zeros(n_steps + 1)
    dose_rec = np.zeros(n_steps + 1)          # скорость дозы, мкг/ч

    state = patient.state.copy()
    ft4_rec[0] = state[0] / 3.0
    tsh_rec[0] = state[6]
    dose_rec[0] = 0.0

    for i in range(n_steps):
        day = i // 24
        dose = daily_doses[day] / 24.0
        t_span = [t[i], t[i+1]]
        sol = odeint(patient.equations, state, t_span, args=(dose,))
        state = sol[-1]
        ft4_rec[i+1] = state[0] / 3.0
        tsh_rec[i+1] = state[6]
        dose_rec[i+1] = dose

    return t, ft4_rec, tsh_rec, dose_rec, daily_doses

# ================== 2. Модель (2) для идентификации ==================
class ThyroidModel:
    def __init__(self, k_exc, k_sec, v_max, k_m, k_reac):
        self.k_exc = k_exc
        self.k_sec = k_sec
        self.v_max = v_max
        self.k_m = k_m
        self.k_reac = k_reac

    def rhs(self, x, t, d, tsh):
        dx1 = d - self.k_exc * x[0]
        dx2 = self.v_max * tsh / (self.k_m + tsh) + self.k_reac * x[0] - self.k_sec * x[1]
        return [dx1, dx2]

    def simulate(self, t, d_array, tsh_array, x0=None):
        if x0 is None:
            x0 = [0.0, tsh_array[0]]
        n = len(t)
        x = np.zeros((n, 2))
        x[0] = x0
        for i in range(n-1):
            dt = t[i+1] - t[i]
            d_i = d_array[i]
            tsh_i = tsh_array[i]
            t_span = [t[i], t[i+1]]
            sol = odeint(self.rhs, x[i], t_span, args=(d_i, tsh_i))
            x[i+1] = sol[-1]
        return x[:, 1]

# ================== 3. Функция потерь и оптимизация ==================
def loss_func(params, t, d_meas, tsh_meas, ft4_meas, k_exc_fixed, k_sec_fixed):
    v_max, k_m, k_reac = params
    model = ThyroidModel(k_exc_fixed, k_sec_fixed, v_max, k_m, k_reac)
    ft4_sim = model.simulate(t, d_meas, tsh_meas, x0=[0.0, ft4_meas[0]])
    return ft4_sim - ft4_meas

def estimate_parameters(t, d_meas, tsh_meas, ft4_meas):
    k_exc_fixed = 0.171      # 1/ч
    k_sec_fixed = 0.00438    # 1/ч

    p0 = [2.0, 5.0, 0.3]
    bounds = ([0, 0, 0], [100, 100, 10])

    result = least_squares(
        loss_func, p0,
        args=(t, d_meas, tsh_meas, ft4_meas, k_exc_fixed, k_sec_fixed),
        bounds=bounds,
        verbose=2
    )
    v_max_opt, k_m_opt, k_reac_opt = result.x
    print(f"Оценённые параметры: v_max={v_max_opt:.4f}, k_m={k_m_opt:.4f}, k_reac={k_reac_opt:.4f}")
    return k_exc_fixed, k_sec_fixed, k_reac_opt

# ================== 4. Построение дискретной модели для MPC ==================
def build_mpc_matrices(k_exc, k_sec, k_reac, Ts=24.0):
    Ac = np.array([[-k_exc, 0],
                   [k_reac, -k_sec]])
    Bc = np.array([[1],
                   [0]])
    Cc = np.array([[0, 1]])
    Dc = np.array([[0]])

    sys_d = cont2discrete((Ac, Bc, Cc, Dc), Ts, method='zoh')
    Ad, Bd, Cd, Dd, _ = sys_d
    return Ad, Bd, Cd

# ================== 5. Функция добавления шума ==================
def add_noise(value, percent):
    """Добавляет относительный шум к значению (нормальное распределение)."""
    noise = np.random.normal(0, percent * value)
    return max(0, value + noise)

# ================== 6. Эксперимент №4 с шумом измерений (MPC) ==================
if __name__ == "__main__":
    # --- Создание виртуального пациента ---
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        condition='moderate hypothyroidism'
    )

    # --- Идентификация параметров (по чистым данным) ---
    print("Генерация обучающих данных...")
    t, ft4_meas, tsh_meas, d_meas, daily_doses = generate_identification_data(patient, days=60, seed=123)

    print("Идентификация параметров...")
    k_exc, k_sec, k_reac = estimate_parameters(t, d_meas, tsh_meas, ft4_meas)

    # --- Построение матриц MPC ---
    Ad, Bd, Cd = build_mpc_matrices(k_exc, k_sec, k_reac, Ts=24.0)
    print("Матрицы модели MPC:")
    print("Ad =\n", Ad)
    print("Bd =\n", Bd)

    # --- Параметры симуляции ---
    simulation_days = 360          # 12 месяцев
    target_ft4 = 13.7
    Np = 14
    Nc = 7
    max_delta_dose = 12.5
    max_total_dose = 200.0
    Q_weight = 5000.0
    R_weight = 5000.0           # изменена ошибка с 1.0 до 5000.0, чтобы не давать алгоритму сильно менять дозу и прыгать

    # Параметры шума
    noise_ft4_percent = 0.07       # 7% шума FT4
    noise_tsh_percent = 0.05       # 5% шума TSH
    np.random.seed(42)             # воспроизводимость

    # --- Сброс пациента и инициализация ---
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        condition='moderate hypothyroidism'
    )
    current_ft4 = patient.state[0] / 3.0
    x_k = np.array([[0.0], [current_ft4]])
    current_dose = 50.0

    # Хранилища для результатов
    true_ft4_history = []
    true_tsh_history = []
    noisy_ft4_history = []
    noisy_tsh_history = []
    dose_history = []

    print("\nЗапуск симуляции MPC с шумом измерений...")

    for day in range(simulation_days):
        # ---- Применение дозы к пациенту ----
        state = patient.run_simulation(current_dose, hours=24)
        true_ft4 = state[0] / 3.0
        true_tsh = state[6]

        # ---- Добавление шума к измерениям ----
        noisy_ft4 = add_noise(true_ft4, noise_ft4_percent)
        noisy_tsh = add_noise(true_tsh, noise_tsh_percent)

        # ---- Обновление внутреннего состояния контроллера (с использованием зашумлённых данных) ----
        dose_rate = current_dose / 24.0
        x_k = Ad @ x_k + Bd * dose_rate
        x_k[1, 0] = noisy_ft4          # коррекция по зашумлённому FT4

        # ---- Оптимизация MPC для вычисления приращения дозы ----
        delta_d = cp.Variable((Nc, 1))
        cost = 0
        constraints = []
        x_curr = x_k.copy()

        for i in range(Np):
            idx = min(i, Nc - 1)
            fut_dose = current_dose + cp.sum(delta_d[:idx+1])
            fut_dose_rate = fut_dose / 24.0
            x_curr = Ad @ x_curr + Bd * fut_dose_rate
            y_curr = Cd @ x_curr

            cost += Q_weight * cp.square(y_curr - target_ft4)
            if i < Nc:
                cost += R_weight * cp.square(delta_d[i])
                constraints += [cp.abs(delta_d[i]) <= max_delta_dose]
                constraints += [current_dose + cp.sum(delta_d[:i+1]) <= max_total_dose]
                constraints += [current_dose + cp.sum(delta_d[:i+1]) >= 0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            applied_delta = 0.0
        else:
            applied_delta = delta_d.value[0, 0]

        # ---- Обновление дозы ----
        current_dose = np.clip(current_dose + applied_delta, 0, max_total_dose)

        # ---- Сохранение данных ----
        true_ft4_history.append(true_ft4)
        true_tsh_history.append(true_tsh)
        noisy_ft4_history.append(noisy_ft4)
        noisy_tsh_history.append(noisy_tsh)
        dose_history.append(current_dose)

        # ---- Печать каждые 30 дней ----
        if day % 30 == 0:
            print(f"Месяц {day//30+1}: FT4 ист={true_ft4:.2f}, FT4 шум={noisy_ft4:.2f}, TSH ист={true_tsh:.2f}, TSH шум={noisy_tsh:.2f}, доза={current_dose:.2f}")

    # ============================================================
    # Визуализация результатов
    # ============================================================
    days = np.arange(simulation_days)

    plt.figure(figsize=(14, 10))

    # График FT4
    plt.subplot(3, 1, 1)
    plt.plot(days, true_ft4_history, 'b-', linewidth=1.5, label='Истинный FT4')
    plt.plot(days, noisy_ft4_history, 'r--', alpha=0.7, label='Зашумлённый FT4 (измеренный)')
    plt.axhline(y=target_ft4, color='g', linestyle='--', label='Целевой FT4')
    plt.axhline(y=9.2, color='orange', linestyle=':', label='Нижняя граница эутиреоза')
    plt.axhline(y=16.0, color='orange', linestyle=':', label='Верхняя граница эутиреоза')
    plt.ylabel('FT4 (нг/л)')
    plt.legend(loc='upper right', fontsize=17)
    plt.grid(True, alpha=0.3)

    # График TSH
    plt.subplot(3, 1, 2)
    plt.plot(days, true_tsh_history, 'b-', linewidth=1.5, label='Истинный TSH')
    plt.plot(days, noisy_tsh_history, 'r--', alpha=0.7, label='Зашумлённый TSH (измеренный)')
    plt.axhline(y=0.46, color='orange', linestyle=':', label='Нижняя граница эутиреоза TSH')
    plt.axhline(y=5.19, color='orange', linestyle=':', label='Верхняя граница эутиреоза TSH')
    plt.ylabel('TSH (мЕд/л)')
    plt.legend(loc='upper right', fontsize=17)
    plt.grid(True, alpha=0.3)

    # График дозы
    plt.subplot(3, 1, 3)
    plt.plot(days, dose_history, 'k-', linewidth=1, label='Назначенная доза')
    plt.xlabel('Дни')
    plt.ylabel('Доза LT4 (мкг/сут)')
    plt.grid(True, alpha=0.3)

    plt.suptitle('Эксперимент №4: Шум измерений (MPC)')
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Анализ
    # ============================================================
    # Количество изменений дозы
    dose_changes = np.sum(np.abs(np.diff(dose_history)) > 0.1)
    print(f"\nКоличество изменений дозы за 12 месяцев: {dose_changes}")

    # Оценка отклонения FT4 в стационарном режиме (последние 3 месяца)
    steady_ft4 = true_ft4_history[-90:]
    mean_ft4 = np.mean(steady_ft4)
    std_ft4 = np.std(steady_ft4)
    print(f"Средний FT4 за последние 3 месяца: {mean_ft4:.2f} ± {std_ft4:.2f} нг/л")

    # Доля времени, когда FT4 выходил за границы эутиреоза (после первых 3 месяцев)
    in_euthyroid = np.sum((np.array(true_ft4_history[90:]) >= 9.2) & (np.array(true_ft4_history[90:]) <= 16.0))
    fraction_in = in_euthyroid / len(true_ft4_history[90:]) * 100
    print(f"Доля времени в эутиреозе (после 3-го месяца): {fraction_in:.1f}%")

    # Оценка влияния шума: вычислим среднеквадратичную ошибку (RMSE) между истинным и зашумлённым FT4
    rmse_ft4 = np.sqrt(np.mean((np.array(true_ft4_history) - np.array(noisy_ft4_history))**2))
    print(f"RMSE между истинным и зашумлённым FT4: {rmse_ft4:.3f} нг/л")