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
    ft4_rec[0] = state[0] / 3.0               # приводим к примерному FT4
    tsh_rec[0] = state[6]
    dose_rec[0] = 0.0

    for i in range(n_steps):
        day = i // 24
        dose = daily_doses[day] / 24.0         # мкг/ч
        # Интегрируем один час с постоянной дозой
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
            x0 = [0.0, tsh_array[0]]   # грубое начальное приближение
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
        return x[:, 1]   # FT4

# ================== 3. Функция потерь и оптимизация ==================
def loss_func(params, t, d_meas, tsh_meas, ft4_meas, k_exc_fixed, k_sec_fixed):
    v_max, k_m, k_reac = params
    model = ThyroidModel(k_exc_fixed, k_sec_fixed, v_max, k_m, k_reac)
    ft4_sim = model.simulate(t, d_meas, tsh_meas, x0=[0.0, ft4_meas[0]])
    return ft4_sim - ft4_meas

def estimate_parameters(t, d_meas, tsh_meas, ft4_meas):
    # Фиксированные параметры из статьи (получены из матриц Ad)
    k_exc_fixed = 0.171      # 1/ч
    k_sec_fixed = 0.00438    # 1/ч

    # Начальные предположения
    p0 = [2.0, 5.0, 0.3]     # v_max, k_m, k_reac
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
    """
    Дискретизация линейной части системы (без TSH) с шагом Ts часов.
    Возвращает Ad, Bd (для входа в мкг/ч).
    """
    # Непрерывная система: dx/dt = Ac x + Bc u, y = Cc x
    Ac = np.array([[-k_exc, 0],
                   [k_reac, -k_sec]])
    Bc = np.array([[1],
                   [0]])
    Cc = np.array([[0, 1]])
    Dc = np.array([[0]])

    sys_d = cont2discrete((Ac, Bc, Cc, Dc), Ts, method='zoh')
    Ad, Bd, Cd, Dd, _ = sys_d
    return Ad, Bd, Cd

# ================== 5. Основной блок ==================
if __name__ == "__main__":
    # Создаём виртуального пациента
    # 6. Пациент с индивидуальными вариациями метаболизма (например, ускоренная элиминация T4)
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        k_ex4=0.018,  # +50% от базового 0.012
        k_ex3=0.052,  # +50% от 0.035
        condition='hypothyroidism, fast metabolism'
    )

    # ---- Шаг 1: генерация данных ----
    print("Генерация обучающих данных...")
    t, ft4_meas, tsh_meas, d_meas, daily_doses = generate_identification_data(patient, days=60, seed=123)

    # ---- Шаг 2: идентификация параметров ----
    print("Идентификация параметров...")
    k_exc, k_sec, k_reac = estimate_parameters(t, d_meas, tsh_meas, ft4_meas)

    # ---- Шаг 3: построение матриц MPC ----
    Ad, Bd, Cd = build_mpc_matrices(k_exc, k_sec, k_reac, Ts=24.0)
    print("Матрицы модели MPC:")
    print("Ad =\n", Ad)
    print("Bd =\n", Bd)

    # ---- Шаг 4: симуляция MPC с новыми матрицами ----
    # Параметры MPC
    Np = 14
    Nc = 7
    max_delta_dose = 12.5      # мкг/сут
    max_total_dose = 200.0
    target_ft4 = 13.7           # целевая FT4

    Q_weight = 5000.0           # увеличенный вес ошибки
    R_weight = 500.0

    simulation_days = 60
    # Сбросим пациента для новой симуляции
    # 6. Пациент с индивидуальными вариациями метаболизма (например, ускоренная элиминация T4)
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        k_ex4=0.018,  # +50% от базового 0.012
        k_ex3=0.052,  # +50% от 0.035
        condition='hypothyroidism, fast metabolism'
    )

    history_ft4 = []
    history_tsh = []
    history_dose = []

    # Начальное состояние алгоритма
    initial_state = patient.state
    current_ft4 = initial_state[0] / 3.0
    x_k = np.array([[0.0], [current_ft4]])   # [x1, x2]
    current_dose = 50.0                       # мкг/сут

    print("\nЗапуск симуляции MPC с калиброванной моделью...")
    for day in range(simulation_days):
        # ---- Оптимизация MPC ----
        delta_d = cp.Variable((Nc, 1))
        cost = 0
        constraints = []
        x_curr = x_k

        for i in range(Np):
            idx = min(i, Nc - 1)
            # Прогнозная доза (мкг/сут) -> переводим в мкг/ч для модели
            fut_dose_ug_per_day = current_dose + cp.sum(delta_d[:idx + 1])
            fut_dose_ug_per_hour = fut_dose_ug_per_day / 24.0

            # Прогноз состояния
            x_curr = Ad @ x_curr + Bd * fut_dose_ug_per_hour
            y_curr = Cd @ x_curr

            cost += Q_weight * cp.square(y_curr - target_ft4)
            if i < Nc:
                cost += R_weight * cp.square(delta_d[i])
                constraints += [cp.abs(delta_d[i]) <= max_delta_dose]
                constraints += [current_dose + cp.sum(delta_d[:i + 1]) <= max_total_dose]
                constraints += [current_dose + cp.sum(delta_d[:i + 1]) >= 0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            applied_delta = 0.0
        else:
            applied_delta = delta_d.value[0, 0]

        # ---- Применяем дозу к пациенту ----
        current_dose = np.clip(current_dose + applied_delta, 0, max_total_dose)
        state = patient.run_simulation(current_dose, hours=24)
        real_ft4 = state[0] / 3.0
        real_tsh = state[6]

        # ---- Обновляем внутреннее состояние алгоритма ----
        # Синхронизируем x2 с измеренным FT4
        x_k[1, 0] = real_ft4
        # Обновляем x1 по модели с фактической дозой (в мкг/ч)
        x_k[0, 0] = Ad[0, 0] * x_k[0, 0] + Bd[0, 0] * (current_dose / 24.0)

        history_ft4.append(real_ft4)
        history_tsh.append(real_tsh)
        history_dose.append(current_dose)

        print(f"День {day}: FT4={real_ft4:.2f}, Доза={current_dose:.2f}")

    # ---- Визуализация результатов ----
    plt.figure(figsize=(12, 8))
    plt.subplot(3,1,1)
    plt.plot(history_dose, label='Доза LT4 (мкг/сут)')
    plt.axhline(target_ft4, color='r', linestyle='--', label='Целевая FT4')
    plt.legend(fontsize=20)
    plt.subplot(3,1,2)
    plt.plot(history_ft4, label='FT4 (нг/л)')
    plt.legend(fontsize=20)
    plt.subplot(3,1,3)
    plt.plot(history_tsh, label='TSH (мЕд/л)')
    plt.legend(fontsize=20)
    plt.xlabel('Дни')
    plt.show()