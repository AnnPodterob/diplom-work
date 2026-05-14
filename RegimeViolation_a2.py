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

# ================== 5. Эксперимент №3 с нарушением режима ==================
if __name__ == "__main__":
    # Создаём виртуального пациента
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        condition='moderate hypothyroidism'
    )

    # ---- Шаг 1: генерация данных для идентификации ----
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

    # ---- Шаг 4: симуляция MPC с нарушением режима ----
    # Параметры MPC
    Np = 14
    Nc = 7
    max_delta_dose = 12.5      # мкг/сут
    max_total_dose = 200.0
    target_ft4 = 13.7           # целевая FT4

    Q_weight = 5000.0           # увеличенный вес ошибки
    R_weight = 5000.0

    simulation_days = 360       # 12 месяцев по 30 дней
    # Сбросим пациента для новой симуляции
    patient = ThyrosimPatient(
        weight_kg=70,
        k_abs=0.80,
        SR4=0.714,
        SR3=0.051,
        condition='moderate hypothyroidism'
    )

    history_ft4 = []
    history_tsh = []
    history_dose = []
    history_dose_applied = []   # фактически принятая доза (может отличаться при нарушении)

    # Начальное состояние алгоритма
    initial_state = patient.state
    current_ft4 = initial_state[0] / 3.0
    x_k = np.array([[0.0], [current_ft4]])   # [x1, x2]
    current_dose = 50.0                       # мкг/сут

    print("\nЗапуск симуляции MPC с нарушением режима (пропуск таблеток на 4-м месяце)...")
    for day in range(simulation_days):
        # ---- Определяем фактическую дозу с учётом нарушения ----
        # Нарушение: на 4-м месяце (дни 90-120) пропускаем 3 дня подряд (дни 90-92)
        # Дни отсчитываются от 0
        if 90 <= day <= 92:
            dose_applied = 0.0
            disturbance_active = True
        else:
            dose_applied = current_dose
            disturbance_active = False

        # ---- Применяем дозу к пациенту ----
        state = patient.run_simulation(dose_applied, hours=24)
        real_ft4 = state[0] / 3.0
        real_tsh = state[6]

        # ---- MPC вычисляет новую дозу на основе измерений (без шума) ----
        # Обновляем внутреннее состояние алгоритма (x_k) по модели,
        # используя фактическую дозу, которая была применена.
        # Это необходимо для синхронизации состояния модели с реальностью.
        # Переводим дозу из мкг/сут в мкг/ч для модели
        dose_rate_applied = dose_applied / 24.0
        x_k = Ad @ x_k + Bd * dose_rate_applied
        # Корректируем x2 по измеренному FT4 (обратная связь по выходу)
        x_k[1, 0] = real_ft4

        # ---- Оптимизация MPC для вычисления приращения дозы на следующий день ----
        delta_d = cp.Variable((Nc, 1))
        cost = 0
        constraints = []
        x_curr = x_k.copy()

        for i in range(Np):
            idx = min(i, Nc - 1)
            # Прогнозная доза (мкг/сут)
            fut_dose = current_dose + cp.sum(delta_d[:idx+1])
            fut_dose_rate = fut_dose / 24.0   # мкг/ч

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

        # Обновляем дозу для следующего дня
        current_dose = np.clip(current_dose + applied_delta, 0, max_total_dose)

        # Сохраняем историю
        history_ft4.append(real_ft4)
        history_tsh.append(real_tsh)
        history_dose.append(current_dose)          # рекомендованная доза
        history_dose_applied.append(dose_applied)  # фактически принятая

        if day % 30 == 0:
            print(f"Месяц {day//30+1}, день {day}: FT4={real_ft4:.2f}, TSH={real_tsh:.2f}, доза рек={current_dose:.2f}, факт={dose_applied:.2f}")

    # ---- Визуализация результатов ----
    days = np.arange(simulation_days)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(days, history_dose, 'b-', label='Рекомендованная доза')
    plt.plot(days, history_dose_applied, 'r--', label='Фактически принятая доза')
    plt.axvspan(90, 93, alpha=0.3, color='orange', label='Пропуск таблеток')
    plt.ylabel('Доза LT4 (мкг/сут)')
    plt.legend(fontsize=17, loc='upper right')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(days, history_ft4, 'g-', label='FT4')
    plt.axhline(y=target_ft4, color='r', linestyle='--', label='Целевой FT4')
    plt.axhline(y=9.2, color='orange', linestyle=':', label='Нижняя граница эутиреоза')
    plt.axhline(y=16.0, color='orange', linestyle=':', label='Верхняя граница эутиреоза')
    plt.axvspan(90, 93, alpha=0.3, color='orange')
    plt.ylabel('FT4 (нг/л)')
    plt.legend(fontsize=17, loc='upper right')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(days, history_tsh, 'm-', label='TSH')
    plt.axhline(y=0.46, color='orange', linestyle=':', label='Нижняя граница эутиреоза TSH')
    plt.axhline(y=5.19, color='orange', linestyle=':', label='Верхняя граница эутиреоза TSH')
    plt.axvspan(90, 93, alpha=0.3, color='orange')
    plt.xlabel('Дни')
    plt.ylabel('TSH (мЕд/л)')
    plt.legend(fontsize=17, loc='upper right')
    plt.grid(True)

    plt.suptitle('Эксперимент №3: Нарушение режима (MPC)')
    plt.tight_layout()
    plt.show()

    # ---- Анализ реакции системы ----
    ft4_array = np.array(history_ft4)
    tsh_array = np.array(history_tsh)

    # Минимум FT4 после нарушения (начиная с дня 93)
    min_ft4 = np.min(ft4_array[93:])
    min_idx = np.argmin(ft4_array[93:]) + 93
    print(f"\nМинимальный FT4 после нарушения: {min_ft4:.2f} нг/л на день {min_idx+1}")

    # Максимальный FT4 после восстановления приёма
    max_ft4 = np.max(ft4_array[min_idx:])
    print(f"Максимальный FT4 после восстановления: {max_ft4:.2f} нг/л")

    if max_ft4 > 16.0:
        print("ВНИМАНИЕ: Зафиксирован перелёт в гипертиреоз!")
    else:
        print("Перелёт в гипертиреоз не произошёл.")

    # Время достижения целевого диапазона после нарушения
    # (момент, когда FT4 снова входит в [target_ft4 - 1, target_ft4 + 1])
    in_range = False
    recovery_day = None
    for i in range(min_idx, len(ft4_array)):
        if target_ft4 - 1 <= ft4_array[i] <= target_ft4 + 1:
            recovery_day = i
            break
    if recovery_day is not None:
        print(f"Время восстановления после нарушения: {recovery_day - min_idx} дней")
    else:
        print("FT4 не вернулся в целевой диапазон за время симуляции.")