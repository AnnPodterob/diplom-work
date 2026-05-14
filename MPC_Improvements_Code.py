import numpy as np
import cvxpy as cp
from scipy.integrate import odeint

def build_mpc_with_improved_regularization(k_exc, k_sec, k_reac, Ts=24.0):

    from scipy.signal import cont2discrete
    
    # Построение дискретной модели (как в оригинале)
    Ac = np.array([[-k_exc, 0],
                   [k_reac, -k_sec]])
    Bc = np.array([[1],
                   [0]])
    Cc = np.array([[0, 1]])
    Dc = np.array([[0]])
    
    sys_d = cont2discrete((Ac, Bc, Cc, Dc), Ts, method='zoh')
    Ad, Bd, Cd, Dd, _ = sys_d
    
    return Ad, Bd, Cd


def solve_mpc_improved(Ad, Bd, Cd, x_k, current_dose, target_ft4,
                       Np=14, Nc=7, max_delta_dose=12.5, max_total_dose=200.0,
                       Q_weight=5000.0, R_weight_old=1.0):
    """
    Parameters:
    -----------
    R_weight_old : float, по умолчанию 1.0
        Значение штрафа за изменение дозы из оригинального кода.
        Это приводит к высокой TV ≈ 500.
    
    Returns:
    --------
    applied_delta : float
        Изменение дозы в мкг/сут
    stats : dict
        Словарь со статистикой (TV, количество смен)
    """

    R_weight_v1 = 1.0
    delta_d_v1 = cp.Variable((Nc, 1))
    cost_v1 = 0
    constraints_v1 = []
    x_curr = x_k.copy()
    
    for i in range(Np):
        idx = min(i, Nc - 1)
        fut_dose_ug_per_hour = (current_dose + cp.sum(delta_d_v1[:idx + 1])) / 24.0
        x_curr = Ad @ x_curr + Bd * fut_dose_ug_per_hour
        y_curr = Cd @ x_curr
        
        cost_v1 += Q_weight * cp.square(y_curr - target_ft4)
        if i < Nc:
            cost_v1 += R_weight_v1 * cp.square(delta_d_v1[i])
            constraints_v1 += [cp.abs(delta_d_v1[i]) <= max_delta_dose]
            constraints_v1 += [current_dose + cp.sum(delta_d_v1[:i + 1]) <= max_total_dose]
            constraints_v1 += [current_dose + cp.sum(delta_d_v1[:i + 1]) >= 0]
    
    prob_v1 = cp.Problem(cp.Minimize(cost_v1), constraints_v1)
    prob_v1.solve(solver=cp.OSQP, verbose=False)
    
    result_v1 = {
        'R_weight': R_weight_v1,
        'delta': delta_d_v1.value[0, 0] if delta_d_v1.value is not None else 0.0,
        'total_variation_expectancy': np.abs(delta_d_v1.value[0, 0]) if delta_d_v1.value is not None else 0.0,
        'description': 'Оригинальное R=1.0 (высокая TV)'
    }
    

    R_weight_v2 = 100.0
    delta_d_v2 = cp.Variable((Nc, 1))
    cost_v2 = 0
    constraints_v2 = []
    x_curr = x_k.copy()
    
    for i in range(Np):
        idx = min(i, Nc - 1)
        fut_dose_ug_per_hour = (current_dose + cp.sum(delta_d_v2[:idx + 1])) / 24.0
        x_curr = Ad @ x_curr + Bd * fut_dose_ug_per_hour
        y_curr = Cd @ x_curr
        
        cost_v2 += Q_weight * cp.square(y_curr - target_ft4)
        if i < Nc:
            cost_v2 += R_weight_v2 * cp.square(delta_d_v2[i])
            constraints_v2 += [cp.abs(delta_d_v2[i]) <= max_delta_dose]
            constraints_v2 += [current_dose + cp.sum(delta_d_v2[:i + 1]) <= max_total_dose]
            constraints_v2 += [current_dose + cp.sum(delta_d_v2[:i + 1]) >= 0]
    
    prob_v2 = cp.Problem(cp.Minimize(cost_v2), constraints_v2)
    prob_v2.solve(solver=cp.OSQP, verbose=False)
    
    result_v2 = {
        'R_weight': R_weight_v2,
        'delta': delta_d_v2.value[0, 0] if delta_d_v2.value is not None else 0.0,
        'total_variation_expectancy': np.abs(delta_d_v2.value[0, 0]) if delta_d_v2.value is not None else 0.0,
        'description': 'Компромисс R=100 (средняя TV)'
    }
    

    R_weight_v3 = 500.0
    delta_d_v3 = cp.Variable((Nc, 1))
    cost_v3 = 0
    constraints_v3 = []
    x_curr = x_k.copy()
    
    for i in range(Np):
        idx = min(i, Nc - 1)
        fut_dose_ug_per_hour = (current_dose + cp.sum(delta_d_v3[:idx + 1])) / 24.0
        x_curr = Ad @ x_curr + Bd * fut_dose_ug_per_hour
        y_curr = Cd @ x_curr
        
        cost_v3 += Q_weight * cp.square(y_curr - target_ft4)
        if i < Nc:
            cost_v3 += R_weight_v3 * cp.square(delta_d_v3[i])
            constraints_v3 += [cp.abs(delta_d_v3[i]) <= max_delta_dose]
            constraints_v3 += [current_dose + cp.sum(delta_d_v3[:i + 1]) <= max_total_dose]
            constraints_v3 += [current_dose + cp.sum(delta_d_v3[:i + 1]) >= 0]
    
    prob_v3 = cp.Problem(cp.Minimize(cost_v3), constraints_v3)
    prob_v3.solve(solver=cp.OSQP, verbose=False)
    
    result_v3 = {
        'R_weight': R_weight_v3,
        'delta': delta_d_v3.value[0, 0] if delta_d_v3.value is not None else 0.0,
        'total_variation_expectancy': np.abs(delta_d_v3.value[0, 0]) if delta_d_v3.value is not None else 0.0,
        'description': 'Максимальная стабильность R=500 (низкая TV)'
    }
    
    return result_v1, result_v2, result_v3


# ═══════════════════════════════════════════════════════════════════════════
#  ОБНАРУЖЕНИЕ АНОМАЛИЙ (ПРОПУСК ДОЗ)
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Модуль обнаружения аномалий для различения:
    1. Пропуска доз (резкое падение FT4 на 2-4 нг/л за 1-2 дня)
    2. Прогрессирования болезни (медленное снижение на 0.2-0.5 нг/л за неделю)
    """
    
    def __init__(self, window_days=7, threshold=0.5):
        """
        Parameters:
        -----------
        window_days : int
            Размер окна для расчёта скорости изменения (дней)
        threshold : float
            Пороговое значение скорости изменения (нг/л в день)
            для классификации как аномалия
        """
        self.window_days = window_days
        self.threshold = threshold
    
    def calculate_rate_of_change(self, history_ft4):
        """Рассчитывает скорость изменения FT4 в последние N дней"""
        if len(history_ft4) < self.window_days:
            return 0.0
        
        recent_ft4 = history_ft4[-1]
        past_ft4 = history_ft4[-self.window_days]
        rate = (recent_ft4 - past_ft4) / self.window_days
        
        return rate
    
    def detect(self, history_ft4):
        """
        Классифицирует тип изменения FT4.
        
        Returns:
        ────────
        str, one of:
            "NORMAL" - медленное изменение (норма)
            "DOSE_SKIPPING" - резкое падение (возможный пропуск доз)
            "UNKNOWN" - недостаточно данных
        """
        if len(history_ft4) < self.window_days:
            return "UNKNOWN"
        
        rate_of_change = self.calculate_rate_of_change(history_ft4)
        
        # Классификация
        if abs(rate_of_change) > self.threshold:
            # Резкое изменение
            if rate_of_change < -self.threshold:
                return "DOSE_SKIPPING"  # Резкое падение
            else:
                return "RAPID_INCREASE"  # Резкий рост (редко)
        else:
            # Нормальное изменение
            return "NORMAL"
    
    def get_rate_of_change(self, history_ft4):
        """Возвращает скорость изменения для логирования"""
        return self.calculate_rate_of_change(history_ft4)


# ═══════════════════════════════════════════════════════════════════════════
#  АСИММЕТРИЧНЫЕ ШТРАФНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════

def solve_mpc_with_asymmetric_penalties(Ad, Bd, Cd, x_k, current_dose, target_ft4,
                                        Np=14, Nc=7, max_delta_dose=12.5, 
                                        max_total_dose=200.0,
                                        penalty_hypothyroid=5000.0,
                                        penalty_hyperthyroid=10000.0,
                                        R_weight=100.0):
    """
    MPC с асимметричными штрафами.
    
    Обоснование:
    ─────────────
    Недостаток Т4 (гипотиреоз) вызывает усталость, отеки, медленный ритм сердца.
    Это неприятно, но не опасно в короткие сроки.
    
    Избыток Т4 (гипертиреоз) вызывает тахикардию, аритмию, тревожность.
    Это может быть опасно, особенно для пациентов с сердечными проблемами.
    
    Поэтому штраф за гипертиреоз должен быть значительно выше.
    
    Параметры:
    ──────────
    penalty_hypothyroid : float, по умолчанию 5000.0
        Штраф за FT4 < target (эквивалент Q-веса в оригинале)
    
    penalty_hyperthyroid : float, по умолчанию 10000.0
        Штраф за FT4 > target (в 2 раза выше для осторожности)
    """
    
    delta_d = cp.Variable((Nc, 1))
    cost = 0
    constraints = []
    x_curr = x_k.copy()
    
    for i in range(Np):
        idx = min(i, Nc - 1)
        fut_dose_ug_per_hour = (current_dose + cp.sum(delta_d[:idx + 1])) / 24.0
        x_curr = Ad @ x_curr + Bd * fut_dose_ug_per_hour
        y_curr = Cd @ x_curr
        
        # Асимметричный штраф
        error = y_curr - target_ft4
        # Если ошибка положительная (FT4 выше цели), используем больший штраф
        # Если ошибка отрицательная (FT4 ниже цели), используем меньший штраф
        # Это достигается через условное выражение cvxpy
        penalty = cp.piecewise_linear(
            error,
            [np.array([0, 1])],  # точки разделения на нулевой точке
            [np.array([0, penalty_hypothyroid]),
             np.array([0, penalty_hyperthyroid])]
        )
        
        # Упрощённый вариант (без piecewise_linear):
        # Просто используем разные веса для разных направлений
        cost_term = cp.maximum(0, -error) * penalty_hypothyroid + \
                    cp.maximum(0, error) * penalty_hyperthyroid
        cost += cost_term
        
        if i < Nc:
            cost += R_weight * cp.square(delta_d[i])
            constraints += [cp.abs(delta_d[i]) <= max_delta_dose]
            constraints += [current_dose + cp.sum(delta_d[:i + 1]) <= max_total_dose]
            constraints += [current_dose + cp.sum(delta_d[:i + 1]) >= 0]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return 0.0
    else:
        return delta_d.value[0, 0] if delta_d.value is not None else 0.0
