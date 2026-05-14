import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Параметры модели (согласно Таблице 1 и тексту статьи) ---
k_exc = 0.26 / 24  # Скорость выведения (преобразовано в 1/час) [cite: 1087]
k_sec = 0.01 / 24  # Скорость элиминации FT4 (~10 дней полураспада) [cite: 977]
k_reac = 7.25e-6  # Влияние принятой дозы (ppm) [cite: 1017]
v_max = 2.19e-4  # Макс. скорость эндогенной выработки (мкмоль/ч) [cite: 1017]
k_m = 9.504e3  # Константа Михаэлиса [cite: 1017]

# Упрощение: для Hashimoto принимаем TSH стабильно высоким на старте
# В реальной модели TSH меняется по принципу обратной связи,
# здесь мы фокусируемся на алгоритме подбора дозы LT4.
TSH_fixed = 5.0  # mU/L

# --- Параметры регулятора ---
Td = 30 * 24  # Период дискретизации (30 дней в часах) [cite: 674]
max_dose_step = 12.5  # Максимальный шаг изменения дозы (мкг) [cite: 1102]
max_dose = 200.0  # Максимальная суточная доза (мкг)


def model_dynamics(x, t, dose):
    """ Динамика системы (уравнения 1a и 1b) """
    x1, x2 = x
    dx1dt = dose - k_exc * x1
    # Эндогенная часть + влияние дозы - естественный распад
    dx2dt = (v_max * TSH_fixed) / (k_m + TSH_fixed) + k_reac * x1 - k_sec * x2
    return [dx1dt, dx2dt]


# --- Симуляция ---
months = 12
time_steps = months * 30 * 24  # Общее время в часах
t_eval = np.linspace(0, time_steps, time_steps)

# Начальные условия (пациент с гипотиреозом)
x_current = [0.0, 8.0]  # Начальное FT4 = 8 ng/L (ниже нормы) [cite: 891]
target_ft4 = 13.0  # Целевое значение FT4 [cite: 715]

history_ft4 = []
history_dose = []
current_dose = 0.0

for month in range(months):
    # 1. "Измерение" FT4 в начале месяца (визит к врачу) [cite: 1109]
    measured_ft4 = x_current[1]

    # 2. Алгоритм управления (упрощенный пропорциональный регулятор с логикой статьи)
    error = target_ft4 - measured_ft4

    # Расчет необходимого изменения (в статье используется передаточная функция R(z))
    # Здесь реализована ключевая особенность: инкрементальное изменение [cite: 1102]
    if error > 0.5:
        suggested_change = max_dose_step
    elif error < -0.5:
        suggested_change = -max_dose_step
    else:
        suggested_change = 0

    current_dose += suggested_change

    # 3. Ограничения (Saturator)
    current_dose = np.clip(current_dose, 0, max_dose)

    # 4. Моделирование месяца жизни пациента на этой дозе
    t_month = np.linspace(0, Td, Td)
    sol = odeint(model_dynamics, x_current, t_month, args=(current_dose / 24,))  # доза в час

    # Сохранение истории
    history_ft4.extend(sol[:, 1])
    history_dose.extend([current_dose] * Td)

    # Обновление состояния для следующего месяца
    x_current = sol[-1]

# --- Визуализация результатов ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(np.array(history_ft4), label='Текущий FT4 (ng/L)', color='blue')
plt.axhline(y=target_ft4, color='red', linestyle='--', label='Цель (Target)')
plt.axhline(y=9.2, color='green', linestyle=':', label='Норма (Euthyroid range)')
plt.axhline(y=16.0, color='green', linestyle=':')
plt.ylabel('FT4 Concentration (ng/L)')
plt.title('Моделирование алгоритма дозирования левотироксина (за 12 месяцев)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.step(range(len(history_dose)), history_dose, where='post', color='black', label='Доза LT4 (мкг/день)')
plt.ylabel('Daily Dosage (µg)')
plt.xlabel('Время (часы)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()