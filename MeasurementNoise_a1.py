import numpy as np
import matplotlib.pyplot as plt
from thyrosim_patient2 import ThyrosimPatient

# ============================================================
# Эксперимент №4: Шум измерений (для первой модели - простой алгоритм)
# ============================================================

# Параметры эксперимента
target_ft4 = 13.7
step_dose = 12.5
months = 12

# Создание пациента с лёгким гипотиреозом
patient = ThyrosimPatient(
    weight_kg=70,
    k_abs=0.80,
    SR4=0.714,
    SR3=0.051,
    condition='moderate hypothyroidism'
)

# Параметры шума
noise_ft4_percent = 0.07   # 7% относительный шум для FT4
noise_tsh_percent = 0.05   # 5% относительный шум для TSH
np.random.seed(42)         # для воспроизводимости

def add_noise(value, percent):
    """Добавляет относительный шум к значению."""
    noise = np.random.normal(0, percent * value)
    return max(0, value + noise)   # исключаем отрицательные концентрации

current_dose = 0.0
history = []               # ежедневные данные (истинные)
monthly_data = []          # данные на начало месяца (истинные и зашумлённые)

print(f"{'Месяц':<8} | {'FT4 ист':<8} | {'FT4 шум':<8} | {'TSH ист':<8} | {'TSH шум':<8} | {'Доза':<10}")
print("-" * 75)

for m in range(months):
    # Получаем истинные значения в начале месяца (перед коррекцией)
    ft4_true = patient.state[0] / 3.0
    tsh_true = patient.state[6]

    # Добавляем шум к измерениям (имитация лабораторной погрешности)
    ft4_noisy = add_noise(ft4_true, noise_ft4_percent)
    tsh_noisy = add_noise(tsh_true, noise_tsh_percent)

    # Логика алгоритма (простой контроллер с шагом) использует зашумлённые значения
    if ft4_noisy < target_ft4 - 0.5:
        current_dose += step_dose
    elif ft4_noisy > target_ft4 + 0.5:
        current_dose -= step_dose

    current_dose = np.clip(current_dose, 0, 200)

    # Сохраняем данные месяца
    monthly_data.append({
        'month': m+1,
        'ft4_true': ft4_true,
        'ft4_noisy': ft4_noisy,
        'tsh_true': tsh_true,
        'tsh_noisy': tsh_noisy,
        'dose': current_dose
    })

    print(f"{m+1:<8} | {ft4_true:<8.2f} | {ft4_noisy:<8.2f} | {tsh_true:<8.2f} | {tsh_noisy:<8.2f} | {current_dose:<10.2f}")

    # Симуляция месяца (30 дней по 24 часа) с постоянной дозой
    for day in range(30):
        state = patient.run_simulation(current_dose, hours=24)
        ft4_daily = state[0] / 3.0
        tsh_daily = state[6]
        history.append({
            'day': m*30 + day + 1,
            'ft4': ft4_daily,
            'tsh': tsh_daily,
            'dose': current_dose
        })

# ============================================================
# Визуализация результатов
# ============================================================
days = [d['day'] for d in history]
ft4_vals = [d['ft4'] for d in history]
tsh_vals = [d['tsh'] for d in history]
dose_vals = [d['dose'] for d in history]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# График FT4
ax1.plot(days, ft4_vals, 'b-', linewidth=1.5, label='Истинный FT4')
# Отметим точки измерений в начале каждого месяца (истинные и зашумлённые)
month_starts = [d['day'] for d in history if (d['day']-1) % 30 == 0]
ft4_true_monthly = [d['ft4_true'] for d in monthly_data]
ft4_noisy_monthly = [d['ft4_noisy'] for d in monthly_data]
ax1.scatter(month_starts, ft4_true_monthly, color='green', marker='o', s=50, label='Измеренный FT4 (истинный)', zorder=5)
ax1.scatter(month_starts, ft4_noisy_monthly, color='red', marker='x', s=70, label='Измеренный FT4 (с шумом)', zorder=5)
ax1.axhline(y=target_ft4, color='g', linestyle='--', label='Целевой FT4')
ax1.axhline(y=9.2, color='r', linestyle=':', alpha=0.7, label='Нижняя граница эутиреоза')
ax1.axhline(y=16.0, color='r', linestyle=':', alpha=0.7, label='Верхняя граница эутиреоза')
ax1.set_ylabel('FT4 (нг/л)')
ax1.legend(loc='upper right', fontsize=17)
ax1.grid(True, alpha=0.3)

# График TSH
ax2.plot(days, tsh_vals, 'r-', linewidth=1.5, label='Истинный TSH')
tsh_true_monthly = [d['tsh_true'] for d in monthly_data]
tsh_noisy_monthly = [d['tsh_noisy'] for d in monthly_data]
ax2.scatter(month_starts, tsh_true_monthly, color='green', marker='o', s=50, label='Измеренный TSH (истинный)', zorder=5)
ax2.scatter(month_starts, tsh_noisy_monthly, color='red', marker='x', s=70, label='Измеренный TSH (с шумом)', zorder=5)
ax2.axhline(y=0.46, color='g', linestyle='--', label='Нижняя граница эутиреоза TSH')
ax2.axhline(y=5.19, color='g', linestyle='--', label='Верхняя граница эутиреоза TSH')
ax2.set_ylabel('TSH (мЕд/л)')
ax2.legend(loc='upper right', fontsize=17)
ax2.grid(True, alpha=0.3)

# График дозы
ax3.step(days, dose_vals, 'k-', where='post', linewidth=1, label='Назначенная доза')
ax3.set_xlabel('День')
ax3.set_ylabel('Доза LT4 (мкг/сут)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Анализ стабильности дозы
# ============================================================
# Количество изменений дозы
dose_changes = np.sum(np.abs(np.diff(dose_vals)) > 0.1)
print(f"\nКоличество изменений дозы за 12 месяцев: {dose_changes}")

# Если бы шума не было, доза изменялась бы только когда FT4 выходит за пределы ±0.5 от target.
# Здесь можно оценить, как шум заставил алгоритм лишний раз менять дозу.
# Для этого можно провести дополнительную симуляцию без шума и сравнить.

# Анализ: находим моменты, когда доза изменилась, и смотрим, было ли это вызвано шумом
# (т.е. истинный FT4 был в пределах допуска, но зашумлённый – нет)
unnecessary_changes = 0
for i in range(1, len(monthly_data)):
    # Получаем данные предыдущего и текущего месяца
    prev_dose = monthly_data[i-1]['dose']
    curr_dose = monthly_data[i]['dose']
    if curr_dose != prev_dose:
        ft4_true_prev = monthly_data[i-1]['ft4_true']
        ft4_noisy_prev = monthly_data[i-1]['ft4_noisy']
        # Проверяем, было ли изменение из-за шума (истинный FT4 был в норме, а зашумлённый нет)
        if (target_ft4 - 0.5 <= ft4_true_prev <= target_ft4 + 0.5) and \
           (ft4_noisy_prev < target_ft4 - 0.5 or ft4_noisy_prev > target_ft4 + 0.5):
            unnecessary_changes += 1
print(f"Изменения дозы, вызванные шумом (истинный FT4 в норме): {unnecessary_changes}")

# Оценка колебаний FT4 вокруг целевого уровня
ft4_array = np.array(ft4_vals)
mean_ft4 = np.mean(ft4_array[-90:])  # последние 3 месяца
std_ft4 = np.std(ft4_array[-90:])
print(f"Средний FT4 за последние 3 месяца: {mean_ft4:.2f} ± {std_ft4:.2f} нг/л")