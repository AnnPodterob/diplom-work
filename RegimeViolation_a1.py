import numpy as np
import matplotlib.pyplot as plt
from thyrosim_patient2 import ThyrosimPatient

# ============================================================
# Эксперимент №3: Нарушение режима (Adherence Test)
# ============================================================

# Параметры эксперимента
target_ft4 = 13.7
step_dose = 12.5
months = 12

# Создание пациента
patient = ThyrosimPatient(
    weight_kg=70,
    k_abs=0.80,
    SR4=0.714,
    SR3=0.051,
    condition='moderate hypothyroidism'
)

current_dose = 0.0
history = []                     # список для хранения ежедневных данных
daily_data = []                  # для удобства построения графиков

print(f"{'Месяц':<8} | {'FT4':<8} | {'TSH':<8} | {'Доза (мкг)':<10} | {'Примечание'}")
print("-" * 70)

for m in range(months):
    # Замер перед коррекцией (значения в начале месяца)
    ft4_before = patient.state[0] / 3.0
    tsh_before = patient.state[6]

    # Логика алгоритма (простой контроллер с шагом)
    if ft4_before < target_ft4 - 0.5:
        current_dose += step_dose
    elif ft4_before > target_ft4 + 0.5:
        current_dose -= step_dose
    current_dose = np.clip(current_dose, 0, 200)

    # Информация о возмущении для вывода
    disturbance_msg = ""

    # Симуляция месяца (30 дней по 24 часа)
    for day in range(30):
        dose_today = current_dose

        # ========== Нарушение режима на 4-м месяце (индекс 3) ==========
        # Вариант 1: пропуск таблеток в течение 3 дней (например, дни 10–12)
        if m == 3 and 10 <= day <= 12:
            dose_today = 0.0
            disturbance_msg = "Пропуск таблеток (дни 10-12)"
        # Вариант 2: временное снижение всасываемости (можно раскомментировать вместо пропуска)
        # if m == 3 and 10 <= day <= 12:
        #     patient.k_abs = 0.1          # резкое снижение всасывания
        # else:
        #     patient.k_abs = 0.8
        # ===============================================================

        # Симуляция одного дня
        state = patient.run_simulation(dose_today, hours=24)

        # Сохраняем ежедневные данные
        ft4_daily = state[0] / 3.0
        tsh_daily = state[6]
        daily_data.append({
            'month': m,
            'day': day,
            'ft4': ft4_daily,
            'tsh': tsh_daily,
            'dose': dose_today
        })

        # Сохраняем для истории (опционально)
        history.append({
            'ft4': ft4_daily,
            'tsh': tsh_daily,
            'dose': dose_today
        })

    # После завершения месяца выводим информацию о коррекции дозы
    print(f"{m+1:<8} | {ft4_before:<8.2f} | {tsh_before:<8.2f} | {current_dose:<10.2f} | {disturbance_msg}")

# ============================================================
# Визуализация результатов
# ============================================================

# Преобразование данных для построения графиков
days = np.arange(1, len(daily_data) + 1)
ft4_vals = [d['ft4'] for d in daily_data]
tsh_vals = [d['tsh'] for d in daily_data]
dose_vals = [d['dose'] for d in daily_data]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# График FT4
ax1.plot(days, ft4_vals, 'b-', linewidth=1.5, label='FT4')
ax1.axhline(y=target_ft4, color='g', linestyle='--', label='Целевой FT4')
ax1.axhline(y=9.2, color='r', linestyle=':', alpha=0.7, label='Нижняя граница эутиреоза (9.2)')
ax1.axhline(y=16.0, color='r', linestyle=':', alpha=0.7, label='Верхняя граница эутиреоза (16.0)')
# Область нарушения режима (4-й месяц, дни 10-12)
ax1.axvspan(90+10, 90+12, alpha=0.3, color='orange', label='Пропуск доз')
ax1.set_ylabel('FT4 (нг/л)')
ax1.legend(loc='upper right', fontsize=17)
ax1.grid(True, alpha=0.3)

# График TSH
ax2.plot(days, tsh_vals, 'r-', linewidth=1.5, label='TSH')
ax2.axhline(y=0.46, color='g', linestyle='--', label='Нижняя граница эутиреоза (0.46)')
ax2.axhline(y=5.19, color='g', linestyle='--', label='Верхняя граница эутиреоза (5.19)')
ax2.axvspan(90+10, 90+12, alpha=0.3, color='orange')
ax2.set_ylabel('TSH (мЕд/л)')
ax2.legend(loc='best', fontsize=17)
ax2.grid(True, alpha=0.3)

# График дозы
ax3.step(days, dose_vals, 'k-', where='post', linewidth=1, label='Назначенная доза')
ax3.axvspan(90+10, 90+12, alpha=0.3, color='orange')
ax3.set_xlabel('День')
ax3.set_ylabel('Доза LT4 (мкг/сут)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Анализ реакции системы
# ============================================================
# Находим минимальный FT4 после нарушения
ft4_array = np.array(ft4_vals)
min_ft4 = np.min(ft4_array[90:])   # после 3-го месяца
min_idx = np.argmin(ft4_array[90:]) + 90
print(f"\nМинимальный FT4 после нарушения: {min_ft4:.2f} нг/л на день {min_idx+1}")

# Максимальный FT4 после восстановления приёма
max_ft4 = np.max(ft4_array[min_idx:])
print(f"Максимальный FT4 после восстановления: {max_ft4:.2f} нг/л")

# Проверка на гипертиреоз
if max_ft4 > 16.0:
    print("ВНИМАНИЕ: Зафиксирован перелёт в гипертиреоз!")
else:
    print("Перелёт в гипертиреоз не произошёл.")