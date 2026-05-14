import numpy as np
import matplotlib.pyplot as plt
from thyrosim_patient2 import ThyrosimPatient

# Параметры эксперимента
target_ft4 = 13.7
step_dose = 12.5
months = 12

# 1. Лёгкий гипотиреоз (секреция 30% от нормы)
patient = ThyrosimPatient(
    weight_kg=70,
    k_abs=0.80,
    SR4=1.26,          # 30% от 4.2
    SR3=0.09,          # 30% от 0.3
    condition='mild hypothyroidism'
)

current_dose = 0.0
history = []
history_dose = []
history_ft4 = []
history_tsh  =[]


print(f"{'Месяц':<8} | {'FT4':<8} | {'TSH':<8} | {'Доза (мкг)':<10}")
print("-" * 45)

for m in range(months):
    # Замер перед коррекцией
    ft4 = patient.state[0] / 3.0
    tsh = patient.state[6]

    # Логика алгоритма
    if ft4 < target_ft4 - 0.5:
        current_dose += step_dose
    elif ft4 > target_ft4 + 0.5:
        current_dose -= step_dose

    current_dose = np.clip(current_dose, 0, 200)
    print(f"{m + 1:<8} | {ft4:<8.2f} | {tsh:<8.2f} | {current_dose:<10.2f}")

    # Симуляция месяца (30 дней по 24 часа)
    for _ in range(30):
        state = patient.run_simulation(current_dose, hours=24)
        history.append({'ft4': state[0] / 3.0, 'tsh': state[6], 'dose': current_dose})
        history_dose.append(current_dose)
        history_tsh.append(state[6])
        history_ft4.append(state[0]/3.0)

# Сохранение результатов для общего сравнения (опционально)
# np.save('results_alg1.npy', history)

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