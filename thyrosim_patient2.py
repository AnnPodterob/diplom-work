import numpy as np
from scipy.integrate import odeint

class ThyrosimPatient:
    def __init__(self, weight_kg=70,
                 k_abs=0.8,          # доля всосавшейся дозы
                 SR4=0.714,          # секреция T4 (мкг/ч) для гипотиреоза (17% нормы)
                 SR3=0.051,          # секреция T3 (мкг/ч) для гипотиреоза
                 k12=0.25, k21=0.15,
                 k13=0.05, k31=0.02,
                 k_ex4=0.012,
                 k_conv=0.04,
                 k45=0.35, k54=0.25,
                 k46=0.12, k64=0.08,
                 k_ex3=0.035,
                 tsh_gain=30.0,
                 tsh_tau=150.0,
                 initial_state=None,
                 condition='custom'):
        self.BW = weight_kg
        self.condition = condition

        self.k_abs = k_abs
        self.SR4 = SR4
        self.SR3 = SR3
        self.k12, self.k21 = k12, k21
        self.k13, self.k31 = k13, k31
        self.k_ex4 = k_ex4
        self.k_conv = k_conv
        self.k45, self.k54 = k45, k54
        self.k46, self.k64 = k46, k64
        self.k_ex3 = k_ex3
        self.tsh_gain = tsh_gain
        self.tsh_tau = tsh_tau

        if initial_state is None:
            # Начальные состояния для гипотиреоза: низкий T4, высокий TSH
            self.state = np.array([28.0, 22.0, 18.0,   # T4p, T4f, T4s
                                    0.5, 0.4, 0.3,     # T3p, T3f, T3s
                                    25.0, 0.0], dtype=float)   # TSH, Gut
        else:
            self.state = np.array(initial_state, dtype=float)

    def equations(self, y, t, dose_input_rate):
        T4p, T4f, T4s, T3p, T3f, T3s, TSH, Gut = y

        dGut_dt = dose_input_rate - self.k_abs * Gut

        dT4p_dt = (self.k_abs * Gut) + self.SR4 - (self.k12 + self.k13 + self.k_ex4) * T4p + self.k21 * T4f + self.k31 * T4s
        dT4f_dt = self.k12 * T4p - (self.k21 + self.k_conv) * T4f
        dT4s_dt = self.k13 * T4p - self.k31 * T4s

        dT3p_dt = self.SR3 - (self.k45 + self.k46 + self.k_ex3) * T3p + self.k54 * T3f + self.k64 * T3s
        dT3f_dt = self.k45 * T3p - self.k54 * T3f + self.k_conv * T4f
        dT3s_dt = self.k46 * T3p - self.k64 * T3s

        tsh_target = 1.0 + self.tsh_gain * np.exp(-0.05 * T4p)  # эмпирическая функция
        dTSH_dt = (tsh_target - TSH) / self.tsh_tau

        return [dT4p_dt, dT4f_dt, dT4s_dt, dT3p_dt, dT3f_dt, dT3s_dt, dTSH_dt, dGut_dt]

    def run_simulation(self, daily_dose, hours=24):
        t = np.linspace(0, hours, hours + 1)
        dose_rate = daily_dose / 24.0
        sol = odeint(self.equations, self.state, t, args=(dose_rate,))
        self.state = sol[-1]
        return self.state

    def get_ft4(self):
        # Примерное преобразование T4p в FT4 (нг/л) – масштаб подобран эмпирически
        return self.state[0] / 3.0

    def get_tsh(self):
        return self.state[6]