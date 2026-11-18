"""Построение ЛП-модели из данных варианта: матрицы для scipy.optimize.linprog.

Порядок переменных: x[s*24 + t] — выработка станции s в часе t, МВт·ч.
Модель min c·x при A_eq·x = b_eq (баланс), A_ub·x <= b_ub (темп), 0 <= x <= pmax.
"""
import numpy as np

HOURS = 24


def build(stations, demand):
    """Собирает задачу в виде dict(c, A_eq, b_eq, A_ub, b_ub, bounds).

    Сценарии реализуются масштабированием самих данных (спрос/мощность),
    поэтому модель всегда строится из таблиц как есть.
    """
    n_s = len(stations)
    n = n_s * HOURS
    cost = stations['cost'].to_numpy()
    pmax = stations['pmax'].to_numpy()
    ramp = stations['ramp'].to_numpy()
    d = demand['demand'].to_numpy()

    c = np.repeat(cost, HOURS)  # руб./(МВт·ч) на каждую переменную

    # Баланс: Σ_s x[s,t] = d_t в каждый час.
    A_eq = np.zeros((HOURS, n))
    for t in range(HOURS):
        A_eq[t, t::HOURS] = 1.0
    b_eq = d

    # Темп: |x[s,t] − x[s,t−1]| ≤ ramp[s] для t = 1..23 (два неравенства на пару).
    rows, b_ub = [], []
    for s in range(n_s):
        for t in range(1, HOURS):
            r = np.zeros(n)
            r[s * HOURS + t] = 1.0
            r[s * HOURS + t - 1] = -1.0
            rows.append(r)
            b_ub.append(ramp[s])   # рост мощности
            rows.append(-r)
            b_ub.append(ramp[s])  # сброс мощности
    A_ub = np.array(rows)
    b_ub = np.array(b_ub)

    bounds = np.column_stack([np.zeros(n), np.repeat(pmax, HOURS)])
    return {'c': c, 'A_eq': A_eq, 'b_eq': b_eq, 'A_ub': A_ub, 'b_ub': b_ub,
            'bounds': bounds, 'n_s': n_s, 'hours': HOURS}
