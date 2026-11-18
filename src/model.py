"""Построение ЛП-модели из данных: матрицы для scipy.optimize.linprog.

Порядок переменных: x[s*T + t] — выработка станции s в часе t, МВт·ч (T — горизонт).
Модель min c·x при A_eq·x = b_eq (баланс), A_ub·x <= b_ub (темп), 0 <= x <= pmax.
"""
import numpy as np


def build(stations, demand):
    """Собирает задачу в виде dict(c, A_eq, b_eq, A_ub, b_ub, bounds).

    Горизонт T берётся из данных (len(demand)) — модель применима и к задачам
    семейства другой размерности (бенчмарк «звёздочки» № 2). Сценарии
    реализуются масштабированием самих данных (спрос/мощность), поэтому модель
    всегда строится из таблиц как есть.
    """
    n_s = len(stations)
    hours = len(demand)
    n = n_s * hours
    cost = stations['cost'].to_numpy()
    pmax = stations['pmax'].to_numpy()
    ramp = stations['ramp'].to_numpy()
    d = demand['demand'].to_numpy()

    c = np.repeat(cost, hours)  # руб./(МВт·ч) на каждую переменную

    # Баланс: Σ_s x[s,t] = d_t в каждый час.
    A_eq = np.zeros((hours, n))
    for t in range(hours):
        A_eq[t, t::hours] = 1.0
    b_eq = d

    # Темп: |x[s,t] − x[s,t−1]| ≤ ramp[s] (два неравенства на пару часов).
    rows, b_ub = [], []
    for s in range(n_s):
        for t in range(1, hours):
            r = np.zeros(n)
            r[s * hours + t] = 1.0
            r[s * hours + t - 1] = -1.0
            rows.append(r)
            b_ub.append(ramp[s])   # рост мощности
            rows.append(-r)
            b_ub.append(ramp[s])  # сброс мощности
    A_ub = np.array(rows)
    b_ub = np.array(b_ub)

    bounds = np.column_stack([np.zeros(n), np.repeat(pmax, hours)])
    return {'c': c, 'A_eq': A_eq, 'b_eq': b_eq, 'A_ub': A_ub, 'b_ub': b_ub,
            'bounds': bounds, 'n_s': n_s, 'hours': hours}
