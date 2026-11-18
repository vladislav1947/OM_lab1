"""Независимая проверка решения чистым NumPy по массивам данных.

Не обращается к объекту солвера: все ограничения восстанавливаются напрямую из
таблиц станций и спроса. Допуск 1e-7.
"""
import numpy as np
import pandas as pd

TOL = 1e-7


def verify(x, stations, demand, tol=TOL):
    """Проверяет план x формы (n_s, 24); возвращает (DataFrame проверок, ok, obj).

    Строки таблицы: тип ограничения, элемент, использование, запас, нарушение.
    Дополнительно пересчитывает целевую функцию.
    """
    cost = stations['cost'].to_numpy()
    pmax = stations['pmax'].to_numpy()
    ramp = stations['ramp'].to_numpy()
    d = demand['demand'].to_numpy()
    names = stations['name'].tolist()

    rows = []

    def add(kind, item, usage, limit):
        # нарушение > 0 — ограничение нарушено; знак определяется типом
        if kind.endswith('≤'):
            violation = usage - limit
        elif kind.endswith('≥'):
            violation = limit - usage
        else:
            violation = abs(usage - limit)
        rows.append({'тип': kind, 'элемент': item, 'использование': usage,
                     'запас/правая часть': limit, 'нарушение': violation,
                     'в допуске': bool(violation <= tol)})

    # 1) баланс мощности: Σ_s x[s,t] = d_t
    for t in range(x.shape[1]):
        add('баланс =', f'час {t}', x[:, t].sum(), d[t])

    # 2) мощность станций: 0 ≤ x[s,t] ≤ pmax[s]
    for s in range(x.shape[0]):
        for t in range(x.shape[1]):
            add('нижняя граница ≥', f'{names[s]}, час {t}', x[s, t], 0.0)
            add('мощность ≤', f'{names[s]}, час {t}', x[s, t], pmax[s])

    # 3) темп: |x[s,t] − x[s,t−1]| ≤ ramp[s]
    for s in range(x.shape[0]):
        for t in range(1, x.shape[1]):
            add('темп ≤', f'{names[s]}, часы {t-1}→{t}',
                abs(x[s, t] - x[s, t - 1]), ramp[s])

    table = pd.DataFrame(rows)
    obj = float(cost @ x.sum(axis=1))
    ok = bool(table['в допуске'].all())
    return table, ok, obj


def summarize(table):
    """Сводка по типам ограничений: сколько проверено, максимальное нарушение."""
    return (table.groupby('тип')
            .agg(проверок=('нарушение', 'size'),
                 max_нарушение=('нарушение', 'max'))
            .reset_index())
