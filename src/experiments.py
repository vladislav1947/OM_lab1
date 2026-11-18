"""Обязательные эксперименты 1–6 и доп. задание (симплекс vs внутренняя точка).

Сценарии реализуются масштабированием исходных данных, поэтому модель и
независимая проверка всегда видят один и те же данные.
"""
import numpy as np
import pandas as pd

from src.model import build
from src.solve import solve, status_text
from src.verify import verify


def scaled(stations, demand, demand_scale=1.0, pmax_scale=1.0):
    """Копии данных с масштабированием спроса и/или мощности станций."""
    st = stations.assign(pmax=stations['pmax'] * pmax_scale)
    dm = demand.assign(demand=demand['demand'] * demand_scale)
    return st, dm


def scenario(name, stations, demand, demand_scale=1.0, pmax_scale=1.0, log_path=None):
    """Решает один сценарий; возвращает dict со статусом, планом и сводкой."""
    st, dm = scaled(stations, demand, demand_scale, pmax_scale)
    prob = build(st, dm)
    res, dt = solve(prob, log_path=log_path, header=f'сценарий: {name}')
    info = {'имя': name, 'статус': status_text(res), 'res': res, 'время': dt}
    if res.status == 0:
        x = res.x.reshape(len(stations), -1)
        vtable, ok, obj = verify(x, st, dm)
        pmax = st['pmax'].to_numpy()[:, None]
        info.update({'x': x, 'проверки': vtable, 'допустимо': ok, 'затраты': obj,
                     'сутки_станции': x.sum(axis=1),
                     'часы_на_максимуме': (x >= pmax - 1e-6).sum(axis=1),
                     'нулевые_часы': (x <= 1e-9).sum(axis=1)})
    return info


def run_all(stations, demand, log_path=None):
    """Эксперименты 1–6: базовый, дефицитный, стрессовый, теневые цены, нули.

    Возвращает dict с DataFrame'ами и сводными числами для отчёта.
    """
    base = scenario('базовый', stations, demand, log_path=log_path)
    deficit = scenario('дефицитный (+20% спроса)', stations, demand,
                      demand_scale=1.2, log_path=log_path)
    stress = scenario('стрессовый (+20% спроса и −20% мощности)', stations, demand,
                      demand_scale=1.2, pmax_scale=0.8, log_path=log_path)

    # Конфликт в стрессовом сценарии: часы, где требуемая выработка
    # превышает доступную мощность (баланс против ограничений мощности).
    req = 1.2 * demand['demand'].to_numpy()
    avail = 0.8 * stations['pmax'].sum()
    hours = np.arange(len(req))[req > avail]
    conflict = pd.DataFrame({
        'час': hours,
        'требуемая выработка 1.2·d_t, МВт·ч': req[hours].round(1),
        'доступная мощность 0.8·Σpmax, МВт·ч': round(avail, 1),
        'дефицит, МВт·ч': (req[hours] - avail).round(1),
    })

    # Теневые цены базового решения.
    lam = base['res'].eqlin.marginals  # руб./(МВт·ч) — цена спроса в час t
    cap_marg = base['res'].upper.marginals.reshape(len(stations), -1)
    duals = pd.DataFrame({
        'час': demand['hour'],
        'теневая цена баланса, руб./(МВт·ч)': lam.round(2),
    })
    # Ценность расширения мощности станции: суммарный эффект +1 МВт во все часы.
    capacity_value = pd.DataFrame({
        'станция': stations['name'],
        'часов на максимуме': base['часы_на_максимуме'],
        'ценность +1 МВт, руб./сутки': (-cap_marg.sum(axis=1)).round(1),
    })

    # Сверка прогноза теневой цены с фактом.
    d0 = demand['demand'].to_numpy()
    micro = scenario('микро (+10 МВт·ч в час 19)', stations,
                     demand.assign(demand=demand['demand'].where(
                         demand['hour'] != 19, demand['demand'] + 10)),
                     log_path=log_path)
    checks = pd.DataFrame([
        {'проверка': 'микро: +10 МВт·ч к спросу в час 19',
         'прогноз по λ_19, руб.': lam[19] * 10,
         'фактическое Δ затрат, руб.': micro['затраты'] - base['затраты']},
        {'проверка': 'дефицит: +20% спроса во все часы',
         'прогноз по λ, руб.': float(lam @ (0.2 * d0)),
         'фактическое Δ затрат, руб.': deficit['затраты'] - base['затраты']},
    ])

    scenarios = pd.DataFrame([
        {'сценарий': i['имя'], 'статус': i['статус'].split(' —')[0],
         'затраты, руб.': round(i.get('затраты', np.nan), 2),
         'Δ к базе, руб.': round(i['затраты'] - base['затраты'], 2)
         if 'затраты' in i else None}
        for i in (base, deficit, stress)
    ])

    return {'base': base, 'deficit': deficit, 'stress': stress,
            'conflict': conflict, 'duals': duals, 'capacity_value': capacity_value,
            'checks': checks, 'scenarios': scenarios,
            'lam': lam, 'cap_marg': cap_marg}


def compare_algorithms(stations, demand, n_runs=3):
    """Доп. задание: двойственный симплекс vs метод внутренней точки."""
    prob = build(stations, demand)
    rows, sols = [], {}
    for method in ('highs-ds', 'highs-ipm'):
        times = []
        for _ in range(n_runs):
            res, dt = solve(prob, method=method)
            times.append(dt)
        sols[method] = res
        rows.append({'метод': method,
                     'время (медиана), с': round(float(np.median(times)), 4),
                     'итерации': res.nit,
                     'итерации crossover': getattr(res, 'crossover_nit', None),
                     'затраты, руб.': round(res.fun, 2),
                     'статус': res.status})
    dx = float(np.abs(sols['highs-ds'].x - sols['highs-ipm'].x).max())
    rows[0]['max|Δx| к IPM, МВт·ч'] = None
    rows[1]['max|Δx| к IPM, МВт·ч'] = dx
    return pd.DataFrame(rows)
