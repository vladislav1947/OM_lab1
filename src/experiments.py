"""Обязательные эксперименты 1–6 и доп. задание (симплекс vs внутренняя точка).

Сценарии реализуются масштабированием исходных данных, поэтому модель и
независимая проверка всегда видят один и те же данные.
"""
import numpy as np
import pandas as pd

from src.data_gen import SEED
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
    info = {'имя': name, 'статус': status_text(res), 'res': res, 'время': dt,
            'затраты': np.nan,
            # масштабированные копии: артефакты строятся из них, а не из ×констант
            'данные_спрос': dm, 'данные_станции': st}
    if res.status == 0:
        x = res.x.reshape(len(stations), -1)
        vtable, ok, obj = verify(x, st, dm)
        # каждый допустимый сценарий обязан проходить независимую проверку
        assert ok, f'сценарий «{name}»: независимая проверка нашла нарушения'
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
    # Числа берём из масштабированных копий стресса, а не из констант.
    req = stress['данные_спрос']['demand'].to_numpy()
    avail = stress['данные_станции']['pmax'].sum()
    hours = np.arange(len(req))[req > avail]
    conflict = pd.DataFrame({
        'час': hours,
        'требуемая выработка, МВт·ч': req[hours].round(1),
        'доступная мощность, МВт·ч': round(avail, 1),
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
         'прогноз по λ, руб.': lam[19] * 10,
         'фактическое Δ затрат, руб.': micro['затраты'] - base['затраты']},
        {'проверка': 'дефицит: +20% спроса во все часы',
         'прогноз по λ, руб.': float(lam @ (0.2 * d0)),
         'фактическое Δ затрат, руб.': deficit['затраты'] - base['затраты']},
    ])

    scenarios = pd.DataFrame([
        {'сценарий': i['имя'], 'статус': i['статус'].split(' —')[0],
         'затраты, руб.': round(i.get('затраты', np.nan), 2),
         'Δ к базе, руб.': round(i['затраты'] - base['затраты'], 2)
         if np.isfinite(i['затраты']) else None}
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


# ===== Задание «со звёздочкой» № 2: бенчмарк симплекс против внутренней точки =====

def benchmark_instances(n_tasks=10, seed=SEED):
    """10 задач того же семейства (план выработки) растущей размерности.

    Детерминировано по seed. Задача i (i = 0..9): станций 5+i, горизонт 24+8i ч.
    Структура как у варианта: половина станций базовые (дешёвые, крупные,
    маломанёвренные), остальные пиковые (дорогие, быстрые); спрос масштабируется
    от суммарной мощности станций, вечерний пик ≈ 0.75·Σpmax — тот же режим
    нагруженности, что и у варианта 6.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_tasks):
        n_s = 5 + i
        n_base = n_s // 2
        hours = 24 + 8 * i
        pmax = np.concatenate([rng.uniform(270, 330, n_base),
                               rng.uniform(60, 80, n_s - n_base)])
        cost = np.concatenate([rng.uniform(800, 1400, n_base),
                               rng.uniform(2200, 3400, n_s - n_base)])
        ramp = np.concatenate([rng.uniform(40, 80, n_base),
                               rng.uniform(120, 200, n_s - n_base)])
        stations = pd.DataFrame({
            'station_id': np.arange(1, n_s + 1),
            'name': [f'S{j + 1}' for j in range(n_s)],
            'type': ['базовая'] * n_base + ['пиковая'] * (n_s - n_base),
            'cost': cost, 'pmax': pmax, 'ramp': ramp,
        })
        cap = pmax.sum()
        t = np.arange(hours)
        demand = pd.DataFrame({
            'hour': t,
            'demand': cap * (0.6 + 0.15 * np.cos(2 * np.pi * (t - 0.8 * hours) / hours))
                     + rng.uniform(-0.01 * cap, 0.01 * cap, hours),
        })
        out.append((stations, demand))
    return out


def star_benchmark(n_tasks=10, n_runs=5, seed=SEED):
    """Бенчмарк: highs-ds (двойственный симплекс) против highs-ipm (IPX+crossover).

    Для каждой задачи — медиана времени по n_runs прогонам и число итераций.
    Возвращает длинный DataFrame (метод — строка).
    """
    rows = []
    for i, (st, dm) in enumerate(benchmark_instances(n_tasks, seed)):
        prob = build(st, dm)
        for method in ('highs-ds', 'highs-ipm'):
            times, res = [], None
            for _ in range(n_runs):
                res, dt = solve(prob, method=method)
                times.append(dt)
            rows.append({'задача': i + 1, 'станций': len(st), 'часов': len(dm),
                         'переменных': prob['c'].size, 'метод': method,
                         'время (медиана), с': round(float(np.median(times)), 4),
                         'итерации': res.nit, 'статус': res.status})
    return pd.DataFrame(rows)
