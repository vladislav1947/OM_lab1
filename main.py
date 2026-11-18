"""Единый конвейер ЛР № 1 (запуск: python main.py).

Данные → модель → решение → независимая проверка → эксперименты → графики.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_gen import generate, save
from src.experiments import compare_algorithms, run_all, star_benchmark
from src.plots import benchmark_plots, make_plots
from src.verify import summarize


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    log_path = 'output/solver_log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)  # лог пишется заново при каждом прогоне

    # 1) данные варианта (seed = 10116)
    stations, demand = generate()
    save(stations, demand)
    print('Данные: 5 станций, 24 ч, seed=10116; Σpmax = '
          f'{stations["pmax"].sum():.1f} МВт, пик спроса = {demand["demand"].max():.1f} МВт·ч')

    # 2–5) обязательные эксперименты 1–6 (внутри — решение и независимая проверка)
    r = run_all(stations, demand, log_path=log_path)
    base, deficit, stress = r['base'], r['deficit'], r['stress']
    print(f"\nБазовый сценарий: {base['статус']}")
    print(f"  суточные затраты: {base['затраты']:.2f} руб., время решения: {base['время']:.3f} с")
    print(f"  независимая проверка (допуск 1e-7): все ограничения в допуске — {base['допустимо']},"
          f"\n  пересчёт цели: {base['затраты']:.2f} руб. (совпадает с солвером)")

    # решение в предметных терминах
    report_st = pd.DataFrame({
        'станция': stations['name'],
        'тип': stations['type'],
        'цена, руб./(МВт·ч)': stations['cost'].round(1),
        'pmax, МВт': stations['pmax'].round(1),
        'выработка за сутки, МВт·ч': base['сутки_станции'].round(1),
        'часов на максимуме': base['часы_на_максимуме'],
        'часов в нуле': base['нулевые_часы'],
    })
    print('\nПлан по станциям (база):\n' + report_st.to_string(index=False))

    print(f"\nДефицитный сценарий (+20% спроса): {deficit['статус']},"
          f" затраты {deficit['затраты']:.2f} руб. (+{deficit['затраты'] - base['затраты']:.2f})")
    print('  часов на максимуме по станциям:',
          dict(zip(stations['name'], deficit['часы_на_максимуме'])))
    print(f"Стрессовый сценарий (+20% спроса, −20% мощности): {stress['статус']}")
    print('  конфликтующие часы (требуемая выработка > доступной мощности):\n'
          + stress_conflict_table(r).to_string(index=False))

    print('\nСверка теневых цен:\n' + r['checks'].round(2).to_string(index=False))
    print('\nЦенность расширения мощности (+1 МВт):\n'
          + r['capacity_value'].to_string(index=False))

    # 6) доп. задание: симплекс против внутренней точки
    algos = compare_algorithms(stations, demand)
    print('\nДоп. задание, симплекс vs внутренняя точка:\n' + algos.to_string(index=False))

    # 7) задание «со звёздочкой» № 2: бенчмарк на 10 задачах растущей размерности
    bench = star_benchmark()
    print('\nЗвёздочка № 2, бенчмарк симплекс/IPM на 10 задачах (5–14 станций, '
          '24–96 ч):\n' + bench.to_string(index=False))

    # выгрузка артефактов
    pd.DataFrame({
        'час': demand['hour'],
        **{nm: base['x'][i].round(3) for i, nm in enumerate(stations['name'])},
        'сумма, МВт·ч': base['x'].sum(axis=0).round(3),
        'спрос, МВт·ч': demand['demand'].round(1),
    }).to_csv('output/solution.csv', index=False)
    base['проверки'].to_csv('output/verification.csv', index=False)
    r['scenarios'].to_csv('output/scenarios.csv', index=False)
    stress_conflict_table(r).to_csv('output/stress_conflict.csv', index=False)
    r['duals'].to_csv('output/duals.csv', index=False)
    r['capacity_value'].to_csv('output/capacity_value.csv', index=False)
    r['checks'].to_csv('output/shadow_check.csv', index=False)
    algos.to_csv('output/algos.csv', index=False)
    bench.to_csv('output/benchmark.csv', index=False)
    with open('output/verification_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summarize(base['проверки']).round(9).to_string(index=False))

    make_plots(stations, demand, r)
    benchmark_plots(bench)

    print(f"\nАртефакты: output/ (solution.csv, verification.csv, scenarios.csv, "
          f"stress_conflict.csv, duals.csv, capacity_value.csv, shadow_check.csv, "
          f"algos.csv, benchmark.csv, solver_log.txt, *.png)")
    print('Сводка проверки по типам ограничений:\n'
          + summarize(base['проверки']).round(9).to_string(index=False))


def stress_conflict_table(r):
    return r['conflict'].rename(columns={
        'требуемая выработка 1.2·d_t, МВт·ч': 'требуемая выработка, МВт·ч',
        'доступная мощность 0.8·Σpmax, МВт·ч': 'доступная мощность, МВт·ч'})


if __name__ == '__main__':
    main()
