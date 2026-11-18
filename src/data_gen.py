"""Детерминированная генерация данных варианта 6 (план выработки электроэнергии).

Тот же seed (10116) даёт те же данные. Процедура описана в data/README.md.
"""
import numpy as np
import pandas as pd

SEED = 10116
HOURS = 24
N_BASE, N_PEAK = 2, 3  # 2 угольные КЭС (базовые) + 3 газотурбинные установки (пиковые)


def generate(seed=SEED):
    """Генерирует таблицы станций и спроса; возвращает (stations, demand)."""
    rng = np.random.default_rng(seed)
    pmax = np.concatenate([rng.uniform(270, 330, N_BASE), rng.uniform(60, 80, N_PEAK)])
    cost = np.concatenate([rng.uniform(800, 1400, N_BASE), rng.uniform(2200, 3400, N_PEAK)])
    ramp = np.concatenate([rng.uniform(40, 80, N_BASE), rng.uniform(120, 200, N_PEAK)])
    stations = pd.DataFrame({
        'station_id': np.arange(1, N_BASE + N_PEAK + 1),
        'name': ['КЭС-1', 'КЭС-2', 'ГТУ-1', 'ГТУ-2', 'ГТУ-3'],
        'type': ['базовая'] * N_BASE + ['пиковая'] * N_PEAK,
        'cost': cost,   # удельная стоимость, руб./(МВт·ч)
        'pmax': pmax,   # максимальная мощность, МВт
        'ramp': ramp,   # допустимый темп изменения мощности, МВт/ч
    })
    t = np.arange(HOURS)
    demand = pd.DataFrame({
        'hour': t,
        # суточный профиль: вечерний пик ~ час 19, ночной минимум ~ час 7
        'demand': 460 + 150 * np.cos(np.pi * (t - 19) / 12) + rng.uniform(-12, 12, HOURS),
    })
    return stations, demand


def save(stations, demand, data_dir='data'):
    """Выгружает данные варианта в CSV."""
    stations.to_csv(f'{data_dir}/stations.csv', index=False)
    demand.to_csv(f'{data_dir}/demand.csv', index=False)


if __name__ == '__main__':
    st, dm = generate()
    save(st, dm)
    print(st.round(1).to_string(index=False))
    print(dm.round(1).to_string(index=False))
