"""Графики matplotlib (backend Agg, без отображения): план, сценарии, теневые цены."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(stations, demand, r, out_dir='output'):
    """Строит PNG в output/: план выработки, спрос против мощности, теневые цены."""
    hours = demand['hour'].to_numpy()
    d = demand['demand'].to_numpy()
    x = r['base']['x']
    names = stations['name'].tolist()
    cost = stations['cost'].to_numpy()
    lam = r['lam']
    deficit = r['deficit']

    # 1) план выработки по станциям и часам (базовый сценарий)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(hours, x, labels=names, alpha=0.85)
    ax.plot(hours, d, 'k-', lw=2, label='спрос')
    ax.set_xlabel('час')
    ax.set_ylabel('МВт·ч за час')
    ax.set_title('Оптимальный план выработки (базовый сценарий)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 23)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/plan_generation.png', dpi=150)
    plt.close(fig)

    # 2) спрос против доступной мощности: почему стресс недопустим
    fig, ax = plt.subplots(figsize=(10, 5))
    cap = stations['pmax'].sum()
    ax.plot(hours, d, 'k-', lw=2, label='спрос (база)')
    ax.plot(hours, 1.2 * d, 'r--', lw=2, label='спрос ×1.2 (дефицит и стресс)')
    ax.axhline(cap, color='g', ls='-', label='Σpmax (база)')
    ax.axhline(0.8 * cap, color='g', ls=':', label='0.8·Σpmax (стресс)')
    ax.fill_between(hours, 1.2 * d, 0.8 * cap, where=1.2 * d > 0.8 * cap,
                    color='red', alpha=0.25, label='дефицит мощности (стресс недопустим)')
    ax.set_xlabel('час')
    ax.set_ylabel('МВт·ч за час')
    ax.set_title('Спрос против доступной мощности по сценариям')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 23)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/demand_vs_capacity.png', dpi=150)
    plt.close(fig)

    # 3) теневые цены баланса против удельных цен станций
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(hours, lam, where='mid', color='b', lw=2, label='λ_t — теневая цена баланса')
    for c, nm in sorted(zip(cost, names)):
        ax.axhline(c, ls='--', lw=0.8, color='gray')
        ax.annotate(nm, (23.2, c), fontsize=8, color='gray', va='center')
    ax.set_xlabel('час')
    ax.set_ylabel('руб./(МВт·ч)')
    ax.set_title('Теневая цена баланса = удельная цена замыкающей станции')
    ax.legend()
    ax.set_xlim(0, 26)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/shadow_prices.png', dpi=150)
    plt.close(fig)

    # 4) суточные затраты по сценариям
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['базовый', 'дефицит\n(+20% спроса)', 'стрессовый\n(+20% спроса,\n−20% мощности)']
    vals = [r['base']['затраты'] / 1e6, deficit['затраты'] / 1e6, 0]
    bars = ax.bar(labels, vals, color=['tab:green', 'tab:orange', 'tab:red'])
    bars[2].set_hatch('//')
    ax.annotate('infeasible', (2, 0.1), ha='center', color='tab:red')
    ax.set_ylabel('млн руб./сутки')
    ax.set_title('Суточные затраты по сценариям')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/scenario_costs.png', dpi=150)
    plt.close(fig)
