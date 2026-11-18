"""Решение ЛП солвером HiGHS через scipy.optimize.linprog + интерпретация статуса."""
import contextlib
import os
import sys
import tempfile
import time

from scipy.optimize import linprog

STATUS_RU = {
    0: 'optimal — найдено глобально оптимальное решение ЛП',
    1: 'iteration limit — достигнут лимит итераций, решение не завершено',
    2: 'infeasible — допустимых планов нет, ограничения противоречат друг другу',
    3: 'unbounded — целевая не ограничена снизу на допустимом множестве',
    4: 'numerical difficulties — численные проблемы солвера',
}


@contextlib.contextmanager
def _capture_stdout_fd():
    """Перехват stdout на уровне дескриптора: HiGHS пишет лог в fd 1 напрямую,
    мимо sys.stdout, поэтому contextlib.redirect_stdout не годится."""
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+b')
    os.dup2(tmp.fileno(), 1)
    try:
        yield tmp
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        tmp.seek(0)


def solve(prob, method='highs', log_path=None, header=''):
    """Решает модель; возвращает (res, время_решения_сек).

    Лог HiGHS (options={'disp': True}) дописывается в log_path, если задан.
    """
    t0 = time.perf_counter()
    with _capture_stdout_fd() as tmp:
        res = linprog(prob['c'], A_ub=prob['A_ub'], b_ub=prob['b_ub'],
                      A_eq=prob['A_eq'], b_eq=prob['b_eq'],
                      bounds=prob['bounds'], method=method,
                      options={'disp': True})
    dt = time.perf_counter() - t0
    if log_path:
        log = tmp.read().decode('utf-8', errors='replace')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n===== {header or method} =====\n" + log)
    return res, dt


def status_text(res):
    """Русская интерпретация статуса солвера (см. п. 1.2.8 TASK.md)."""
    return STATUS_RU.get(res.status, f'неизвестный статус {res.status}')
