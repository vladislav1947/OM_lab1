"""Сборка отчёта в PDF: REPORT.md → HTML (markdown) → PDF (weasyprint).

Запуск: python build_pdf.py → ЛР1_Осипов_ВА_Б24-517.pdf в корне репо.
Относительные пути к PNG (output/*.png) подтягиваются через base_url='.'.
"""
import markdown
from weasyprint import HTML

TITLE = ('Лабораторная работа № 1. ЛП-модель распределения ресурсов. '
         'Вариант 6. Осипов В. А., группа Б24-517')
OUT = 'ЛР1_Осипов_ВА_Б24-517.pdf'

CSS = """
body{font-family:"DejaVu Sans"; max-width:800px; margin:2em auto; font-size:11pt; line-height:1.45}
@page{margin:2cm}
table{border-collapse:collapse} td,th{border:1px solid #999; padding:4px 8px}
img{max-width:100%}
h1,h2,h3{color:#1a3c6e}
pre{background:#f5f5f5; padding:8px; overflow-x:auto}
.title{font-size:14pt; font-weight:bold; color:#1a3c6e; margin-bottom:1em}
"""


def build():
    with open('REPORT.md', encoding='utf-8') as f:
        body = markdown.markdown(f.read(), extensions=['tables', 'fenced_code'])
    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<style>{CSS}</style></head><body>
<p class="title">{TITLE}</p>
{body}
</body></html>"""
    HTML(string=html, base_url='.').write_pdf(OUT)
    print(f'PDF собран: {OUT}')


if __name__ == '__main__':
    build()
