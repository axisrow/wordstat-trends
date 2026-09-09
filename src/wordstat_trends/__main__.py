"""CLI контракта данных: ``python -m wordstat_trends <путь>...`` (issue #94).

Аргументы — run-каталоги или одиночные dynamics.csv. Отчёт — построчно в
stdout, каждое нарушение — именованная проверка; exit 1 при любом нарушении
серьёзности error (info не роняет). Проверяемые инварианты — docs/DATA.md.
"""

from __future__ import annotations

import sys

from wordstat_trends.validate import validate_dataset


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("использование: python -m wordstat_trends <run-каталог|dynamics.csv>...", file=sys.stderr)
        return 2

    reports = validate_dataset(args)
    for report in reports:
        report.print()
    failed = [r for r in reports if not r.ok()]
    if failed:
        print(f"{len(failed)}/{len(reports)} не прошли контракт данных", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
