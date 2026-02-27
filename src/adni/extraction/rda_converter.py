"""
rda_converter.py — ADNIMERGE2/data/*.rda -> csv/tables/*.csv 일괄 변환

pyreadr를 사용하여 R의 .rda 파일을 Python DataFrame으로 읽고 CSV로 저장.
"""

import os
import logging
from glob import glob

import pyreadr
import pandas as pd

logger = logging.getLogger(__name__)


def convert_single_rda(rda_path: str, output_dir: str) -> dict:
    """단일 .rda 파일을 CSV로 변환.

    Returns:
        dict with keys: name, rows, cols, status, error
    """
    basename = os.path.splitext(os.path.basename(rda_path))[0]
    out_path = os.path.join(output_dir, basename + '.csv')
    info = {'name': basename, 'rows': 0, 'cols': 0, 'status': 'ok', 'error': None}

    try:
        result = pyreadr.read_r(rda_path)
        if not result:
            info['status'] = 'empty'
            info['error'] = 'No objects in .rda file'
            return info

        # .rda 파일은 여러 객체를 포함할 수 있음 -> 첫 번째 DataFrame 사용
        obj_name = list(result.keys())[0]
        df = result[obj_name]

        if not isinstance(df, pd.DataFrame):
            info['status'] = 'skip'
            info['error'] = f'Object "{obj_name}" is not a DataFrame'
            return info

        info['rows'] = len(df)
        info['cols'] = len(df.columns)

        df.to_csv(out_path, index=False)

    except Exception as e:
        info['status'] = 'error'
        info['error'] = str(e)

    return info


def convert_all_rda(rda_dir: str, output_dir: str) -> list:
    """rda_dir 내 모든 .rda 파일을 output_dir에 CSV로 변환.

    Returns:
        list of info dicts (하나당 하나의 .rda 파일)
    """
    os.makedirs(output_dir, exist_ok=True)

    rda_files = sorted(glob(os.path.join(rda_dir, '*.rda')))
    if not rda_files:
        logger.warning('No .rda files found in %s', rda_dir)
        return []

    logger.info('Found %d .rda files in %s', len(rda_files), rda_dir)

    results = []
    ok_count = 0
    err_count = 0

    for rda_path in rda_files:
        info = convert_single_rda(rda_path, output_dir)
        results.append(info)

        if info['status'] == 'ok':
            ok_count += 1
            logger.info('  [OK] %s -> %d rows x %d cols',
                         info['name'], info['rows'], info['cols'])
        else:
            err_count += 1
            logger.warning('  [%s] %s: %s',
                            info['status'].upper(), info['name'], info['error'])

    logger.info('Conversion complete: %d OK, %d errors/skipped out of %d total',
                ok_count, err_count, len(rda_files))

    return results


def print_report(results: list):
    """변환 결과 리포트 출력."""
    print('=' * 70)
    print('RDA -> CSV Conversion Report')
    print('=' * 70)

    ok = [r for r in results if r['status'] == 'ok']
    errors = [r for r in results if r['status'] != 'ok']

    print(f'Total: {len(results)} files')
    print(f'  OK: {len(ok)}')
    print(f'  Errors/Skipped: {len(errors)}')
    print(f'  Total rows: {sum(r["rows"] for r in ok):,}')

    if errors:
        print('\nErrors/Skipped:')
        for r in errors:
            print(f'  [{r["status"]}] {r["name"]}: {r["error"]}')

    print('=' * 70)
