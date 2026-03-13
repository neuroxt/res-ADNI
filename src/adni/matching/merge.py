"""
merge.py — CSV 병합

ADNI.py unique_csv_merge()와 동일 로직:
- 모든 *_unique.csv를 PTID+VISCODE_FIX 기준 outer join
- 가장 큰 CSV부터 시작 (init), 나머지를 순서대로 병합
- 출력: MERGED.csv
"""

import os
import logging
import pandas as pd
from glob import glob


def unique_csv_merge(output_directory: str, output_filename: str = 'MERGED.csv',
                     exclude_modalities: list = None):
    """
    *_unique.csv 파일들을 PTID+VISCODE_FIX 기준으로 병합

    ADNI.py unique_csv_merge()와 동일 로직:
    1. output_directory에서 *_unique.csv 파일 수집
    2. 행 수 기준 내림차순 정렬 (가장 큰 것이 init)
    3. outer join으로 병합 (중복 컬럼은 init 우선)
    4. 새 인덱스(다른 CSV에만 존재하는 PTID+VISCODE)는 해당 CSV 값 사용

    Args:
        output_directory: *_unique.csv 파일이 있는 디렉토리
        output_filename: 출력 파일명 (기본: MERGED.csv)
        exclude_modalities: 제외할 모달리티 (e.g., ['ADC', 'FA', 'TRACEW'])
    """
    logging.info('-------------------- Unique csv merge --------------------')
    output_path = os.path.join(output_directory, output_filename)

    flist = sorted(glob(os.path.join(output_directory, '*_unique.csv')))

    # 제외 모달리티 필터링
    if exclude_modalities:
        exclude_set = {m.upper() for m in exclude_modalities}
        flist = [f for f in flist if
                 os.path.basename(f).replace('_unique.csv', '').upper() not in exclude_set]

    if not flist:
        logging.warning('No *_unique.csv files found in %s' % output_directory)
        return

    # 로드 + 행 수 기준 정렬 (중복 인덱스 제거)
    df_list = []
    for f in flist:
        logging.info('input csv: %s' % f)
        df = pd.read_csv(f, low_memory=False).set_index(['PTID', 'VISCODE_FIX'])
        n_before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < n_before:
            logging.warning('  dropped %d duplicate index rows in %s' % (n_before - len(df), os.path.basename(f)))
        df_list.append(df)
    df_list = pd.DataFrame(dict(df=df_list, path=flist), index=[len(e) for e in df_list])
    df_list.sort_index(ascending=False, inplace=True)

    # 가장 큰 CSV로 초기화
    init_df = df_list.df.iloc[0]
    logging.info('')
    logging.info('init csv: %s, current shape: %s' % (df_list.path.iloc[0], str(init_df.shape)))

    # 나머지 CSV 순차 병합
    for i, row in df_list.iloc[1:].iterrows():
        overlap_cols = list(set(init_df.columns).intersection(set(row.df.columns)))
        new_index = row.df.index.difference(init_df.index).unique()
        init_df = init_df.join(row.df.drop(overlap_cols, axis=1), how='outer')
        init_df = init_df[~init_df.index.duplicated(keep='first')]
        if len(new_index) > 0:
            init_df.loc[new_index] = row.df.loc[new_index]
        # 기존 행의 NaN 셀을 새 CSV 값으로 채움 (같은 prefix 공유 시 필요)
        if overlap_cols:
            existing_idx = row.df.index.intersection(init_df.index)
            if len(existing_idx) > 0:
                update_df = row.df.loc[existing_idx, overlap_cols]
                init_df.loc[existing_idx, overlap_cols] = \
                    init_df.loc[existing_idx, overlap_cols].combine_first(update_df)
        logging.info('merge csv: %s, current shape: %s' % (row.path, str(init_df.shape)))

    init_df.sort_index().to_csv(output_path)
    logging.info('Output saved at %s' % output_path)
    logging.info('-----------------------------------------------------\n')
