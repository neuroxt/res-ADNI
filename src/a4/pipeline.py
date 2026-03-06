"""
pipeline.py — A4/LEARN 메인 파이프라인

NII inventory + clinical table → 모달리티별 CSV → MERGED.csv

ADNI 매칭과의 핵심 차이:
- 날짜 매칭 불필요 (NII session_code로 직접 조인)
- regex 분류 불필요 (폴더명 = 모달리티)
- 시리즈 선택 불필요 (모달리티별 1개 primary .nii.gz)
"""

import os
import csv
import logging
from glob import glob

import pandas as pd

from .config import MODALITY_CONFIG, MERGE_EXCLUDE


def build_modality_csv(modality: str,
                       inventory: dict,
                       clinical: pd.DataFrame,
                       output_dir: str,
                       overwrite: bool = False) -> str:
    """단일 모달리티에 대해 inventory + clinical 조인 → {MOD}_unique.csv 저장.

    Args:
        modality: 모달리티 키 (e.g., 'T1', 'FBP')
        inventory: build_inventory() 결과
        clinical: build_clinical_table() 결과 (BID 인덱스)
        output_dir: 출력 디렉토리
        overwrite: 기존 파일 덮어쓰기

    Returns:
        저장된 CSV 경로 (레코드 없으면 '')
    """
    output_path = os.path.join(output_dir, '%s_unique.csv' % modality)
    if os.path.isfile(output_path) and not overwrite:
        logging.info('%s: already exists, skipping (use --overwrite)' % modality)
        return output_path

    mod_config = MODALITY_CONFIG.get(modality)
    if mod_config is None:
        logging.warning('Unknown modality: %s' % modality)
        return ''

    by_modality = inventory.get('by_modality', {})
    mod_data = by_modality.get(modality, {})

    if not mod_data:
        logging.warning('%s: no records in inventory' % modality)
        return ''

    # 레코드 수집
    records = []
    for bid, image_list in mod_data.items():
        for rec in image_list:
            row = {
                'BID': bid,
                'SESSION_CODE': rec['session'],
                'MODALITY': modality,
                'NII_PATH': rec.get('nii_path', ''),
            }

            # JSON sidecar 메타데이터
            json_meta = rec.get('json_meta', {})
            for field_name, value in json_meta.items():
                col = 'protocol/%s/%s' % (modality, field_name)
                row[col] = value

            # Clinical 데이터 조인 (BID 기준)
            if bid in clinical.index:
                clin_row = clinical.loc[bid]
                if isinstance(clin_row, pd.DataFrame):
                    clin_row = clin_row.iloc[0]
                for col_name, val in clin_row.items():
                    if pd.notna(val):
                        row[col_name] = val
                    else:
                        row[col_name] = ''

            records.append(row)

    if not records:
        logging.warning('%s: no records after processing' % modality)
        return ''

    df = pd.DataFrame(records)
    df.set_index(['BID', 'SESSION_CODE'], inplace=True)

    # 중복 인덱스 제거 (첫 번째 유지)
    n_before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    if len(df) < n_before:
        logging.warning('%s: dropped %d duplicate BID+SESSION rows' % (modality, n_before - len(df)))

    df.sort_index(inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, quoting=csv.QUOTE_NONNUMERIC)
    logging.info('%s: saved %d rows → %s' % (modality, len(df), output_path))

    return output_path


def unique_csv_merge(output_directory: str,
                     output_filename: str = 'MERGED.csv',
                     exclude_modalities: list = None):
    """*_unique.csv 파일들을 BID+SESSION_CODE 기준으로 병합.

    ADNI merge.py와 동일 로직이나 인덱스 키가 BID+SESSION_CODE.

    Args:
        output_directory: *_unique.csv 파일이 있는 디렉토리
        output_filename: 출력 파일명 (기본: MERGED.csv)
        exclude_modalities: 제외할 모달리티 (e.g., ['DWI'])
    """
    logging.info('-------------------- A4 Unique CSV Merge --------------------')
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
        df = pd.read_csv(f, low_memory=False).set_index(['BID', 'SESSION_CODE'])
        n_before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < n_before:
            logging.warning('  dropped %d duplicate index rows in %s' % (
                n_before - len(df), os.path.basename(f)))
        df_list.append(df)
    df_meta = pd.DataFrame(dict(df=df_list, path=flist), index=[len(e) for e in df_list])
    df_meta.sort_index(ascending=False, inplace=True)

    # 가장 큰 CSV로 초기화
    init_df = df_meta.df.iloc[0]
    logging.info('')
    logging.info('init csv: %s, current shape: %s' % (df_meta.path.iloc[0], str(init_df.shape)))

    # 나머지 CSV 순차 병합
    for i, row in df_meta.iloc[1:].iterrows():
        drop_columns = list(set(init_df.columns).intersection(set(row.df.columns)))
        new_index = row.df.index.difference(init_df.index).unique()
        init_df = init_df.join(row.df.drop(drop_columns, axis=1), how='outer')
        init_df = init_df[~init_df.index.duplicated(keep='first')]
        if len(new_index) > 0:
            init_df.loc[new_index] = row.df.loc[new_index]
        logging.info('merge csv: %s, current shape: %s' % (row.path, str(init_df.shape)))

    init_df.sort_index().to_csv(output_path)
    logging.info('Output saved at %s' % output_path)
    logging.info('-----------------------------------------------------\n')


def run_pipeline(inventory: dict,
                 clinical: pd.DataFrame,
                 output_dir: str,
                 modalities: list = None,
                 overwrite: bool = False,
                 skip_merge: bool = False):
    """전체 파이프라인 실행.

    Args:
        inventory: build_inventory() 결과
        clinical: build_clinical_table() 결과
        output_dir: 출력 디렉토리
        modalities: 처리할 모달리티 (None이면 인벤토리에 있는 전체)
        overwrite: 기존 결과 덮어쓰기
        skip_merge: merge 단계 건너뛰기
    """
    if modalities is None:
        # 인벤토리에 실제 데이터가 있는 모달리티만
        modalities = [m for m in MODALITY_CONFIG
                      if inventory.get('by_modality', {}).get(m)]

    logging.info('Processing %d modalities: %s' % (len(modalities), ', '.join(modalities)))

    for mod in modalities:
        build_modality_csv(mod, inventory, clinical, output_dir, overwrite=overwrite)

    if not skip_merge:
        unique_csv_merge(output_dir, exclude_modalities=MERGE_EXCLUDE)
