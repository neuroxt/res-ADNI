"""
matching.py — 핵심 매칭 로직 (ADNI.py 재구현, XML 제거)

ADNI.py 대응:
  - _nearest_adinmerge()  → nearest_adnimerge()
  - _calc_viscode()       → calc_viscode()
  - _demo_matching()      → match_image()       (XML → 경로+MRIQC+DCM+demographics)
  - subj_matching()       → match_subject()
  - adnimerge_matching()  → match_modality()
"""

import os
import csv
import fnmatch
import warnings
import logging
import datetime

import numpy as np
import pandas as pd
from glob import glob
from joblib import Parallel, delayed

from .config import (
    ADNIMERGE_MATCHING_TARGET_COLUMN,
    ADNIMERGE_VISCODE_TARGET_COLUMN,
    ADNIMERGE_NO_MATCHING_RESET_COLUMN,
    MONTH_KEYS,
    MONTHS,
    MRIQC_PROTOCOL_FIELDS,
    DCM_PROTOCOL_FIELDS,
    format_apoe_genotype,
)
from .utils import (
    extract_date_from_path,
    extract_image_uid_from_path,
    extract_series_uid_from_path,
    extract_ptid_from_path,
    read_dicom_metadata,
    find_dcm_file,
    calc_age_from_birth,
    map_ptgender,
    parse_date,
    reset_logger,
)


# =============================================================================
# Core Matching Functions
# =============================================================================

def nearest_adnimerge(subj_adnimerge: pd.DataFrame, aqutime: datetime.datetime,
                      threshold: int) -> pd.DataFrame:
    """
    ADNIMERGE에서 촬영일에 가장 가까운 행 반환

    ADNI.py _nearest_adinmerge()와 동일 로직:
    - EXAMDATE_datetime 컬럼과의 시간차 계산
    - threshold(일) 초과 시 방문별 컬럼 비움
    """
    if len(subj_adnimerge):
        subj_adnimerge = subj_adnimerge.copy()
        target_column = ADNIMERGE_MATCHING_TARGET_COLUMN
        delta = aqutime - subj_adnimerge[target_column]
        subj_adnimerge[target_column] = np.abs(delta)
        subj_adnimerge['DAYSDELTA'] = delta.dt.days
        nn_idx = subj_adnimerge[target_column].argmin()
        nn_row = subj_adnimerge.iloc[nn_idx:nn_idx + 1]
        if pd.Timedelta(threshold, unit='d') < subj_adnimerge[target_column].iloc[nn_idx]:
            nn_row[ADNIMERGE_NO_MATCHING_RESET_COLUMN] = ''
    else:
        nn_row = pd.DataFrame([pd.Series([], dtype=object)])
    return nn_row


def calc_viscode(timedelta, threshold: int) -> str:
    """
    daysdelta를 VISCODE_FIX로 변환

    ADNI.py _calc_viscode()와 동일 로직:
    - daysdelta = AQUDATE - EXAMDATE_bl
    - MONTH_KEYS에서 가장 가까운 표준 방문 시점 선택
    - threshold 초과 시 'error' 반환
    """
    daysdelta = timedelta.total_seconds() / 86400
    delta = np.abs(MONTH_KEYS - daysdelta)
    idx = np.argmin(delta)
    if delta[idx] > threshold:
        return 'error'
    key = MONTH_KEYS[idx]
    return MONTHS[key]


# =============================================================================
# Image-Level Matching (replaces _demo_matching + _demo_matching_from_dicom)
# =============================================================================

def match_image(subj_adnimerge: pd.DataFrame, image_path: str,
                threshold: int, modality: str,
                mriqc_index: dict = None,
                dcm_inventory: dict = None,
                _birth_row=None, _apoe_row=None,
                log_path: str = '') -> pd.DataFrame:
    """
    단일 이미지에 대한 ADNIMERGE 매칭 + demographics + protocol

    XML 대신:
    - 날짜: 경로에서 추출
    - I_{mod}, S_{mod}: 경로에서 추출
    - researchGroup: ADNIMERGE DX (방문별 진단)
    - subjectAge: birth_dates.csv + 촬영일 역산
    - subjectSex: ADNIMERGE PTGENDER
    - Apoe: APOERES GENOTYPE
    - 프로토콜: MRIQC (LONIImage=ImageUID 조인) + DCM pydicom (TE/TR/TI)

    최적화:
    - mriqc_index: {image_uid_str: Series} dict (#1)
    - _birth_row/_apoe_row: subject 레벨에서 사전 조회된 행 (#3)
    """
    if log_path:
        reset_logger(log_path)

    # 1) 날짜 추출 (경로에서)
    aqudate = extract_date_from_path(image_path)
    if not aqudate:
        logging.warning('\tNo date in path: %s' % image_path)
        return None

    aqutime = datetime.datetime.strptime(aqudate, '%Y-%m-%d')

    # 2) ImageUID, SeriesUID 추출 (경로에서)
    image_uid = extract_image_uid_from_path(image_path)
    series_uid = extract_series_uid_from_path(image_path)

    # 3) ADNIMERGE 매칭 (동일 로직)
    row = nearest_adnimerge(subj_adnimerge, aqutime, threshold)
    if not pd.isna(row.iloc[0].get(ADNIMERGE_VISCODE_TARGET_COLUMN, pd.NaT)):
        row['VISCODE_FIX'] = calc_viscode(
            aqutime - row.iloc[0][ADNIMERGE_VISCODE_TARGET_COLUMN], threshold)
    else:
        row['VISCODE_FIX'] = 'error'
        ptid = extract_ptid_from_path(image_path)
        if ptid:
            row['PTID'] = ptid

    # 4) 기본 정보
    row['MODALITY'] = modality
    row['AQUDATE_%s' % modality] = aqudate
    row['visitIdentifier_%s' % modality] = ''  # XML에만 존재, 빈 값
    row['S_%s' % modality] = series_uid
    row['I_%s' % modality] = image_uid

    # 5) Demographics (XML → ADNIMERGE + APOERES + birth_dates)
    ptid = row.iloc[0].get('PTID', '')
    if not ptid:
        ptid = extract_ptid_from_path(image_path)

    # researchGroup: ADNIMERGE DX (방문별 진단, XML의 enrollment group보다 정확)
    row['researchGroup'] = row.iloc[0].get('DX', '')

    # subjectAge: 사전 조회된 birth_row 사용 (#3 최적화)
    if _birth_row is not None and ptid:
        row['subjectAge'] = calc_age_from_birth(
            str(_birth_row['est_birth_date']), aqudate)
    else:
        row['subjectAge'] = ''

    # subjectSex: ADNIMERGE PTGENDER
    row['subjectSex'] = map_ptgender(str(row.iloc[0].get('PTGENDER', '')))

    # weightKg: 대체 불가
    row['weightKg'] = ''

    # Apoe: 사전 조회된 apoe_row 사용 (#3 최적화)
    if _apoe_row is not None and ptid:
        row['Apoe'] = format_apoe_genotype(str(_apoe_row.get('GENOTYPE', '')))
    else:
        row['Apoe'] = ''

    # 6) Protocol: MRIQC (dict) + DCM
    _fill_protocol(row, modality, image_uid, mriqc_index, dcm_inventory)

    # 7) 이미지 경로
    row['%s_image_path' % modality] = image_path

    return row


def _fill_protocol(row: pd.DataFrame, modality: str, image_uid: str,
                   mriqc_index: dict = None,
                   dcm_inventory: dict = None):
    """프로토콜 필드 채우기: MRIQC (dict) + DCM (inventory 우선, pydicom fallback)

    최적화:
    - mriqc_index: {image_uid_str: Series} dict O(1) 조회 (#1)
    - inventory에 dcm_TE/TR/TI/FlipAngle 있으면 NFS I/O 없이 사용 (#5+#6)
    """

    # 기본값: 모두 빈 문자열
    protocol_prefix = 'protocol/%s/' % modality
    all_fields = list(MRIQC_PROTOCOL_FIELDS.values()) + list(DCM_PROTOCOL_FIELDS.values())
    # Pixel Spacing → Pixel Spacing X/Y 로 분리 출력
    if 'Pixel Spacing' in all_fields:
        all_fields.remove('Pixel Spacing')
        all_fields.extend(['Pixel Spacing X', 'Pixel Spacing Y'])
    for field_name in all_fields:
        col = protocol_prefix + field_name
        if col not in row.columns:
            row[col] = ''

    # MRIQC: dict O(1) 조회 (#1 최적화)
    if mriqc_index and image_uid:
        mriqc_row = mriqc_index.get(str(image_uid))
        if mriqc_row is not None:
            for mriqc_col, output_name in MRIQC_PROTOCOL_FIELDS.items():
                col = protocol_prefix + output_name
                val = mriqc_row.get(mriqc_col, '')
                row[col] = '' if pd.isna(val) else str(val)

    # DCM 메타: inventory에서 먼저 조회 (NFS I/O 제거 #5+#6)
    if dcm_inventory and image_uid:
        dcm_record = dcm_inventory.get('by_image_uid', {}).get(str(image_uid), {})
        if not isinstance(dcm_record, dict):
            dcm_record = {}

        # 인벤토리에 DCM 메타가 실제로 추출되었는지 확인
        # (빈 문자열 = 추출 실패 가능 → fallback 필요)
        has_inv_dcm = any(dcm_record.get(k) for k in
                          ('dcm_TE', 'dcm_TR', 'dcm_FlipAngle'))
        if has_inv_dcm:
            for field, inv_key in [('TE', 'dcm_TE'), ('TR', 'dcm_TR'),
                                    ('TI', 'dcm_TI'), ('Flip Angle', 'dcm_FlipAngle'),
                                    ('Pulse Sequence', 'dcm_PulseSequence'),
                                    ('Matrix X', 'dcm_MatrixX'),
                                    ('Matrix Y', 'dcm_MatrixY')]:
                val = dcm_record.get(inv_key, '')
                col = protocol_prefix + field
                if val:
                    row[col] = val
            # Pixel Spacing: split into X/Y
            ps = dcm_record.get('dcm_PixelSpacing', '')
            if ps:
                parts = str(ps).replace('[', '').replace(']', '').split('\\')
                if len(parts) >= 2:
                    row[protocol_prefix + 'Pixel Spacing X'] = parts[0].strip()
                    row[protocol_prefix + 'Pixel Spacing Y'] = parts[1].strip()
                elif len(parts) == 1:
                    row[protocol_prefix + 'Pixel Spacing X'] = parts[0].strip()
                    row[protocol_prefix + 'Pixel Spacing Y'] = parts[0].strip()
            # Matrix Z: NumberOfFrames 우선, 없으면 dcm_count fallback
            matrix_z = dcm_record.get('dcm_MatrixZ', '')
            if not matrix_z:
                matrix_z = dcm_record.get('dcm_count', '')
            if matrix_z:
                row[protocol_prefix + 'Matrix Z'] = str(matrix_z)
        else:
            # 구형 인벤토리 또는 추출 실패 fallback → pydicom NFS 읽기
            dcm_dir = dcm_record.get('dcm_path', '')
            if dcm_dir:
                dcm_file = find_dcm_file(dcm_dir)
                if dcm_file:
                    dcm_meta = read_dicom_metadata(dcm_file)
                    for field_name, value in dcm_meta.items():
                        if field_name == 'Pixel Spacing' and value:
                            parts = str(value).replace('[', '').replace(']', '').split('\\')
                            row[protocol_prefix + 'Pixel Spacing X'] = parts[0].strip()
                            row[protocol_prefix + 'Pixel Spacing Y'] = parts[-1].strip()
                            continue
                        if field_name == 'Matrix Z' and not value:
                            value = dcm_record.get('dcm_count', '')
                        col = protocol_prefix + field_name
                        if value:
                            row[col] = value


# =============================================================================
# Image Collection (NII + DCM)
# =============================================================================

def collect_images(subj_dir: str, regex: list, file_type: str = 'nii',
                   exclude_regex: list = None) -> list:
    """
    모달리티에 맞는 이미지(NII 파일) 또는 시리즈 폴더(DCM) 수집

    DCM 구조: {subj}/{protocol}/{date}/I{UID}/*.dcm
    NII 구조: {subj}/{protocol}/{date}/I{UID}/*.nii*

    Args:
        subj_dir: 피험자 디렉토리
        regex: 프로토콜 매칭 glob 패턴 리스트
        file_type: 'nii' (NII 파일 반환) 또는 'dcm' (DCM 시리즈 폴더 반환)
        exclude_regex: 제외할 프로토콜 glob 패턴 리스트

    Returns:
        NII: 파일 경로 리스트
        DCM: 시리즈 폴더 경로 리스트 (I{UID} 디렉토리)
    """
    if isinstance(regex, str):
        regex = [regex]

    image_list = []

    if file_type == 'nii':
        # 기존 로직: {subj}/{protocol}/{date}/I{UID}/*.nii*
        for re_pattern in regex:
            image_list += glob(os.path.join(subj_dir, re_pattern, '*', '*', '*.nii*'))
    else:  # dcm
        # DCM: {subj}/{protocol}/{date}/I{UID}/ → 폴더 자체가 "이미지"
        for re_pattern in regex:
            # protocol 매칭 → 그 아래 date/I{UID} 폴더들
            protocol_dirs = glob(os.path.join(subj_dir, re_pattern))
            for proto_dir in protocol_dirs:
                if not os.path.isdir(proto_dir):
                    continue
                # date 폴더들
                for date_dir in glob(os.path.join(proto_dir, '*')):
                    if not os.path.isdir(date_dir):
                        continue
                    # I{UID} 폴더들
                    for uid_dir in glob(os.path.join(date_dir, 'I*')):
                        if os.path.isdir(uid_dir):
                            # DCM 파일 존재 확인
                            dcm_files = glob(os.path.join(uid_dir, '*.dcm'))
                            if dcm_files:
                                image_list.append(uid_dir)

    # exclude_regex 적용 (프로토콜 레벨에서 필터링)
    if exclude_regex:
        if isinstance(exclude_regex, str):
            exclude_regex = [exclude_regex]
        filtered = []
        for img in image_list:
            # 프로토콜은 경로에서 PTID 바로 다음 디렉토리
            # {base}/{PTID}/{protocol}/{date}/I{UID}[/file.nii]
            # DCM: img = .../{protocol}/{date}/I{UID}
            # NII: img = .../{protocol}/{date}/I{UID}/file.nii
            if file_type == 'nii':
                # img 경로: .../protocol/date/IUID/file.nii → protocol은 4단계 위
                protocol_name = img.split(os.sep)[-4]
            else:
                # img 경로: .../protocol/date/IUID → protocol은 3단계 위
                protocol_name = img.split(os.sep)[-3]

            excluded = False
            for ex_pat in exclude_regex:
                if fnmatch.fnmatch(protocol_name, ex_pat):
                    excluded = True
                    break
            if not excluded:
                filtered.append(img)
        image_list = filtered

    return sorted(set(image_list))


# =============================================================================
# Subject-Level Matching (replaces subj_matching)
# =============================================================================

def match_subject(output_directory: str, subj_dir: str,
                  adnimerge_df: pd.DataFrame,
                  threshold: int, modality: str,
                  regex: list,
                  file_type: str = 'nii',
                  exclude_regex: list = None,
                  create_symlinks: bool = False,
                  mriqc_df: pd.DataFrame = None,
                  apoeres_df: pd.DataFrame = None,
                  birth_dates_df: pd.DataFrame = None,
                  dcm_inventory: dict = None,
                  log_path: str = '') -> pd.DataFrame:
    """
    단일 피험자의 모든 이미지 매칭

    ADNI.py subj_matching()과 동일 로직:
    - regex로 이미지 파일(NII) 또는 시리즈 폴더(DCM) 수집
    - 각 이미지에 match_image() 적용
    - VISCODE_FIX + EXAMDATE + ImageUID 정렬 (closest & largest)
    - 심볼릭 링크 생성 (NII: 파일, DCM: 디렉토리)
    """
    warnings.filterwarnings(action='ignore')
    if log_path:
        reset_logger(log_path)

    ptid = os.path.split(subj_dir)[1]
    subj_adnimerge = adnimerge_df.query("PTID=='%s'" % ptid)

    # collect_images()로 NII/DCM 통합 수집
    image_list = collect_images(subj_dir, regex, file_type=file_type,
                                exclude_regex=exclude_regex)

    # 각 이미지 매칭
    subj_result = []
    for image in image_list:
        result = match_image(
            subj_adnimerge, image, threshold, modality,
            mriqc_df=mriqc_df,
            apoeres_df=apoeres_df,
            birth_dates_df=birth_dates_df,
            dcm_inventory=dcm_inventory,
            log_path=log_path,
        )
        if result is not None:
            subj_result.append(result)

    if not subj_result:
        return None

    subj_result = pd.concat(subj_result, ignore_index=True)

    # EXAMDATE_datetime 기반 정렬 (closest & largest ImageUID)
    if ADNIMERGE_MATCHING_TARGET_COLUMN in subj_result.columns:
        subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] = subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN].fillna(
            pd.Timedelta(99999, unit='d'))
        subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] *= -1
        subj_result = subj_result.sort_values(
            by=['VISCODE_FIX', ADNIMERGE_MATCHING_TARGET_COLUMN, 'I_%s' % modality])
    else:
        subj_result = subj_result.sort_values(
            by=['VISCODE_FIX', 'I_%s' % modality])

    # 심볼릭 링크 생성 (옵션, 기본 꺼짐 — MERGED.csv만 필요하면 불필요)
    if create_symlinks:
        for _, row in subj_result.iterrows():
            directory = os.path.join(output_directory, 'image', ptid, str(row['VISCODE_FIX']), modality)
            os.makedirs(directory, exist_ok=True)
            img_path = row['%s_image_path' % modality]
            link_path = os.path.join(directory, os.path.basename(img_path))
            if not os.path.exists(link_path):
                try:
                    os.symlink(img_path, link_path)
                except OSError:
                    pass

    logging.debug('\t%s: %s visit points and %s images detected' % (
        ptid, len(subj_result['VISCODE_FIX'].unique()), len(subj_result)))
    return subj_result


# =============================================================================
# Subject-Level Matching from Inventory (v2)
# =============================================================================

def match_subject_from_inventory(output_directory: str, ptid: str,
                                  image_records: list,
                                  adnimerge_groups: dict,
                                  threshold: int, modality: str,
                                  create_symlinks: bool = False,
                                  mriqc_index: dict = None,
                                  apoeres_index: dict = None,
                                  birth_index: dict = None,
                                  dcm_inventory: dict = None,
                                  log_path: str = '') -> pd.DataFrame:
    """
    인벤토리 기반 단일 피험자 매칭 (파일시스템 I/O 없음)

    match_subject()와 동일 로직이나 collect_images() 대신
    인벤토리의 image_records를 직접 사용.

    최적화:
    - adnimerge_groups: {ptid: DataFrame} groupby 인덱스 (#2)
    - mriqc_index/apoeres_index/birth_index: dict O(1) 조회 (#1)
    - demographics를 subject 레벨에서 1회 조회 (#3)
    """
    warnings.filterwarnings(action='ignore')
    if log_path:
        reset_logger(log_path)

    subj_adnimerge = adnimerge_groups.get(ptid, pd.DataFrame())

    # Subject 레벨에서 demographics 1회 조회 (#3 최적화)
    birth_row = birth_index.get(ptid) if birth_index else None
    apoe_row = apoeres_index.get(ptid) if apoeres_index else None

    # 각 이미지 레코드에 대해 match_image() 호출
    subj_result = []
    for rec in image_records:
        image_path = rec.get('dcm_path', '')
        if not image_path:
            continue
        result = match_image(
            subj_adnimerge, image_path, threshold, modality,
            mriqc_index=mriqc_index,
            dcm_inventory=dcm_inventory,
            _birth_row=birth_row,
            _apoe_row=apoe_row,
            log_path=log_path,
        )
        if result is not None:
            subj_result.append(result)

    if not subj_result:
        return None

    subj_result = pd.concat(subj_result, ignore_index=True)

    # EXAMDATE_datetime 기반 정렬 (closest & largest ImageUID)
    if ADNIMERGE_MATCHING_TARGET_COLUMN in subj_result.columns:
        subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] = subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN].fillna(
            pd.Timedelta(99999, unit='d'))
        subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] *= -1
        subj_result = subj_result.sort_values(
            by=['VISCODE_FIX', ADNIMERGE_MATCHING_TARGET_COLUMN, 'I_%s' % modality])
    else:
        subj_result = subj_result.sort_values(
            by=['VISCODE_FIX', 'I_%s' % modality])

    # 심볼릭 링크 생성
    if create_symlinks:
        for _, row in subj_result.iterrows():
            directory = os.path.join(output_directory, 'image', ptid, str(row['VISCODE_FIX']), modality)
            os.makedirs(directory, exist_ok=True)
            img_path = row['%s_image_path' % modality]
            link_path = os.path.join(directory, os.path.basename(img_path))
            if not os.path.exists(link_path):
                try:
                    os.symlink(img_path, link_path)
                except OSError:
                    pass

    logging.debug('\t%s: %s visit points and %s images detected' % (
        ptid, len(subj_result['VISCODE_FIX'].unique()), len(subj_result)))
    return subj_result


# =============================================================================
# Modality-Level Matching (replaces adnimerge_matching)
# =============================================================================

def match_modality(adnimerge_csv: str,
                   output_directory: str, threshold: int,
                   modality: str,
                   dcm_inventory: dict,
                   create_symlinks: bool = False,
                   n_jobs: int = 8, overwrite: bool = False,
                   mriqc_csv: str = None,
                   apoeres_csv: str = None,
                   birth_dates_csv: str = None,
                   log_path: str = ''):
    """
    모달리티별 전체 매칭 실행 (v2: 인벤토리 전용)

    인벤토리의 by_modality[mod]에서 피험자 목록 + 이미지 레코드를
    직접 조회하여 파일시스템 스캔 없이 매칭.
    """
    modality = modality.upper()

    logging.info('-------------------- %s matching --------------------' % modality)
    logging.info('adnimerge=%s' % adnimerge_csv)
    logging.info('threshold=%s' % threshold)

    output_path_all = os.path.join(output_directory, '%s_all.csv' % modality)
    output_path_unique = os.path.join(output_directory, '%s_unique.csv' % modality)

    if not overwrite and (os.path.isfile(output_path_all) or os.path.isfile(output_path_unique)):
        logging.error('File exists: %s or %s. Set overwrite=True' % (output_path_all, output_path_unique))
        return

    # 인벤토리에서 모달리티별 피험자 조회
    mod_data = dcm_inventory.get('by_modality', {}).get(modality, {})
    if not mod_data:
        logging.warning('No inventory data for %s, skipping' % modality)
        return

    logging.info('Inventory: %d subjects for %s' % (len(mod_data), modality))

    # 임시 컬럼 (나중에 제거)
    temp_columns = [ADNIMERGE_MATCHING_TARGET_COLUMN, ADNIMERGE_VISCODE_TARGET_COLUMN]

    # ADNIMERGE 로드
    adnimerge_df = pd.read_csv(adnimerge_csv)
    adnimerge_df[ADNIMERGE_MATCHING_TARGET_COLUMN] = pd.to_datetime(
        adnimerge_df['EXAMDATE'], format='%Y-%m-%d', errors='coerce')
    adnimerge_df[ADNIMERGE_VISCODE_TARGET_COLUMN] = pd.to_datetime(
        adnimerge_df['EXAMDATE_bl'], format='%Y-%m-%d', errors='coerce')

    # 보조 테이블 로드 + O(1) dict 인덱싱 (#1 최적화)
    mriqc_df = pd.read_csv(mriqc_csv) if mriqc_csv and os.path.isfile(mriqc_csv) else None
    apoeres_df = pd.read_csv(apoeres_csv) if apoeres_csv and os.path.isfile(apoeres_csv) else None
    birth_dates_df = pd.read_csv(birth_dates_csv) if birth_dates_csv and os.path.isfile(birth_dates_csv) else None

    # MRIQC: {image_uid_str: Series} — 90K행 → dict O(1) 조회
    mriqc_index = {}
    if mriqc_df is not None:
        logging.info('MRIQC loaded: %d rows' % len(mriqc_df))
        for _, row in mriqc_df.iterrows():
            loni_val = row['LONIImage']
            key = str(int(loni_val)) if pd.notna(loni_val) else ''
            if key:
                mriqc_index[key] = row

    # APOERES: {ptid: Series}
    apoeres_index = {}
    if apoeres_df is not None:
        logging.info('APOERES loaded: %d rows' % len(apoeres_df))
        for _, row in apoeres_df.iterrows():
            apoeres_index[row['PTID']] = row

    # birth_dates: {ptid: Series}
    birth_index = {}
    if birth_dates_df is not None:
        logging.info('birth_dates loaded: %d rows' % len(birth_dates_df))
        for _, row in birth_dates_df.iterrows():
            birth_index[row['PTID']] = row

    # ADNIMERGE groupby 인덱싱 (#2 최적화)
    adnimerge_groups = {ptid: group for ptid, group in adnimerge_df.groupby('PTID')}

    # 피험자 목록 (인벤토리에서)
    ptid_list = sorted(mod_data.keys())
    logging.info('Total %d subjects detected' % len(ptid_list))

    # 병렬 매칭 (인벤토리 기반, threading 백엔드 #4 최적화)
    result = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(match_subject_from_inventory)(
            output_directory, ptid, mod_data[ptid],
            adnimerge_groups, threshold, modality,
            create_symlinks=create_symlinks,
            mriqc_index=mriqc_index,
            apoeres_index=apoeres_index,
            birth_index=birth_index,
            dcm_inventory=dcm_inventory,
            log_path=log_path,
        ) for ptid in ptid_list
    )
    result = [e for e in result if e is not None]

    if not result:
        logging.warning('No results for %s matching' % modality)
        return

    result = pd.concat(result, ignore_index=True)

    # 임시 컬럼 제거
    for col in temp_columns:
        if col in result.columns:
            result = result.drop(col, axis=1)

    # 저장 (MRIQC 필드에 쉼표 포함 가능 → quoting 필수)
    result.to_csv(output_path_all, index=False, quoting=csv.QUOTE_NONNUMERIC)
    result.drop_duplicates(
        subset=['PTID', 'VISCODE_FIX'], keep='last'
    ).query("VISCODE_FIX != 'error'").to_csv(output_path_unique, index=False, quoting=csv.QUOTE_NONNUMERIC)

    logging.info('Total %d subjects and %d visit points' % (len(result['PTID'].unique()), len(result)))
    logging.info('Output saved at %s' % output_path_all)
    logging.info('Output saved at %s' % output_path_unique)
    logging.info('-----------------------------------------------------\n')


# =============================================================================
# UCBerkeley PET Quantification Attach (replaces attach_ucberkeley)
# =============================================================================

def attach_ucberkeley(matching_csv: str, ucberkeley_csv: str,
                      modality: str, date_threshold: int = 30,
                      ucb_date_col: str = 'SCANDATE',
                      tracer_filter: list = None,
                      column_prefix: str = ''):
    """
    UCBerkeley PET quantification 결과를 매칭 CSV에 left join

    ADNI.py attach_ucberkeley()와 동일 알고리즘:
    1. (RID, 촬영일) 기준 정확 매칭
    2. 비매칭 건: 동일 RID 내 threshold 이내 최근접 날짜 리매핑
    3. 리매핑된 인덱스로 left join

    Args:
        matching_csv: 모달리티 매칭 결과 *_unique.csv
        ucberkeley_csv: UCBerkeley 테이블 CSV
        modality: 모달리티명 (e.g., 'AV45_6MM')
        date_threshold: 날짜 매칭 허용 오차 (일, 기본 30)
        ucb_date_col: UCBerkeley 날짜 컬럼 ('SCANDATE' 또는 'EXAMDATE')
        tracer_filter: TRACER 컬럼 필터 (e.g., ['FBP']). None이면 필터 안 함
        column_prefix: UCBerkeley 컬럼 접두사 (e.g., 'UCB_AMY_')
    """
    logging.info('-------------------- attach UCBerkeley (%s) --------------------' % modality)
    logging.info('matching_csv=%s' % matching_csv)
    logging.info('ucberkeley_csv=%s' % ucberkeley_csv)
    logging.info('date_threshold=%d days, ucb_date_col=%s' % (date_threshold, ucb_date_col))

    if not os.path.isfile(matching_csv):
        logging.warning('Matching CSV not found: %s, skipping' % matching_csv)
        return
    if not os.path.isfile(ucberkeley_csv):
        logging.warning('UCBerkeley CSV not found: %s, skipping' % ucberkeley_csv)
        return

    result = pd.read_csv(matching_csv)
    ucb = pd.read_csv(ucberkeley_csv)

    if len(result) == 0 or len(ucb) == 0:
        logging.warning('Empty CSV, skipping UCBerkeley attach')
        return

    # Tracer filter (e.g., AMY 테이블에서 FBP/FBB 구분)
    if tracer_filter and 'TRACER' in ucb.columns:
        ucb = ucb[ucb['TRACER'].isin(tracer_filter)].copy()
        logging.info('Tracer filter %s: %d rows remaining' % (tracer_filter, len(ucb)))
        if len(ucb) == 0:
            logging.warning('No rows after tracer filter, skipping')
            return

    aqudate_col = 'AQUDATE_%s' % modality

    # -- Index keys: (RID string, date string) --
    # Original: PTID "002_S_0413" → RID "413"
    def _ptid_to_rid(ptid):
        if pd.isna(ptid):
            return ''
        return str(ptid).split('_S_')[-1].lstrip('0')

    def _rid_to_str(val):
        if pd.isna(val):
            return ''
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)

    result['_ind1'] = result['PTID'].apply(_ptid_to_rid)
    result['_ind2'] = result[aqudate_col]

    ucb['_ind1'] = ucb['RID'].apply(_rid_to_str)
    ucb['_ind2'] = ucb[ucb_date_col]

    # -- Set index --
    result = result.set_index(['_ind1', '_ind2'], drop=True)

    # UCBerkeley: drop=False → _ind1, _ind2를 컬럼에 유지 (리매핑용)
    ucb = ucb.set_index(['_ind1', '_ind2'], drop=False)

    # 관리 컬럼 제거 (매칭 결과에 이미 존재)
    ucb_admin_drop = [c for c in ['RID', 'PTID', 'VISCODE', 'VISCODE2',
                                   'ORIGPROT', 'SITEID', ucb_date_col, 'update_stamp']
                      if c in ucb.columns]
    ucb = ucb.drop(ucb_admin_drop, axis=1)

    # 결과에 이미 존재하는 UCBerkeley 컬럼 제거 (UCB 값 우선)
    overlap = list(set(result.columns) & set(ucb.columns) - {'_ind1', '_ind2'})
    if overlap:
        result = result.drop(overlap, axis=1)

    # -- 비매칭 인덱스 탐색 --
    left_diff = result.index.difference(ucb.index)    # result에 있고 UCB에 없는 것
    right_diff = ucb.index.difference(result.index)    # UCB에 있고 result에 없는 것

    logging.info('%d exact matches, %d non-matching result rows' % (
        len(result.index.intersection(ucb.index)), len(left_diff)))

    # -- Fuzzy date matching (동일 RID 내 최근접 날짜) --
    # UCB 비매칭 행을 RID별로 그룹화
    ucb_by_rid = {}
    for multi_idx in right_diff:
        rid, date_str = multi_idx
        try:
            dt = datetime.datetime.strptime(str(date_str), '%Y-%m-%d')
            ucb_by_rid.setdefault(rid, []).append((dt, multi_idx))
        except (ValueError, TypeError):
            pass

    fuzzy_count = 0
    remap = {}  # {ucb_original_multi_index: new_date_str}

    for multi_idx in left_diff:
        rid, date_str = multi_idx
        try:
            aqutime = datetime.datetime.strptime(str(date_str), '%Y-%m-%d')
        except (ValueError, TypeError):
            continue

        candidates = ucb_by_rid.get(rid, [])
        if not candidates:
            logging.debug('\tMatching (%s, %s) failed: no UCBerkeley data for RID' % (rid, date_str))
            continue

        best_dt, best_multi_idx = min(candidates, key=lambda c: abs((aqutime - c[0]).days))
        best_delta = abs((aqutime - best_dt).days)

        if best_delta <= date_threshold:
            remap[best_multi_idx] = date_str
            fuzzy_count += 1
            logging.debug('\tMatched (%s, %s) <-> (%s, %s) delta=%d days' % (
                rid, date_str, best_multi_idx[0], best_multi_idx[1], best_delta))
        else:
            logging.debug('\tMatching (%s, %s) failed: nearest=%d days > threshold=%d' % (
                rid, date_str, best_delta, date_threshold))

    logging.info('%d fuzzy matches within %d-day threshold' % (fuzzy_count, date_threshold))

    # -- UCBerkeley 날짜 리매핑 (원본 알고리즘 핵심) --
    # _ind2 컬럼값을 result 날짜로 변경 → 재인덱싱 시 join 가능
    for old_idx, new_date in remap.items():
        ucb.at[old_idx, '_ind2'] = new_date

    # 리매핑된 컬럼값으로 재인덱싱
    ucb = ucb.reset_index(drop=True).set_index(['_ind1', '_ind2'], drop=True)

    # -- Column prefix --
    if column_prefix:
        ucb = ucb.rename(columns={c: column_prefix + c for c in ucb.columns})

    # -- Left join --
    result = result.join(ucb, how='left')

    # -- 저장 --
    result.to_csv(matching_csv, index=False)
    logging.info('Output saved at %s (%d rows, %d columns)' % (
        matching_csv, len(result), len(result.columns)))
    logging.info('-----------------------------------------------------\n')
