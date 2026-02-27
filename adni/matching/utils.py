"""
utils.py — 경로 추출, pydicom 읽기, 로깅 설정

ADNI.py의 _extract_*_from_path(), _setup_logger() 등에 대응
"""

import os
import re
import logging
import threading
import datetime

from .config import LOG_FORMAT, DCM_PROTOCOL_FIELDS

_logger_lock = threading.Lock()


# =============================================================================
# Logging
# =============================================================================

def setup_logger(log_path: str, level=logging.DEBUG):
    """로거 설정 (파일 + 콘솔). 스레드 안전."""
    logger = logging.getLogger()
    logger.setLevel(level)

    with _logger_lock:
        # 동일 log_path의 FileHandler가 이미 있으면 추가하지 않음
        has_fh = any(
            isinstance(h, logging.FileHandler) and
            getattr(h, 'baseFilename', None) == os.path.abspath(log_path)
            for h in logger.handlers
        )
        has_sh = any(
            isinstance(h, logging.StreamHandler) and
            not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )

        formatter = logging.Formatter(LOG_FORMAT)

        if not has_fh:
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if not has_sh:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(formatter)
            logger.addHandler(sh)


def reset_logger(log_path: str, level=logging.DEBUG):
    """로거 핸들러 초기화 후 재설정 (워커 프로세스용).

    스레드 환경에서는 no-op (root logger 공유).
    멀티프로세스 환경에서만 핸들러를 재설정.
    """
    if threading.current_thread() is not threading.main_thread():
        # 워커 스레드: root logger가 이미 설정됨, 재설정 불필요
        return
    logger = logging.getLogger()
    with _logger_lock:
        logger.handlers.clear()
    setup_logger(log_path, level)


# =============================================================================
# Path Extraction
# =============================================================================

def extract_date_from_path(path: str) -> str:
    """경로에서 YYYY-MM-DD 날짜 추출"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', path)
    return match.group(1) if match else ''


def extract_image_uid_from_path(path: str) -> str:
    """경로에서 ImageUID 추출 (/I{num}/ 또는 _I{num})"""
    # 디렉토리 패턴: /I{number}/ 또는 경로 끝 /I{number}
    match = re.search(r'/I(\d+)(?:/|$)', path)
    if match:
        return match.group(1)
    # 파일명 패턴: _I{number}
    match = re.search(r'_I(\d+)', os.path.basename(path))
    if match:
        return match.group(1)
    return ''


def extract_series_uid_from_path(path: str) -> str:
    """경로에서 SeriesUID 추출 (_S{num})"""
    match = re.search(r'_S(\d+)', path)
    return match.group(1) if match else ''


def extract_ptid_from_path(path: str) -> str:
    """경로에서 PTID 추출 (XXX_S_XXXX)"""
    match = re.search(r'(\d{3}_S_\d{4,5})', path)
    return match.group(1) if match else ''


def parse_date(date_str: str) -> datetime.datetime:
    """YYYY-MM-DD 문자열을 datetime으로 변환, 실패 시 None"""
    if isinstance(date_str, str) and date_str:
        try:
            return datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None
    return None


# =============================================================================
# pydicom DCM 읽기
# =============================================================================

def read_dicom_metadata(dcm_path: str) -> dict:
    """
    DCM 파일에서 TE/TR/TI/Flip Angle 추출 (pydicom)

    stop_before_pixels=True로 빠른 읽기.
    반환: {'TE': value, 'TR': value, 'TI': value, 'Flip Angle': value}
    """
    try:
        import pydicom
    except ImportError:
        logging.warning('pydicom not installed, skipping DCM metadata extraction')
        return {}

    result = {}
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        for tag, field_name in DCM_PROTOCOL_FIELDS.items():
            elem = ds.get(tag, None)
            if elem is not None:
                result[field_name] = str(elem.value)
            else:
                result[field_name] = ''
    except Exception as e:
        logging.debug('Failed to read DCM %s: %s' % (dcm_path, str(e)))

    return result


def find_dcm_file(dcm_dir: str) -> str:
    """DCM 디렉토리에서 첫 번째 .dcm 파일 경로 반환"""
    if not dcm_dir or not os.path.isdir(dcm_dir):
        return ''
    for f in os.listdir(dcm_dir):
        if f.lower().endswith('.dcm'):
            return os.path.join(dcm_dir, f)
    # .dcm 확장자 없는 경우 (ADNI에서 간혹 있음) 첫 번째 파일 시도
    files = [f for f in os.listdir(dcm_dir) if os.path.isfile(os.path.join(dcm_dir, f))]
    if files:
        return os.path.join(dcm_dir, files[0])
    return ''


# =============================================================================
# Demographics Helpers
# =============================================================================

def calc_age_from_birth(birth_date_str: str, scan_date_str: str) -> float:
    """생년월일과 촬영일로부터 나이 계산"""
    birth = parse_date(birth_date_str)
    scan = parse_date(scan_date_str)
    if birth and scan:
        delta = scan - birth
        return round(delta.days / 365.25, 1)
    return ''


def map_ptgender(gender_str: str) -> str:
    """PTGENDER → ADNI.py subjectSex 형식"""
    if not gender_str or not isinstance(gender_str, str):
        return ''
    g = gender_str.strip().upper()
    if g in ('MALE', 'M'):
        return 'M'
    elif g in ('FEMALE', 'F'):
        return 'F'
    return gender_str
