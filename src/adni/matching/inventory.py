"""
inventory_improved.py — 최적화된 DCM inventory 생성 + 모달리티 분류

inventory.py와 동일한 출력 형식, 병렬화 + I/O 최적화 적용:
  #1: 구조화된 3단계 os.scandir 순회 (os.walk 대체)
  #2: os.scandir 단일 패스 (glob+sorted 대체)
  #3: 단일 패스 경로 파싱 (반복 regex 대체)
  #4: PTID 단위 병렬화 (ThreadPoolExecutor)
  #5: 소스 단위 병렬화 (ThreadPoolExecutor)
  #6: classify_series 소스 디스패치 테이블
  #7: tqdm 진행률 표시
"""

import os
import re
import json
import fnmatch
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import NFS_BASE, DCM_SOURCES, MODALITY_CONFIG, DCM_PROTOCOL_FIELDS

# Optional tqdm (#7)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Compiled regexes (module-level, compiled once) (#3)
_PTID_RE = re.compile(r'\d{3}_S_\d{4,5}')
_IUID_FILENAME_RE = re.compile(r'_I(\d+)')
_SUID_RE = re.compile(r'_S(\d+)')
_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


# =============================================================================
# Progress Fallback (when tqdm unavailable)
# =============================================================================

class _ProgressFallback:
    """Simple progress counter when tqdm is not available."""

    def __init__(self, total, desc=''):
        self.total = total
        self.desc = desc
        self.n = 0
        self._lock = threading.Lock()
        self._last_pct = 0

    def update(self, n=1):
        with self._lock:
            self.n += n
            pct = (self.n * 100) // self.total if self.total else 0
            if pct >= self._last_pct + 10:
                self._last_pct = pct
                logging.info('%s: %d/%d (%d%%)' % (self.desc, self.n, self.total, pct))

    def close(self):
        logging.info('%s: complete (%d/%d)' % (self.desc, self.n, self.total))


# =============================================================================
# Protocol Extraction (compatible with original)
# =============================================================================

def extract_protocol_from_path(series_path: str) -> str:
    """경로에서 프로토콜 디렉토리명 추출"""
    parts = series_path.rstrip(os.sep).split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return ''


# =============================================================================
# DCM Protocol Extraction (#5 최적화: 인벤토리 빌드 시 메타데이터 사전 추출)
# =============================================================================

def _read_dcm_protocol(dcm_path: str) -> dict:
    """DCM 파일에서 TE/TR/TI/Flip Angle 추출 (pydicom, stop_before_pixels).

    Returns: {'TE': val, 'TR': val, 'TI': val, 'Flip Angle': val}
    """
    try:
        import pydicom
    except ImportError:
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
    except Exception:
        pass
    return result


# =============================================================================
# Optimized Series Scan (#2 + #3)
# =============================================================================

def _scan_series_fast(series_path, source_name):
    """Optimized series folder scan: os.scandir + single-pass path parsing.

    Replaces scan_series_folder():
      - os.scandir single pass instead of glob('*.dcm') + sorted()
      - Path component extraction instead of 5 separate regex calls
    """
    # Count DCM files and find first filename (alphabetically) in one pass (#2)
    dcm_count = 0
    first_dcm = None

    try:
        with os.scandir(series_path) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=True) and entry.name.endswith('.dcm'):
                    dcm_count += 1
                    if first_dcm is None or entry.name < first_dcm:
                        first_dcm = entry.name
    except OSError:
        return None

    if dcm_count == 0:
        return None

    # Single-pass path parsing (#3)
    # Expected: .../SOURCE/PTID/PROTOCOL/DATETIME/I{UID}
    parts = series_path.rstrip(os.sep).split(os.sep)
    if len(parts) < 4:
        return None

    uid_folder = parts[-1]       # I{UID}
    datetime_part = parts[-2]    # YYYY-MM-DD_HH_MM_SS.0
    protocol = parts[-3]         # protocol name
    ptid_part = parts[-4]        # XXX_S_XXXX(X)

    # PTID validation
    m = _PTID_RE.search(ptid_part)
    if not m:
        return None
    ptid = m.group(0)

    # Date: fast path from datetime folder name, regex fallback
    if len(datetime_part) >= 10 and datetime_part[4] == '-' and datetime_part[7] == '-':
        date = datetime_part[:10]
    else:
        m = _DATE_RE.search(series_path)
        date = m.group(1) if m else ''

    # Image UID: filename first (_I{num}), folder name fallback (I{num})
    image_uid = ''
    if first_dcm:
        m = _IUID_FILENAME_RE.search(first_dcm)
        if m:
            image_uid = m.group(1)
    if not image_uid and uid_folder.startswith('I') and uid_folder[1:].isdigit():
        image_uid = uid_folder[1:]

    # Series UID from filename (_S{num})
    series_uid = ''
    if first_dcm:
        m = _SUID_RE.search(first_dcm)
        if m:
            series_uid = m.group(1)

    # DCM 메타데이터 사전 추출 (#5 최적화)
    dcm_meta = {}
    if first_dcm:
        dcm_file = os.path.join(series_path, first_dcm)
        dcm_meta = _read_dcm_protocol(dcm_file)

    return {
        'ptid': ptid,
        'source': source_name,
        'series_uid': series_uid,
        'image_uid': image_uid,
        'date': date,
        'protocol': protocol,
        'dcm_count': dcm_count,
        'dcm_path': series_path,
        'first_dcm': first_dcm or '',
        'dcm_TE': dcm_meta.get('TE', ''),
        'dcm_TR': dcm_meta.get('TR', ''),
        'dcm_TI': dcm_meta.get('TI', ''),
        'dcm_FlipAngle': dcm_meta.get('Flip Angle', ''),
        'dcm_PulseSequence': dcm_meta.get('Pulse Sequence', ''),
        'dcm_PixelSpacing': dcm_meta.get('Pixel Spacing', ''),
        'dcm_MatrixX': dcm_meta.get('Matrix X', ''),
        'dcm_MatrixY': dcm_meta.get('Matrix Y', ''),
        'dcm_MatrixZ': dcm_meta.get('Matrix Z', ''),
    }


# =============================================================================
# Structured 3-level Scan (#1)
# =============================================================================

def _collect_series_paths_structured(ptid_path):
    """3-level structured scan: PTID/protocol/datetime/I{UID}

    Uses os.scandir at each level for DirEntry caching (fewer NFS stat calls).
    Silently skips levels that fail with OSError.
    """
    paths = []
    try:
        for proto in os.scandir(ptid_path):
            if not proto.is_dir(follow_symlinks=True):
                continue
            try:
                for dt in os.scandir(proto.path):
                    if not dt.is_dir(follow_symlinks=True):
                        continue
                    try:
                        for uid_dir in os.scandir(dt.path):
                            if uid_dir.is_dir(follow_symlinks=True):
                                paths.append(uid_dir.path)
                    except OSError:
                        pass
            except OSError:
                pass
    except OSError:
        pass
    return paths


def _collect_series_paths_walk(ptid_path):
    """Fallback: os.walk to find all directories containing .dcm files."""
    paths = []
    for root, _dirs, files in os.walk(ptid_path):
        if any(f.endswith('.dcm') for f in files):
            paths.append(root)
    return paths


def _scan_ptid_folder(ptid_path, source_name, pbar=None):
    """Scan a single PTID folder: structured 3-level scan + walk fallback."""
    # Try structured scan first (#1)
    series_paths = _collect_series_paths_structured(ptid_path)

    # Fallback to os.walk if structured scan found nothing
    if not series_paths:
        walk_paths = _collect_series_paths_walk(ptid_path)
        if walk_paths:
            logging.debug(
                'Non-standard depth in %s, found %d series via walk fallback'
                % (os.path.basename(ptid_path), len(walk_paths)))
            series_paths = walk_paths

    results = []
    for sp in series_paths:
        rec = _scan_series_fast(sp, source_name)
        if rec:
            results.append(rec)

    if pbar is not None:
        pbar.update(1)

    return results


# =============================================================================
# Source Scanning with PTID Parallelization (#4)
# =============================================================================

def scan_source(source_name, source_path, ptid_workers=8, pbar=None):
    """Scan a single DCM source folder with PTID-level parallelization."""
    if not os.path.exists(source_path):
        logging.warning('DCM source not found: %s (%s)' % (source_name, source_path))
        return []

    # Enumerate PTID folders using os.scandir (faster than glob)
    ptid_folders = []
    try:
        with os.scandir(source_path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=True) and '_S_' in entry.name:
                    ptid_folders.append(entry.path)
    except OSError as e:
        logging.error('Cannot scan source %s: %s' % (source_path, e))
        return []

    if not ptid_folders:
        return []

    results = []

    if ptid_workers <= 1:
        for pf in ptid_folders:
            results.extend(_scan_ptid_folder(pf, source_name, pbar))
    else:
        with ThreadPoolExecutor(max_workers=ptid_workers) as executor:
            futures = [
                executor.submit(_scan_ptid_folder, pf, source_name, pbar)
                for pf in ptid_folders
            ]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error('PTID scan error in %s: %s' % (source_name, e))

    return results


# =============================================================================
# Source Dispatch Table for classify_series (#6)
# =============================================================================

def _build_source_dispatch(modality_config):
    """Build source → [(mod_key, mod_cfg)] dispatch table.

    Instead of iterating all 16 modalities per series, only check
    modalities whose 'sources' list includes the series' source.
    """
    dispatch = defaultdict(list)
    for mod_key, mod_cfg in modality_config.items():
        for source in mod_cfg.get('sources', []):
            dispatch[source].append((mod_key, mod_cfg))
    return dict(dispatch)


def classify_series(series_record: dict, modality_config: dict = None,
                    _dispatch: dict = None) -> list:
    """Classify series into modalities (dispatch-table optimized).

    Compatible with original classify_series() signature.
    Pass _dispatch for batch use to avoid rebuilding per call.
    """
    if modality_config is None:
        modality_config = MODALITY_CONFIG
    if _dispatch is None:
        _dispatch = _build_source_dispatch(modality_config)

    source = series_record.get('source', '')
    protocol = series_record.get('protocol', '')
    candidates = _dispatch.get(source, [])
    matched = []

    for mod_key, mod_cfg in candidates:
        # include pattern: source_specific_regex first, then global regex
        src_regex = mod_cfg.get('source_specific_regex', {})
        regex_list = src_regex.get(source, mod_cfg.get('regex', ['*']))

        if not any(fnmatch.fnmatch(protocol, pat) for pat in regex_list):
            continue

        # exclude check
        exclude = mod_cfg.get('exclude_regex', [])
        if any(fnmatch.fnmatch(protocol, pat) for pat in exclude):
            continue

        matched.append(mod_key)

    return matched


# =============================================================================
# Inventory Build with Parallel Scanning (#5 + #7)
# =============================================================================

def build_inventory(nfs_base: str = None, modality_config: dict = None,
                    source_workers: int = 6, ptid_workers: int = 8) -> dict:
    """Build DCM inventory with parallel scanning.

    Args:
        nfs_base: DCM root path. None → config.NFS_BASE
        modality_config: MODALITY_CONFIG dict. None → default
        source_workers: source-level parallel threads (default: 6)
        ptid_workers: PTID-level parallel threads per source (default: 8)

    Returns:
        inventory dict with 'metadata', 'by_modality', 'by_image_uid', 'unclassified'
    """
    if nfs_base is None:
        nfs_base = NFS_BASE
    if modality_config is None:
        modality_config = MODALITY_CONFIG

    # Pre-count total PTID folders for progress bar
    sources = []
    total_ptids = 0
    for source_name, dir_name in DCM_SOURCES.items():
        source_path = os.path.join(nfs_base, dir_name)
        if os.path.isdir(source_path):
            sources.append((source_name, source_path))
            try:
                count = sum(
                    1 for e in os.scandir(source_path)
                    if e.is_dir(follow_symlinks=True) and '_S_' in e.name
                )
                total_ptids += count
            except OSError:
                pass

    logging.info('Scanning DCM sources from %s (%d sources, ~%d PTID folders)' % (
        nfs_base, len(sources), total_ptids))

    # Progress bar (#7)
    if _HAS_TQDM and total_ptids > 0:
        pbar = tqdm(
            total=total_ptids, desc='Scanning DCM', unit='ptid',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    elif total_ptids > 0:
        pbar = _ProgressFallback(total=total_ptids, desc='Scanning DCM')
    else:
        pbar = None

    # Phase 1: Parallel source scanning (#5)
    all_series = []

    if source_workers <= 1 or len(sources) <= 1:
        for source_name, source_path in sources:
            logging.info('  Scanning %s: %s' % (source_name, source_path))
            results = scan_source(source_name, source_path, ptid_workers, pbar)
            logging.info('  Found %d series' % len(results))
            all_series.extend(results)
    else:
        with ThreadPoolExecutor(max_workers=source_workers) as executor:
            futures = {
                executor.submit(scan_source, name, path, ptid_workers, pbar): name
                for name, path in sources
            }
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    results = future.result()
                    logging.info('  %s: %d series' % (source_name, len(results)))
                    all_series.extend(results)
                except Exception as e:
                    logging.error('Source scan error (%s): %s' % (source_name, e))

    if pbar is not None:
        pbar.close()

    # Sort for deterministic output
    all_series.sort(key=lambda r: (r['source'], r['ptid'], r['date']))

    logging.info('Total %d series scanned' % len(all_series))

    # Phase 2: classify_series with dispatch table (#6)
    dispatch = _build_source_dispatch(modality_config)
    classified_count = 0
    unclassified = []
    series_modalities = []

    for rec in all_series:
        mods = classify_series(rec, modality_config, _dispatch=dispatch)
        series_modalities.append((rec, mods))
        if mods:
            classified_count += 1
        else:
            unclassified.append(rec)

    logging.info('Classified: %d, Unclassified: %d' % (classified_count, len(unclassified)))

    # Phase 3: by_modality[mod][ptid] 구축
    by_modality = {mod_key: defaultdict(list) for mod_key in modality_config}

    for rec, mods in series_modalities:
        for mod_key in mods:
            entry = {
                'image_uid': rec['image_uid'],
                'date': rec['date'],
                'dcm_path': rec['dcm_path'],
                'protocol': rec['protocol'],
                'source': rec['source'],
                'dcm_count': rec['dcm_count'],
                'dcm_TE': rec.get('dcm_TE', ''),
                'dcm_TR': rec.get('dcm_TR', ''),
                'dcm_TI': rec.get('dcm_TI', ''),
                'dcm_FlipAngle': rec.get('dcm_FlipAngle', ''),
                'dcm_PulseSequence': rec.get('dcm_PulseSequence', ''),
                'dcm_PixelSpacing': rec.get('dcm_PixelSpacing', ''),
                'dcm_MatrixX': rec.get('dcm_MatrixX', ''),
                'dcm_MatrixY': rec.get('dcm_MatrixY', ''),
                'dcm_MatrixZ': rec.get('dcm_MatrixZ', ''),
            }
            by_modality[mod_key][rec['ptid']].append(entry)

    # defaultdict → dict 변환
    by_modality = {mod_key: dict(ptid_dict) for mod_key, ptid_dict in by_modality.items()}

    # Phase 4: by_image_uid (full record dict, 비MRI 우선)
    by_image_uid = {}
    for rec in all_series:
        uid = rec.get('image_uid', '')
        if not uid:
            continue
        # 이미 존재하면 비MRI 소스 우선 (전용 폴더가 더 정확)
        if uid in by_image_uid:
            existing = by_image_uid[uid]
            if existing.get('source') != 'MRI' and rec.get('source') == 'MRI':
                continue  # 기존이 비MRI → MRI 레코드 무시
        by_image_uid[uid] = {
            'ptid': rec['ptid'],
            'source': rec['source'],
            'image_uid': uid,
            'date': rec['date'],
            'protocol': rec['protocol'],
            'dcm_path': rec['dcm_path'],
            'dcm_count': rec['dcm_count'],
            'dcm_TE': rec.get('dcm_TE', ''),
            'dcm_TR': rec.get('dcm_TR', ''),
            'dcm_TI': rec.get('dcm_TI', ''),
            'dcm_FlipAngle': rec.get('dcm_FlipAngle', ''),
            'dcm_PulseSequence': rec.get('dcm_PulseSequence', ''),
            'dcm_PixelSpacing': rec.get('dcm_PixelSpacing', ''),
            'dcm_MatrixX': rec.get('dcm_MatrixX', ''),
            'dcm_MatrixY': rec.get('dcm_MatrixY', ''),
            'dcm_MatrixZ': rec.get('dcm_MatrixZ', ''),
        }

    # Phase 5: 메타데이터 집계
    modality_counts = {}
    for mod_key in modality_config:
        ptid_dict = by_modality.get(mod_key, {})
        series_count = sum(len(entries) for entries in ptid_dict.values())
        modality_counts[mod_key] = {
            'ptids': len(ptid_dict),
            'series': series_count,
        }

    # unclassified 프로토콜별 카운트 로깅
    if unclassified:
        proto_counts = defaultdict(int)
        for rec in unclassified:
            key = '%s/%s' % (rec['source'], rec['protocol'])
            proto_counts[key] += 1
        logging.info('Unclassified protocols:')
        for proto, cnt in sorted(proto_counts.items(), key=lambda x: -x[1])[:20]:
            logging.info('  %s: %d series' % (proto, cnt))

    inventory = {
        'metadata': {
            'version': 2,
            'nfs_base': nfs_base,
            'total_series': len(all_series),
            'classified_series': classified_count,
            'unclassified_count': len(unclassified),
            'modality_counts': modality_counts,
        },
        'by_modality': by_modality,
        'by_image_uid': by_image_uid,
        'unclassified': unclassified,
    }

    logging.info('Inventory v2 complete: %d series, %d modalities' % (
        len(all_series), len(modality_config)))
    for mod_key, counts in modality_counts.items():
        if counts['series'] > 0:
            logging.info('  %s: %d PTIDs, %d series' % (mod_key, counts['ptids'], counts['series']))

    return inventory


# =============================================================================
# I/O (identical to inventory.py)
# =============================================================================

def save_inventory(inventory: dict, output_path: str):
    """inventory를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    logging.info('Inventory saved to %s' % output_path)


def load_inventory(input_path: str) -> dict:
    """저장된 inventory JSON 로드"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dcm_path_by_image_uid(inventory: dict, image_uid: str) -> str:
    """ImageUID로 DCM 경로 조회"""
    record = inventory.get('by_image_uid', {}).get(str(image_uid), {})
    if isinstance(record, dict):
        return record.get('dcm_path', '')
    # v1 호환: str 값
    return record if isinstance(record, str) else ''
