import pandas as pd
import os
import re
import logging
from xmltodict3 import XmlTextToDict
from copy import deepcopy
from glob import glob
import numpy as np
import datetime
import warnings
from joblib import Parallel, delayed

try:
    from adni.matching.reference.params import *
except ImportError:
    from params import *

warnings.filterwarnings(action='ignore')


class ADNI:
    """
    1.Closest & largest ImageUID T1 image - ADNIMERGE matching
    *need confirm* 2. VisitIdentifier-based multi-modality matching
    3. Sort columns & convert VISCODE_FIX into 'bl & m%02d' format
    """

    __version__ = '2023-10-12'

    def __init__(self, output_directory: str, n_jobs=8):
        self.output_directory = output_directory
        self.n_jobs = n_jobs
        os.makedirs(output_directory, exist_ok=True)
        self.log_path = os.path.join(output_directory, 'matching.log')
        _setup_logger(self.log_path)
        logging.info('--------------------    INIT     --------------------')
        logging.info('ADNI matching Version=%s' % self.__version__)
        param_logging(output_directory=output_directory, n_jobs=n_jobs)
        logging.info('')

    def adnimerge_matching(self, img_dir, xml_dir, adnimerge, threshold, modality: str, overwrite=False,
                           regex: (str, list) = '*'):
        modality = modality.upper()

        logging.info('-------------------- %s matching --------------------' % modality)
        param_logging(img_dir=img_dir, xml_dir=xml_dir, adnimerge=adnimerge,
                      adnimerge_threshold=threshold, overwrite=overwrite, regex=regex)
        logging.info('')
        output_path_all = os.path.join(self.output_directory, '%s_all.csv' % modality)
        output_path_unique = os.path.join(self.output_directory, '%s_unique.csv' % modality)
        if not overwrite and (os.path.isfile(output_path_all) or os.path.isfile(output_path_unique)):
            logging.error('File exists: %s or %s. If you want to update file, set overwrite=True' % (
                output_path_all, output_path_unique))
            return

        temp_columns = []  # for clearance

        # 2023-09-12: add ucberkeley_av45
        # 2023-10-11: fix ADNIMERGE and UCBERKELEYAV45 matching [RID, EXAMDATE] -> [RID, VISCODE]
        # 2023-10-12: move UCBERKELEYAV45 matching to attach_ucberkeley
        ADNIMERGE = pd.read_csv(adnimerge)

        # convert into datetime format
        ADNIMERGE[ADNIMERGE_MATCHING_TARGET_COLUMN] = ADNIMERGE.apply(lambda x: _strptime(x['EXAMDATE']), axis=1)
        ADNIMERGE[ADNIMERGE_VISCODE_TARGET_COLUMN] = ADNIMERGE.apply(lambda x: _strptime(x['EXAMDATE_bl']), axis=1)
        temp_columns.append(ADNIMERGE_MATCHING_TARGET_COLUMN)
        temp_columns.append(ADNIMERGE_VISCODE_TARGET_COLUMN)

        # get subject list
        subj_list = [e for e in glob(os.path.join(img_dir, '*_S_*')) if os.path.isdir(e)]
        subj_list.sort()
        logging.info('Total %s subject detected' % len(subj_list))
        if xml_dir and os.path.isdir(str(xml_dir)):
            logging.info('Total %s xml detected' % len(glob(os.path.join(xml_dir, '*.xml'))))
        else:
            logging.info('No XML directory - using DICOM/path-based fallback matching')

        # multiprocessing
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(subj_matching)(self.output_directory, subj, ADNIMERGE, xml_dir, threshold, modality, regex, self.log_path) for subj in subj_list)
        result = [e for e in result if e is not None]

        result = pd.concat(result, ignore_index=True)
        result = result.drop(temp_columns, axis=1)  # clearance
        result.to_csv(output_path_all, index=False)  # 2023-09-12: save intermediate csv
        result.drop_duplicates(subset=['PTID', 'VISCODE_FIX'], keep='last').query("VISCODE_FIX != 'error'").to_csv(output_path_unique,
                                                                                   index=False)  # 2023-09-12: save final csv; 2023-10-12: drop error for unique
        logging.info('Total %s subjects and %s visit points' % (len(result['PTID'].unique()), len(result)))
        logging.info('Output saved at %s' % output_path_all)
        logging.info('Output saved at %s' % output_path_unique)
        logging.info('-----------------------------------------------------\n')

    def attach_ucberkeley(self, av45_unique_csv, ucberkeley_av45):
        logging.info('-------------------- attach ucberekeley result --------------------')
        param_logging(av45_unique_csv=av45_unique_csv, ucberkeley_av45=ucberkeley_av45)
        logging.info('')
        result = pd.read_csv(av45_unique_csv)
        UCBERKELEYAV45 = pd.read_csv(ucberkeley_av45)
        result['ind1'] = result.apply(lambda x: x['PTID'].split('_S_')[-1].lstrip('0'), axis=1)
        result['ind2'] = result['AQUDATE_AV45']
        UCBERKELEYAV45['ind1'] = UCBERKELEYAV45.apply(lambda x: float2str(x['RID']), axis=1)
        UCBERKELEYAV45['ind2'] = UCBERKELEYAV45['EXAMDATE']
        result = result.set_index(['ind1', 'ind2'], drop=True)
        UCBERKELEYAV45 = UCBERKELEYAV45.set_index(['ind1', 'ind2'], drop=False).drop(
            ['VISCODE2', 'update_stamp', 'EXAMDATE', 'RID', 'VISCODE'], axis=1)
        result = result.drop(result.columns.intersection(UCBERKELEYAV45.columns), axis=1)
        left_diff = result.index.difference(UCBERKELEYAV45.index).to_frame()
        right_diff = UCBERKELEYAV45.index.difference(result.index).to_frame()
        temp_diff = deepcopy(right_diff)
        left_diff['ind2_aqutime'] = left_diff.apply(lambda x: datetime.datetime.strptime(x['ind2'], '%Y-%m-%d'), axis=1)
        right_diff['ind2_aqutime'] = right_diff.apply(lambda x: datetime.datetime.strptime(x['ind2'], '%Y-%m-%d'), axis=1)

        logging.info('%s non-matching indices found' % len(left_diff))
        for i, row in left_diff.iterrows():
            rid = row['ind1']
            aqutime = row['ind2_aqutime']
            rid_diff = right_diff.query("ind1=='%s'" % rid)
            if not len(rid_diff):
                logging.info('\tMatching %s failed: no matching index in UCBERKELEYAV45.' % i)
            delta = aqutime - rid_diff['ind2_aqutime']
            rid_diff['ind2_aqutime'] = np.abs(delta)
            nn_idx = rid_diff['ind2_aqutime'].argmin()
            if pd.Timedelta(30, unit='d') < rid_diff['ind2_aqutime'].iloc[nn_idx]:
                logging.info('\tMatching %s failed: over threshold.' % i)
            else:
                temp_diff.loc[rid_diff.index[nn_idx]]['ind2'] = row['ind2']
                logging.info('\tMatching %s <-> %s' % (i, rid_diff.index[nn_idx]))

        for i, row in temp_diff.iterrows():
            UCBERKELEYAV45.at[i, 'ind2'] = row['ind2']

        UCBERKELEYAV45 = UCBERKELEYAV45.reset_index(drop=True).set_index(['ind1', 'ind2'], drop=True)
        result = result.join(UCBERKELEYAV45, how='left')
        result.to_csv(av45_unique_csv, index=False)
        logging.info('Output saved at %s' % av45_unique_csv)
        logging.info('-----------------------------------------------------\n')

    def attach_preprocess_path(self, adnimerge_matching_csv, key, regex, index: (str, list)):
        if isinstance(index, str):
            index = [index]
        logging.info('-------------------- attach preprocess path --------------------')
        param_logging(adnimerge_matching_csv=adnimerge_matching_csv, key=key, regex=regex, index=index)
        logging.info('')
        result = pd.read_csv(adnimerge_matching_csv)

        def glob_by_index(row):
            row_regex = glob(regex.format(**{key: row[key] for key in index}))
            if len(row_regex) == 0:
                logging.debug('[%s|%s] no path found' % (':'.join(index), ':'.join([str(row[key]) for key in index])))
                return ''
            elif len(row_regex) > 1:
                logging.debug('[%s|%s] duplicated path found %s' % (
                    ':'.join(index), ':'.join([str(row[key]) for key in index]), '|'.join(row_regex)))
            else:
                return row_regex[0]

        result[key] = result.apply(lambda x: glob_by_index(x), axis=1)
        result.to_csv(adnimerge_matching_csv, index=False)
        logging.info('Output saved at %s' % adnimerge_matching_csv)
        logging.info('-----------------------------------------------------\n')

    def unique_csv_merge(self):
        logging.info('-------------------- Unique csv merge --------------------')
        output_path = os.path.join(self.output_directory, 'MERGED.csv')
        flist = glob(os.path.join(self.output_directory, '*_unique.csv'))
        df_list = []
        for f in flist:
            logging.info('input csv: %s' % f)
            df_list.append(pd.read_csv(f).set_index(['PTID', 'VISCODE_FIX']))
        df_list = pd.DataFrame(dict(df=df_list, path=flist), index=[len(e) for e in df_list])
        df_list.sort_index(ascending=False, inplace=True)
        init_df = df_list.df.iloc[0]
        logging.info('')
        logging.info('init csv: %s, current shape: %s' % (df_list.path.iloc[0], init_df.shape))
        for i, row in df_list.iloc[1:].iterrows():
            drop_columns = list(set(init_df.columns).intersection(set(row.df.columns)))
            new_index = row.df.index.difference(init_df.index)
            init_df = init_df.join(row.df.drop(drop_columns, axis=1), how='outer')
            init_df = init_df[~init_df.index.duplicated(keep='first')]
            init_df.loc[new_index] = row.df.loc[new_index]
            logging.info('merge csv: %s, current shape: %s' % (row.path, init_df.shape))
        init_df.sort_index().to_csv(output_path)
        logging.info('Output saved at %s' % output_path)
        logging.info('-----------------------------------------------------\n')


# -------------------- Functions for adnimerge matching -------------------- #


def _strptime(value):
    if isinstance(value, str):
        return datetime.datetime.strptime(value, '%Y-%m-%d')
    else:
        return pd.NaT


def _nearest_adinmerge(subj_adnimerge, aqutime, threshold):
    """
    :return: closest adnimerge row based on aqudate and EXAMDATE
    """
    if len(subj_adnimerge):
        subj_adnimerge = subj_adnimerge.copy()
        target_column = ADNIMERGE_MATCHING_TARGET_COLUMN
        delta = aqutime - subj_adnimerge[target_column]
        subj_adnimerge[target_column] = np.abs(delta)
        subj_adnimerge['DAYSDELTA'] = delta.dt.days  # 2023-10-11: fix DAYSDELTA
        nn_idx = subj_adnimerge[target_column].argmin()
        nn_row = subj_adnimerge.iloc[nn_idx:nn_idx + 1]
        if pd.Timedelta(threshold, unit='d') < subj_adnimerge[target_column].iloc[nn_idx]:  # if over than threshold
            nn_row[ADNIMERGE_NO_MATCHING_RESET_COLUMN] = ''  # remove
    else:  # if subject not exists in ADNIMERGE
        nn_row = subj_adnimerge.append(pd.Series([]), ignore_index=True)  # empty rowv
    return nn_row


def _calc_viscode(timedelta, threshold):
    """
    Converts daysdelta to VISCODE_FIX (daysdelta = AQUDATE - EXAMDATE_BL)
    """
    daysdelta = timedelta.total_seconds() / 86400
    delta = np.abs(MONTH_KEYS - daysdelta)
    idx = np.argmin(delta)
    if delta[idx] > threshold:
        return 'error'  # line 308 2023-10-12: fix
    key = MONTH_KEYS[idx]
    return MONTHS[key]


def _demo_matching(subj_adnimerge, xml_path, threshold, modality, image_path, log_path):
    """
    :return: xml with closest ADNIMERGE row
    """
    _setup_logger(log_path)
    xml = read_xml(xml_path).get('project', '')
    if xml == '':
        logging.warning('Invalid xml: %s' % xml_path)
    aqudate = safe_dict_search(xml, 'subject', 'study', 'series', 'dateAcquired')
    aqutime = datetime.datetime.strptime(aqudate, '%Y-%m-%d')

    row = _nearest_adinmerge(subj_adnimerge, aqutime, threshold)
    if not pd.isna(row.iloc[0][ADNIMERGE_VISCODE_TARGET_COLUMN]):
        row['VISCODE_FIX'] = _calc_viscode(aqutime - row.iloc[0][ADNIMERGE_VISCODE_TARGET_COLUMN], threshold)
    else:  # if subject not exists in ADNIMERGE
        row['VISCODE_FIX'] = 'm000'  # 2023-10-11: fix
        row['PTID'] = safe_dict_search(xml, 'subject', 'subjectIdentifier')
    row['MODALITY'] = modality
    row['AQUDATE_%s' % modality] = aqudate
    row['visitIdentifier_%s' % modality] = safe_dict_search(xml, 'subject', 'visit', 'visitIdentifier')
    row['S_%s' % modality] = str(safe_dict_search(xml, 'subject', 'study', 'series', 'seriesIdentifier'))
    row['I_%s' % modality] = str(safe_dict_search(xml, 'subject', 'study', 'imagingProtocol', 'imageUID',
                                                  default=safe_dict_search(xml, 'subject', 'study', 'series',
                                                                           'seriesLevelMeta', 'derivedProduct',
                                                                           'imageUID')))  # 2023-10-12 fix imageUID key
    protocol = list2dict(
        safe_dict_search(xml, 'subject', 'study', 'imagingProtocol', 'protocolTerm', 'protocol',
                         default=safe_dict_search(xml, 'subject', 'study', 'series', 'seriesLevelMeta',
                                                  'relatedImageDetail', 'originalRelatedImage', 'protocolTerm',
                                                  'protocol')))

    row['researchGroup'] = safe_dict_search(xml, 'subject', 'researchGroup')
    row['subjectAge'] = safe_dict_search(xml, 'subject', 'study', 'subjectAge')
    row['subjectSex'] = safe_dict_search(xml, 'subject', 'subjectSex')
    row['weightKg'] = safe_dict_search(xml, 'subject', 'study', 'weightKg')
    row['Apoe'] = get_apoe(safe_dict_search(xml, 'subject', 'subjectInfo'))

    row['protocol/%s/description' % modality] = safe_dict_search(xml, 'subject', 'study', 'imagingProtocol', 'description')
    row['protocol/%s/Acquisition Type' % modality] = safe_dict_search(protocol, 'Acquisition Type')
    row['protocol/%s/Acquisition Plane' % modality] = safe_dict_search(protocol, 'Acquisition Plane')
    row['protocol/%s/Weighting' % modality] = safe_dict_search(protocol, 'Weighting')
    row['protocol/%s/Pulse Sequence' % modality] = safe_dict_search(protocol, 'Pulse Sequence')
    row['protocol/%s/Slice Thickness' % modality] = safe_dict_search(protocol, 'Slice Thickness')
    row['protocol/%s/Pixel Spacing X' % modality] = safe_dict_search(protocol, 'Pixel Spacing X')
    row['protocol/%s/Pixel Spacing Y' % modality] = safe_dict_search(protocol, 'Pixel Spacing Y')
    row['protocol/%s/TE' % modality] = safe_dict_search(protocol, 'TE')
    row['protocol/T%s/R' % modality] = safe_dict_search(protocol, 'TR')
    row['protocol/%s/TI' % modality] = safe_dict_search(protocol, 'TI')
    row['protocol/%s/Coil' % modality] = safe_dict_search(protocol, 'Coil')
    row['protocol/%s/Flip Angle' % modality] = safe_dict_search(protocol, 'Flip Angle')
    row['protocol/%s/Matrix X' % modality] = safe_dict_search(protocol, 'Matrix X',
                                                default=safe_dict_search(protocol, 'Number of Rows'))
    row['protocol/%s/Matrix Y' % modality] = safe_dict_search(protocol, 'Matrix Y',
                                                default=safe_dict_search(protocol, 'Number of Columns'))
    row['protocol/%s/Matrix Z' % modality] = safe_dict_search(protocol, 'Matrix Z',
                                                default=safe_dict_search(protocol, 'Number of Slices'))
    row['protocol/%s/Manufacturer' % modality] = safe_dict_search(protocol, 'Manufacturer')
    row['protocol/%s/Mfg Model' % modality] = safe_dict_search(protocol, 'Mfg Model')
    row['protocol/%s/Field Strength' % modality] = safe_dict_search(protocol, 'Field Strength')
    row['%s_image_path' % modality] = image_path
    return row


def _extract_image_uid_from_path(path):
    """Extract ImageUID from /I{number}/ in path (for ADNI_extract structure)"""
    match = re.search(r'/I(\d+)/', path)
    if match:
        return match.group(1)
    # Also try from filename
    match = re.search(r'_I(\d+)', os.path.basename(path))
    if match:
        return match.group(1)
    return ''


def _extract_series_uid_from_path(path):
    """Extract SeriesUID from _S{number}_ in path/filename"""
    match = re.search(r'_S(\d+)', path)
    return match.group(1) if match else ''


def _extract_date_from_path(path):
    """Extract acquisition date (YYYY-MM-DD) from path"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', path)
    return match.group(1) if match else None


def _demo_matching_from_dicom(subj_adnimerge, threshold, modality, image_path, log_path):
    """
    DICOM-header-free matching for DTI/fMRI (no XML available).
    Extracts date/ImageUID from path, uses ADNIMERGE for demographics.
    Returns: DataFrame row compatible with _demo_matching() output.
    """
    _setup_logger(log_path)

    # Extract date from path
    aqudate = _extract_date_from_path(image_path)
    if aqudate is None:
        logging.warning('\t[DICOM fallback] no date in path: %s' % image_path)
        return None

    aqutime = datetime.datetime.strptime(aqudate, '%Y-%m-%d')

    # Extract ImageUID and SeriesUID from path
    image_uid = _extract_image_uid_from_path(image_path)
    series_uid = _extract_series_uid_from_path(image_path)

    # Match to ADNIMERGE (same logic as _demo_matching)
    row = _nearest_adinmerge(subj_adnimerge, aqutime, threshold)
    if not pd.isna(row.iloc[0][ADNIMERGE_VISCODE_TARGET_COLUMN]):
        row['VISCODE_FIX'] = _calc_viscode(aqutime - row.iloc[0][ADNIMERGE_VISCODE_TARGET_COLUMN], threshold)
    else:
        row['VISCODE_FIX'] = 'm000'
        # Try to get PTID from path
        ptid_match = re.search(r'(\d{3}_S_\d{4})', image_path)
        if ptid_match:
            row['PTID'] = ptid_match.group(1)

    row['MODALITY'] = modality
    row['AQUDATE_%s' % modality] = aqudate
    row['visitIdentifier_%s' % modality] = ''  # not available without XML
    row['S_%s' % modality] = series_uid
    row['I_%s' % modality] = image_uid

    # Demographics from ADNIMERGE (fallback for XML fields)
    if len(subj_adnimerge) > 0:
        first_row = subj_adnimerge.iloc[0]
        row['researchGroup'] = first_row.get('DX_bl', '')
        row['subjectAge'] = first_row.get('AGE', '')
        row['subjectSex'] = first_row.get('PTGENDER', '')
        row['Apoe'] = str(first_row.get('APOE4', ''))
    else:
        row['researchGroup'] = ''
        row['subjectAge'] = ''
        row['subjectSex'] = ''
        row['Apoe'] = ''

    row['weightKg'] = ''  # not available without XML/DICOM

    # Protocol fields - set to empty (no XML available)
    row['protocol/%s/description' % modality] = ''
    row['protocol/%s/Acquisition Type' % modality] = ''
    row['protocol/%s/Acquisition Plane' % modality] = ''
    row['protocol/%s/Weighting' % modality] = ''
    row['protocol/%s/Pulse Sequence' % modality] = ''
    row['protocol/%s/Slice Thickness' % modality] = ''
    row['protocol/%s/Pixel Spacing X' % modality] = ''
    row['protocol/%s/Pixel Spacing Y' % modality] = ''
    row['protocol/%s/TE' % modality] = ''
    row['protocol/T%s/R' % modality] = ''
    row['protocol/%s/TI' % modality] = ''
    row['protocol/%s/Coil' % modality] = ''
    row['protocol/%s/Flip Angle' % modality] = ''
    row['protocol/%s/Matrix X' % modality] = ''
    row['protocol/%s/Matrix Y' % modality] = ''
    row['protocol/%s/Matrix Z' % modality] = ''
    row['protocol/%s/Manufacturer' % modality] = ''
    row['protocol/%s/Mfg Model' % modality] = ''
    row['protocol/%s/Field Strength' % modality] = ''
    row['%s_image_path' % modality] = image_path
    return row


def subj_matching(output_directory, subj, ADNIMERGE, xml_dir, threshold, modality, regex, log_path):
    """
    :return: matched subject DataFrame
    """
    warnings.filterwarnings(action='ignore')
    _setup_logger(log_path)
    ptid = os.path.split(subj)[1]
    subj_adnimerge = ADNIMERGE.query("PTID=='%s'" % ptid)
    if isinstance(regex, str):
        regex = [regex]
    image_list = []
    for re_pattern in regex:
        image_list += glob(os.path.join(subj, re_pattern, '*', '*', '*.nii*'))  # get all images from one subject
    subj_result = []

    # Determine if XML-based or DICOM-fallback mode
    use_dicom_fallback = (xml_dir == '' or xml_dir is None or not os.path.isdir(str(xml_dir)))

    for image in image_list:
        if use_dicom_fallback:
            # DTI/fMRI: no XML available, use DICOM/path-based matching
            result = _demo_matching_from_dicom(subj_adnimerge, threshold, modality, image, log_path)
            if result is not None:
                subj_result.append(result)
        else:
            # Original XML-based matching
            xml_path = glob(os.path.join(xml_dir, '*_I%s.xml' % image.split('_I')[-1].split('.nii')[0]))  # get xml
            if len(xml_path) == 0:  # no xml -> try DICOM fallback
                logging.warning('\t[XML] not exists: %s' % image)
                result = _demo_matching_from_dicom(subj_adnimerge, threshold, modality, image, log_path)
                if result is not None:
                    subj_result.append(result)
                continue
            elif len(xml_path) > 1:  # at least 2 xml -> continue
                logging.warning('\t[XML] duplicated: %s' % image)
                continue
            subj_result.append(
                _demo_matching(subj_adnimerge, xml_path[0], threshold, modality, image, log_path))  # only one xml

    image_list = [e for e in subj_result if e is not None]

    if not len(image_list):
        return None

    subj_result = pd.concat(image_list, ignore_index=True)
    subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] = subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN].fillna(
        pd.Timedelta(99999, unit='d'))
    subj_result[ADNIMERGE_MATCHING_TARGET_COLUMN] *= -1
    subj_result = subj_result.sort_values(by=['VISCODE_FIX', ADNIMERGE_MATCHING_TARGET_COLUMN,
                                              'I_%s' % modality])  # sort to keep closest and largest ImageUID image (line 352, 434)
    # add copy/move/symlink here
    for _, row in subj_result.iterrows():
        directory = os.path.join(output_directory, 'image', os.path.split(subj)[1], str(row['VISCODE_FIX']), modality)
        os.makedirs(directory, exist_ok=True)
        if not os.path.isfile(os.path.join(directory, os.path.split(row['%s_image_path' % modality])[1])):
            try:
                os.symlink(row['%s_image_path' % modality], os.path.join(directory, os.path.split(row['%s_image_path' % modality])[1]))  # symlink
            except OSError:
                pass  # symlink may fail on NFS, not critical

    # subj_result = subj_result.drop_duplicates('VISCODE_FIX', keep='last')  # 2023-09-12: to make intermediate csv, removed
    logging.debug('\t%s: %s visit point and %s image detected' % (
        ptid, len(subj_result['VISCODE_FIX'].unique()), len(image_list)))
    return subj_result


# -------------------- Utils -------------------- #


def _setup_logger(log_path, level=logging.DEBUG):
    logging.basicConfig(level=level,
                        format=FORMAT,
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])


def param_logging(**kwargs):
    for key, value in kwargs.items():
        logging.info('%s=%s' % (key, value))


def read_xml(path) -> dict:
    with open(path) as f:
        xml = ''.join(f.readlines())
    res = XmlTextToDict(xml).get_dict()
    return res[list(res.keys())[0]]


def safe_dict_search(dict, *args, default=''):
    """
    Search dict with keys in *args
    If fails, return ''.
    """
    res = deepcopy(dict)
    for key in args:
        if res == default:
            break
        res = res.get(key, default)
    return res


def get_apoe(apoe_list):
    """
    Converts apoe list to string
    If fails, return ''.
    """
    if isinstance(apoe_list, list):
        apoe = [e['#text'] for e in apoe_list]
        return 'e%s/e%s' % (apoe[0], apoe[1])
    return ''


def list2dict(inp: list):
    """
    Converts xml list to dict
    If fails, return ''.
    """
    try:
        res = dict()
        for e in inp:
            res[e['@term']] = e['#text']
        return res
    except:
        return ''


def float2str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return '%d' % value
    if isinstance(value, float):
        p = value % 1
        if p < 1e-8 or p > 1 - 1e-8:
            return '%d' % np.round(value)
    if pd.isna(value):
        return ''
    return str(value)


if __name__ == '__main__':
    adni = ADNI('/home/syh6087/ADNI_matching/ADNI_matching_231012_final', n_jobs=8)

    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/AV45',
                            '/home/BreinData/ADNI/new_ADNI_230810/AV45',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='av45',
                            overwrite=True)
    adni.attach_ucberkeley('/home/syh6087/ADNI_matching/ADNI_matching_231012_final/AV45_unique.csv',
                           '/home/syh6087/ADNI_db/UCBERKELEYAV45_8mm_02_17_23.csv')

    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/T1',
                            '/home/BreinData/ADNI/new_ADNI_230810/T1',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='t1',
                            overwrite=True)
    adni.attach_preprocess_path('/home/syh6087/ADNI_matching/ADNI_matching_231012_final/T1_unique.csv',
                                'FS_path',
                                '/home/BreinData/ADNI/ADNI_T1_2021_1013_fs_72/*_I{I_T1}',
                                'I_T1')

    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='flair',
                            overwrite=True,
                            regex='*[Ff][Ll][Aa][Ii][Rr]*')
    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='t2_fse',
                            overwrite=True,
                            regex='*[Ff][Ss][Ee]*')
    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='t2_tse',
                            overwrite=True,
                            regex='*[Tt][Ss][Ee]*')
    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/BreinData/ADNI/new_ADNI_230810/T2',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='t2_star',
                            overwrite=True,
                            regex=['*[Ss][Tt][Aa][Rr]*', '*[Gg][Rr][Ee]*'])

    adni.adnimerge_matching('/home/BreinData/ADNI/new_ADNI_230810/AV1451',
                            '/home/BreinData/ADNI/new_ADNI_230810/AV1451',
                            '/home/syh6087/ADNI_db/ADNIMERGE.csv',
                            180,
                            modality='av1451',
                            overwrite=True)

    adni.unique_csv_merge()
