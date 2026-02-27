# adnimerge_py

> LONI ADNIMERGE2 R 패키지의 .rda 추출 파이프라인을 Python으로 이식한 패키지.

ADNIMERGE2 R 패키지의 전체 로직을 Python/pandas로 1:1 이식하여,
.rda 원본 데이터에서 직접 ADNIMERGE CSV를 생성한다.

---

## 패키지 구조

| 파일 | 줄 수 | 역할 |
|------|-------|------|
| `__init__.py` | 3 | 패키지 메타 (`__version__ = '1.0.0'`) |
| `__main__.py` | 4 | `python -m adnimerge_py` 진입점 |
| `rda_converter.py` | 112 | .rda → CSV 일괄 변환기 |
| `build_adnimerge.py` | 1,171 | ADNIMERGE/MRIQC/APOERES 빌드 로직 |
| `compare_ref.py` | 167 | REF vs NEW CSV 비교 유틸리티 |
| `run.py` | 122 | CLI 인터페이스 (argparse) |

### rda_converter.py

- **`convert_single_rda(rda_path, output_dir)`** — 단일 .rda → CSV 변환. pyreadr로 로드 후 첫 번째 DataFrame을 CSV로 저장.
- **`convert_all_rda(rda_dir, output_dir)`** — 디렉토리 내 모든 .rda 파일을 일괄 변환 (217개).
- **`print_report(results)`** — 변환 결과 리포트 (OK/error/skipped 집계).

### build_adnimerge.py

- **`build_adnimerge(rda_dir, output_dir, date_str)`** — 12단계 빌드 프로세스로 ADNIMERGE CSV 생성.
- 헬퍼: `load_rda()`, `standardize_viscode()`, `convert_ecog_to_numeric()`, `first_non_na()`, `_group_first_non_na()`

### compare_ref.py

- **`compare_csvs(ref_path, new_path, cols, tolerance)`** — PTID+VISCODE 기준 inner join → 컬럼별 일치율, 불일치 건수, Pearson r.
- CLI: `python adnimerge_py/compare_ref.py REF.csv NEW.csv --cols DX_bl MMSE`

### run.py

- argparse 기반 CLI. 플래그 미지정 시 전체 실행 (`--all`과 동일).
- `python -m adnimerge_py` 또는 `python adnimerge_py/run.py`로 실행.

---

## 의존성 및 실행 방법

### 의존성

```
pyreadr    # .rda 파일 읽기
pandas     # 데이터 병합/변환
numpy      # 수치 연산
```

### 실행

```bash
# 전체 실행 (rda 변환 + ADNIMERGE + MRIQC + APOERES)
python -m adnimerge_py

# 개별 단계만 실행
python -m adnimerge_py --build-adnimerge
python -m adnimerge_py --convert-all

# 옵션 지정
python -m adnimerge_py --rda-dir /path/to/ADNIMERGE2/data --output-dir /path/to/csv --date 260212 -v
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--convert-all` | 217개 .rda → CSV 일괄 변환 | — |
| `--build-adnimerge` | ADNIMERGE_{DATE}.csv 빌드 | — |
| `--all` | 전체 실행 (플래그 미지정 시 기본) | — |
| `--rda-dir DIR` | .rda 소스 디렉토리 | `ADNIMERGE2/data` |
| `--output-dir DIR` | 출력 디렉토리 | `csv/` |
| `--date YYMMDD` | 출력 파일명 날짜 | 오늘 날짜 |
| `-v, --verbose` | DEBUG 로그 출력 | OFF |

---

## 출력 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `csv/tables/*.csv` | 217개 | .rda 원본 1:1 CSV 변환 (MRIQC.csv, APOERES.csv 포함) |
| `csv/ADNIMERGE_{DATE}.csv` | 1개 | 12단계 빌드 최종 결과 |

---

## 12단계 빌드 프로세스

`build_adnimerge()` 함수의 12단계:

| 단계 | 내용 | 주요 소스 테이블 |
|------|------|-----------------|
| 1 | 코어 데이터셋 로드 | ADSL, REGISTRY, DXSUM, PTDEMOG, APOERES, ARM 등 22개 |
| 2 | 베이스 프레임 생성 (REGISTRY) | REGISTRY — VISCODE 표준화, sc/scmri/v01/nv → bl |
| 3 | 인구통계 + EXAMDATE_bl | PTDEMOG, APOERES, ADSL — **enrollment visit 기준** EXAMDATE_bl |
| 4 | 진단 (DX) | DXSUM + ARM.rda — DX_bl에 EMCI/LMCI/SMC 유지 (ADNI1/GO/2) |
| 5 | 인지검사 점수 | ADAS, MMSE, CDR, MOCA, NEUROBAT, FAQ, ECOGPT, ECOGSP |
| 6 | CSF 바이오마커 | UPENNBIOMK_MASTER + ROCHE_ELECSYS (ABETA/TAU/PTAU) |
| 7 | MRI 용적 | UCSFFSX51 (FreeSurfer 5.1, primary) + UCSDVOL (Ventricles only) |
| 8 | PET 데이터 | UCBERKELEYFDG_8mm, AMY_6MM, TAU_6MM |
| 9 | 혈장 바이오마커 (5개 플랫폼) | UPENNPLASMA, C2N, FUJIREBIO_QUANTERIX, BLENNOW, UGOT |
| 10 | 파생 변수 계산 | Baseline 인지점수 coalesce, Month, SITE, mPACC |
| 11 | 132개 컬럼 최종 스키마 선택 및 정렬 | — |
| 12 | CSV 저장 | ADNIMERGE_{DATE}.csv |

### 주요 로직 상세

- **Step 2**: REGISTRY에서 RID-VISCODE별 1행 유지. `bl` 방문이 `sc`보다 우선 (LONI 제공 CSV 동작 재현).
  ADNI4 VISCODE 형식 (`4_sc`, `4_bl`, `4_m12`) 자동 변환.
- **Step 3**: `EXAMDATE_bl` = enrollment visit (`VISCODE='bl'`) 기준.
  `bl` 행이 없는 피험자는 `min(EXAMDATE)` fallback 사용.
- **Step 4**: DX는 DXSUM `DIAGNOSIS` 기반 (CN/MCI/Dementia).
  DX_bl은 ADSL 우선, DXSUM fallback. **ARM.rda**로 ADNI1/GO/2의 enrollment group(EMCI/LMCI/SMC) 유지.
- **Step 9**: 5개 독립 플랫폼에서 혈장 바이오마커 15개 컬럼 추가:
  UPenn (AB40/AB42), C2N PrecivityAD2 (pTau217/AB42/AB40/ratio/APS2),
  Fujirebio+Quanterix (pTau217/AB42/AB40/ratio/NFL/GFAP), Blennow NFL, UGOT pTau181.
- **Step 11**: 132개 컬럼 고정 스키마 — 없는 컬럼은 NA로 생성.

---

## LONI 레퍼런스 대비 재현성 검증

`csv/ref/ADNIMERGE_240821.csv` (LONI 사이트 제공, 2024-08-21)와
`csv/ADNIMERGE_260213.csv` (Python 빌드)를 114개 공통 컬럼에 대해 비교 검증.

### 규모 비교

| 항목 | 레퍼런스 (LONI) | Python 빌드 | 비고 |
|------|-------------|------------|------|
| 행 수 | 16,421 | 23,479 | +43% (ADNI4 + 연장 방문 포함) |
| 열 수 | 116 | 132 | +16 (혈장 바이오마커 등) |
| 피험자 | 2,430 | 4,498 | +85% (ADNI4 신규 등록) |

### 컬럼 비교

- **공통**: 114개
- **레퍼런스에만**: `IMAGEUID`, `IMAGEUID_bl` (2개) — 매칭 파이프라인에서 별도 처리하므로 미포함
- **Python에만**: 18개 — 아래 표 참조

| 추가 컬럼 | 유형 |
|-----------|------|
| `PLASMA_AB40_UPENN`, `PLASMA_AB42_UPENN` | UPenn 혈장 |
| `PLASMA_PTAU217_C2N`, `PLASMA_AB42_C2N`, `PLASMA_AB40_C2N`, `PLASMA_AB42_40_RATIO_C2N`, `PLASMA_APS2_C2N` | C2N PrecivityAD2 혈장 |
| `PLASMA_PTAU217_F`, `PLASMA_AB42_F`, `PLASMA_AB40_F`, `PLASMA_AB42_40_RATIO_F` | Fujirebio 혈장 |
| `PLASMA_NFL`, `PLASMA_GFAP` | Quanterix 혈장 |
| `PLASMA_NFL_BLENNOW` | Blennow NFL 혈장 |
| `PLASMA_PTAU181` | UGOT pTau181 혈장 |
| `AMY_LONIUID`, `TAU_LONIUID` | PET 식별자 |
| `TAU_META_TEMPORAL_SUVR` | Tau PET SUVR |

### 불일치 범주 분류

PTID + VISCODE 기준 16,411개 공통 행에서, 양쪽 모두 non-NA인 값 기준으로 비교.
불일치 원인은 3개 범주로 분류된다.

---

### (A) 핵심 임상 변수 — 동일 로직 (99~100% 일치)

| 변수 | 비교 건수 | 일치율 | 불일치 건수 | 불일치 원인 |
|------|----------|--------|-----------|-----------|
| SITE, PTEDUCAT, PTETHCAT | 16,411 | **100%** | 0 | — |
| PTMARRY, RAVLT_immediate, RAVLT_learning, DIGITSCOR | 3,800~16,410 | **100%** | 0 | — |
| RAVLT_forgetting | 11,314 | **99.9%** | 13 | 데이터 업데이트 |
| CDRSB, LDELTOTAL | 9,441~11,745 | **99.99%** | 1 | 데이터 업데이트 |
| FAQ | 11,741 | **99.97%** | 4 | 데이터 업데이트 |
| TRABSCOR | 11,008 | **99.98%** | 2 | 데이터 업데이트 |
| DX (방문별 진단) | 11,457 | **99.90%** | 11 | sc→bl 병합 순서 |
| APOE4 | 16,046 | **99.90%** | 16 | APOERES.rda 버전 차이 |
| PTGENDER | 16,411 | **99.96%** | 7 | PTDEMOG 2024→2026 업데이트 |
| MMSE | 11,468 | **98.51%** | 171 | .rda 2024→2026 데이터 업데이트 |
| DX_bl (기저 진단) | 16,400 | **91.3%** | 1,432 | 아래 (B) 참조 |
| MOCA | 2,447 | **90.03%** | 244 | .rda 2024→2026 데이터 업데이트 |

MMSE, MOCA의 불일치는 소스 테이블(.rda)의 RID-VISCODE 중복이 아닌
**2024→2026 데이터 업데이트**에 의한 것임을 확인 (각 테이블 RID-VISCODE 중복 1쌍 이하).

#### 개발 과정 참고 (로직 버그 수정 2건)

빌드 과정에서 RAVLT_forgetting (잘못된 컬럼 참조: `AVDELTOT`→`AVDEL30MIN`)과
EcogTotal (VISSPAT 항목 누락 + mean-of-means 오류)을 발견하여 수정 완료.
상세 이력은 `docs/ADNIMERGE_reproducibility_audit.md` 참조.

---

### (B) DX_bl — ARM.rda 연동 (91.3% 일치)

#### 현재 구현

- **ADNI1/GO/2**: `ARM.rda` enrollment group 기준 (EMCI/LMCI/SMC/CN/AD 유지)
- **ADNI3/4**: `ADSL.DX` + `DXSUM.DIAGNOSIS` 기준 (CN/MCI/AD만 가능)

ARM.rda의 `ARM` 컬럼 prefix로 enrollment group 파싱:
NL→CN, MCI(ADNI1)→LMCI, EMCI→EMCI, LMCI→LMCI, SMC→SMC, AD→AD.

#### 잔여 불일치 (1,432건 = 8.7%)

| REF | NEW | 건수 | 프로토콜 | 설명 |
|-----|-----|------|---------|------|
| SMC | CN | 814 | ADNI3 | ARM.rda에 ADNI3 미포함 |
| LMCI | MCI | 321 | ADNI3 | ARM.rda에 ADNI3 미포함 |
| EMCI | MCI | 289 | ADNI3 | ARM.rda에 ADNI3 미포함 |
| LMCI | SMC | 8 | ADNI2 | ARM.rda 자체 1건 차이 |

ADNI3/4의 EMCI/LMCI/SMC 구분은 ADNIMERGE2 패키지에 포함되지 않음.
구버전 ADNIMERGE(v1)에만 존재하던 내부 매핑으로, 재현할 수 없는 외부 의존성.

#### Month_bl

- **REF**: 소수값 (0.33, 0.46 등) — screening→enrollment 간격 반영
- **NEW**: 항상 0 — enrollment visit 자체가 기준점
- 정의 차이이며 Month/M도 이에 연쇄하여 차이 발생.

---

### (C) 데이터 소스 차이 — ADNIMERGE2 설계에 따른 소스 선택

동일 컬럼명이지만 **측정 플랫폼, 파이프라인 버전, 또는 빌드 시점**이 다른 경우.

#### AGE (~60% 일치, r≈1.000)

REF와 NEW 모두 ADSL `AGE` 기반. 소수점 반올림 차이 (평균 0.04세, 최대 0.1세).

#### ADAS11 / ADAS13 (~69% 일치, r=0.999)

- **REF**: 대부분 정수 (63.3%). **NEW**: 소수점 포함 (Q4 sub-scoring 반영)
- `round()` 적용 시 99.7% 일치 → 동일 소스, 정밀도만 다름.

#### FDG (~6% 일치, r=0.847)

- **REF**: 구버전 FDG 파이프라인 (`UCBERKELEYFDG`)
- **NEW**: `UCBERKELEYFDG_8mm.rda` MetaROI (최신 파이프라인)
- FDG PET 처리 파이프라인 세대교체에 의한 차이 (-0.063 SUVR 평균).

#### AV45 (~24% 일치, r=0.998)

- **REF**: 구버전 AV45 SUVR (8mm). **NEW**: `UCBERKELEY_AMY_6MM.rda` SUMMARY_SUVR
- 해상도(8mm→6mm) 및 reference region 업데이트.

#### ABETA / TAU / PTAU (~7-10% 일치)

- **현재 소스**: `UPENNBIOMK_MASTER` + `ROCHE_ELECSYS` (Luminex + Elecsys 병합)
- **REF 소스**: 구버전 `UPENNBIOMK` (Luminex 기반)
- **MASTER만 사용 시**: ABETA r=0.83, TAU r=0.94, PTAU r=0.71 (exact match ~0-1%)
  — MASTER 자체가 Luminex+Elecsys 혼합 데이터셋이므로 Luminex 단독 분리가 불가능.
- **현재 소스를 선택한 이유**: ADNIMERGE2 R 패키지 설계에 따름.
  ADNI3 이후 Elecsys로 전환되어 Luminex만 사용 시 최신 데이터 누락.
  ABETA ratio≈0.22는 Elecsys의 알려진 변환계수.
- 참조: [Shaw et al. 2019](https://www.nature.com/articles/s41598-019-54204-z),
  [UPenn Roche Elecsys Methods](https://adni.bitbucket.io/reference/docs/UPENNBIOMK_ADNIDIAN_ES_2017/ADNI_Methods_Template_Shaw_2017_2018_Roche_Elecsys_ADNI_CSFs_in_DIAN_ADNI_study_3_29_2018_v3.pdf)

#### MRI 용적 — Hippocampus/ICV ~93%, 그 외 ~0%

- **현재 소스**: `UCSFFSX51` (FreeSurfer 5.1) primary + `UCSDVOL` Ventricles only
- **REF 소스**: `UCSFFSX` (FreeSurfer cross-sectional, 구버전)
- **소스별 매치율 비교** (REF 대비):

| 구성 | Hippocampus | ICV | WholeBrain | Ventricles | Entorhinal 등 |
|------|------------|-----|------------|------------|---------------|
| UCSDVOL 우선 + UCSFFSX51 fallback | 57% (r=0.988) | ~0% | ~0% | ~0% | ~0% |
| **UCSFFSX51 primary (현재)** | **93.3%** (r=0.998) | **93.4%** (r=0.997) | ~0% (r=0.715) | ~0% (r=0.995) | ~0% (r=0.5) |
| UCSFFSX (구버전) | 98.1% (r=1.000) | — | ~0% | — | ~0% |

- **현재 소스를 선택한 이유**: coverage가 가장 높은 단일 소스를 사용하여 측정 기준 일관성을 유지.
  UCSFFSX51: 4,896행, 1,067명 / UCSDVOL: 3,128행, 811명 / UCSFFSX: 1,138행, 573명.
  UCSDVOL은 Ventricles(lateral ventricle) 전용으로만 사용 — FreeSurfer에는 lateral ventricle 컬럼이 없음.
- **WholeBrain/Entorhinal/Fusiform/MidTemp ~0% 원인**: FreeSurfer 버전(4.x→5.1) 간
  atlas 차이 + ROI 정의 변경. r=0.5~0.7로 상관은 있으나 절대값이 다름.
- **Ventricles ~0% (r=0.995) 원인**: UCSDVOL과 REF(UCSFFSX)는 서로 다른 파이프라인.
  r≈1.0으로 거의 완벽한 상관이지만 절대값 차이 존재.
- 참조: [UCSFFSX 문서](https://adni.bitbucket.io/reference/ucsffsx.html),
  [UCSDVOL 문서](https://adni.bitbucket.io/reference/ucsdvol.html)

---

### 검증 요약

| 범주 | 변수 | 일치율 | 비고 |
|------|------|--------|------|
| **(A) 핵심 임상** | 인구통계 (SITE, PTEDUCAT, PTETHCAT 등) | **100%** | — |
| **(A) 핵심 임상** | 인지검사 (CDRSB, FAQ, TRABSCOR 등) | **99.9%+** | 데이터 업데이트 |
| **(A) 핵심 임상** | MMSE | **98.5%** | 데이터 업데이트 |
| **(A) 핵심 임상** | MOCA | **90.0%** | 데이터 업데이트 |
| **(B) DX_bl** | DX_bl (ARM.rda 연동) | **91.3%** | ADNI3 ARM 부재 (8.7%) |
| **(C) 소스 차이** | ADAS11/13 | ~69% | r=0.999, 소수점 정밀도 |
| **(C) 소스 차이** | AGE | ~60% | r≈1.000, 소수점 반올림 |
| **(C) 소스 차이** | Hippocampus / ICV | ~93% | r≈0.998, UCSFFSX51 primary |
| **(C) 소스 차이** | AV45 | ~24% | r=0.998, 해상도/reference 변경 |
| **(C) 소스 차이** | ABETA/TAU/PTAU | ~7-10% | Luminex→Elecsys 플랫폼 교체 |
| **(C) 소스 차이** | FDG | ~6% | r=0.847, 파이프라인 세대교체 |
| **(C) 소스 차이** | MRI 용적 (그 외) | ~0% | FreeSurfer 버전/파이프라인 차이 |

> **결론**: 핵심 임상 변수(인구통계, 진단, 인지검사)는 **90~100% 일치**.
> DX_bl은 ARM.rda 연동으로 EMCI/LMCI/SMC를 유지하여 40.5%→**91.3%** 개선 (잔여 8.7%는 ADNI3 ARM 부재).
> MRI 용적은 UCSFFSX51 단일 소스로 전환하여 Hippocampus 57%→**93.3%**, ICV 0%→**93.4%** 개선.
> 바이오마커/MRI 차이는 데이터 소스 설계에 의한 것이며,
> 각 소스의 대안 매치율을 실측하여 선택 근거를 확인하였다.

---

## 참고 문헌

### 프로젝트 내부 문서

| 문서 | 위치 |
|------|------|
| ADNIMERGE2 Methods | `ADNIMERGE2/ADNIMERGE2_R_Package_Methods_20260105.pdf` |
| RAVLT scoring R 코드 | `ADNIMERGE2/R/score-function.R` L24-54 |
| UPENNBIOMK_MASTER.Rd | `ADNIMERGE2/man/UPENNBIOMK_MASTER.Rd` |
| UPENNBIOMK_ROCHE_ELECSYS.Rd | `ADNIMERGE2/man/UPENNBIOMK_ROCHE_ELECSYS.Rd` |
| UCSDVOL.Rd | `ADNIMERGE2/man/UCSDVOL.Rd` |
| UCSFFSX51.Rd | `ADNIMERGE2/man/UCSFFSX51.Rd` |
| ADNIMERGE 재현성 감사 | `docs/ADNIMERGE_reproducibility_audit.md` |
| ADNIMERGE 컬럼 사전 | `docs/ADNIMERGE_columns.md` |

### 외부 참조

| 주제 | 링크 |
|------|------|
| ADNIMERGE2 R 패키지 | [atri-biostats.github.io/ADNIMERGE2](https://atri-biostats.github.io/ADNIMERGE2) |
| ADNIMERGE2 GitHub | [github.com/atri-biostats/ADNIMERGE2](https://github.com/atri-biostats/ADNIMERGE2/) |
| CSF Luminex→Elecsys 검증 | [Shaw et al. 2019, Sci Rep](https://www.nature.com/articles/s41598-019-54204-z) |
| CSF Elecsys 분석 메소드 | [UPenn Roche Elecsys Methods (PDF)](https://adni.bitbucket.io/reference/docs/UPENNBIOMK_ADNIDIAN_ES_2017/ADNI_Methods_Template_Shaw_2017_2018_Roche_Elecsys_ADNI_CSFs_in_DIAN_ADNI_study_3_29_2018_v3.pdf) |
| CSF 분석 일자 요약 | [ADNI CSF Biomarker Analysis Dates](https://adni.loni.usc.edu/summary-of-dates-of-csf-biomarker-analysis/) |
| AV45 처리 메소드 (2021) | [Jagust Lab AV45 Methods (PDF)](https://adni.bitbucket.io/reference/docs/UCBERKELEYAV45/UCBERKELEY_AV45_Methods_11.15.2021.pdf) |
| AV45 초기 메소드 (2015) | [Landau & Jagust AV45 Methods (PDF)](https://adni.bitbucket.io/reference/docs/UCBERKELEYAV45/ADNI_AV45_Methods_JagustLab_06.25.15.pdf) |
| ADNI PET Core 종합 | [Jagust et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4510459/) |
| ADNI PET Core 20주년 | [Jagust et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11485322/) |
| ADNI Biomarker Core 리뷰 | [Shaw et al. 2025, Alz & Dem](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.14264) |
| UCSFFSX 문서 | [adni.bitbucket.io/reference/ucsffsx.html](https://adni.bitbucket.io/reference/ucsffsx.html) |
| UCSDVOL 문서 | [adni.bitbucket.io/reference/ucsdvol.html](https://adni.bitbucket.io/reference/ucsdvol.html) |
| alzverse 논문 | Donohue et al. arXiv:2510.02318 (2025) — "Alzheimer's Clinical Research Data via R Packages: the alzverse" |
