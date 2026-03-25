# KMDB 영화 추천 시스템 -- 코사인 유사도 기반 하이브리드 임베딩

코사인 유사도 기반 **하이브리드(텍스트+메타데이터) 콘텐츠 필터링** 영화 추천 엔진.
KMDB(한국영화데이터베이스)에서 수집한 **19,354편(1980~2026.03)**의 영화를 **499차원 하이브리드 벡터**로 임베딩하여,
자유 텍스트 검색어 또는 테스트 영화에 대해 가장 유사한 영화 **Top-20**을 추천한다.

---

## 주요 특징

- **499차원 하이브리드 임베딩**: 장르(30D) + 키워드(80D) + 파생 수치(5D) + 텍스트 임베딩(384D), 그룹별 L2 정규화 + 독립 가중치
- **sentence-transformers**: `paraphrase-multilingual-MiniLM-L12-v2` 모델로 한국어 줄거리 텍스트를 384D 벡터로 인코딩
- **GPU 가속**: PyTorch CUDA로 StandardScaler, 코사인 유사도, KMeans, PCA SVD 연산 가속 (CPU fallback 지원)
- **27조합 민감도 분석**: 장르/키워드/텍스트 가중치를 상(1.5)/중(1.0)/하(0.5)으로 조합하여 추천 변화 정량 측정
- **인터랙티브 대시보드**: Dash 기반 한국어 UI, 5개 탭 (검색/클러스터/파라미터/평가/민감도), 실시간 가중치 조정
- **시각화 7종 + 종합 보고서**: Plotly 인터랙티브 HTML (Sankey, 2D/3D 산점도, 히트맵, 평가 차트, 기여도 분석, 민감도 분석) + 종합 분석 보고서(summary.html)

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 캐시 생성 (첫 실행 시 필수, 554 JSON 파싱)
python main.py --build-cache

# 3. 전체 파이프라인 실행 (6단계 + 시각화 6종)
python main.py

# 4. 대시보드 실행
python main.py --dashboard
# http://127.0.0.1:8050 에서 확인
```

---

## 아키텍처

### 6단계 파이프라인

```
Stage 1: 데이터 로드      (data_loader.py)    554 JSON 파싱, 장르 정규화(66->30), 파생 수치, pkl 캐시
    |
Stage 2: 임베딩 생성      (embedding.py)      499D 하이브리드 벡터 (그룹별 L2 정규화 + 가중치) + StandardScaler
    |
Stage 3: 군집화           (clustering.py)     KMeans(k=12) 클러스터 라벨 (GPU seed=42 결정성 보장)
    |
Stage 4: 차원 축소        (reduction.py)      PCA/t-SNE -> 2D/3D 좌표 (시각화용)
    |
Stage 5: 추천             (recommender.py)    코사인 유사도 Top-20 + 그룹별 기여도 분해
    |
Stage 6: 평가             (evaluator.py)      정량/정성 평가 + 텍스트 일관성 + 적합/부적합 판정
```

### 연산 흐름

```
KMDB 원본 (554 JSON, 19,354편)
    |
    v
data_loader.py -----> 장르 정규화 (66종 -> 30종), 키워드 매칭, 파생 수치 계산
    |                  pkl 캐시 (data/kmdb_processed.pkl)
    v
embedding.py -------> 4개 그룹 벡터 생성 + L2 정규화 + 가중치 적용
    |                  텍스트 임베딩 캐시 (data/text_embeddings.npy)
    |
    +-- raw_vectors (499D) -----> cosine_similarity -> Top-20 추천 목록
    |                                                      |
    +-- StandardScaler                                     v
         |          |                               그룹별 기여도 분해
         v          v                               (장르/키워드/수치/텍스트)
       KMeans      PCA                                     |
       (k=12)    (3D/2D)                                   v
         |          |                               평가 메트릭
         v          v                               (유사도, 정밀도, 다양성, 일관성)
       클러스터   좌표
       (0~11)   (시각화)
```

**핵심 설계**:
- **추천(Stage 5)**: raw 가중 벡터 간 코사인 유사도 (4개 그룹 가중치가 직접 반영)
- **군집화/PCA(Stage 3,4)**: scaled 벡터 사용 (피처 스케일 통일)
- **대시보드 가중치 조정**: `rebuild_with_weights()`로 텍스트 임베딩 재계산 없이 실시간 벡터 재생성

---

## 499차원 하이브리드 임베딩

### 벡터 구조

```
V = [ L2norm(genre) x Wg | L2norm(keyword) x Wk | L2norm(numeric) x Wn | L2norm(text) x Wt ]
         30D                    80D                     5D                    384D
```

각 그룹을 **독립적으로 L2 정규화** 한 후 가중치를 곱하고 연결(concatenate).
차원 수 차이(30D vs 384D)로 인한 편향을 방지하는 핵심 설계.

### 4개 그룹 상세

| 그룹 | 차원 | 인코딩 | 가중치 | 기본값 | 설명 |
|------|------|--------|--------|--------|------|
| 장르 | 30D | 원-핫 | WEIGHT_GENRE | 1.0 | KMDB 66종 -> 30개 표준 카테고리 매핑 |
| 키워드 | 80D | 바이너리 | WEIGHT_KEYWORD | 1.0 | 빈도 상위 80개 키워드 매칭 |
| 수치 | 5D | 연속값 0.0~1.0 | WEIGHT_NUMERIC | 0.5 | 파생 수치 특징 |
| 텍스트 | 384D | sentence-transformers | WEIGHT_TEXT | 1.5 | 줄거리 텍스트 임베딩 |

### 파생 수치 특징 5가지

| 특징 | 계산 | 범위 |
|------|------|------|
| runtime_norm | (상영시간 - 60) / 120, 결측 시 0.5 | 0.0 ~ 1.0 |
| year_norm | (연도 - 1980) / (2026 - 1980) | 0.0 ~ 1.0 |
| keyword_richness | min(매칭 키워드 수 / 10, 1.0) | 0.0 ~ 1.0 |
| cast_size_norm | min(출연진 수 / 20, 1.0) | 0.0 ~ 1.0 |
| genre_count_norm | 장르 수 / 5 | 0.0 ~ 1.0 |

### 장르 매핑 (66 -> 30)

KMDB 원본의 66개 장르를 30개 표준 카테고리로 정규화.
예: "멜로/로맨스" -> "로맨스", "로드무비" -> "모험", "기업/기관/단체" -> "기타"

---

## 데이터

### 학습 데이터
- **출처**: KMDB(한국영화데이터베이스) API
- **규모**: 19,354편 (1980년 ~ 2026년 3월)
- **원본 구조**: 554개 JSON 파일 + 19,791개 포스터 JPG (`Data_new/movies/`)
- **허용 유형**: 극영화, 애니메이션, 다큐멘터리 (ALLOWED_TYPES)
- **캐시**: `data/kmdb_processed.pkl` (파싱 결과), `data/text_embeddings.npy` (384D 텍스트 벡터)

### JSON 파일 구조
```
Data_new/movies/
  YYYY/
    json/
      YYYY_MM_movies.json    # 월별 영화 데이터
    posters/
      *.jpg                  # 포스터 이미지
```

### 결측치 처리

| 필드 | 조건 | 처리 |
|------|------|------|
| runtime | 0 또는 None | runtime_norm = 0.5 (중립값) |
| plot | 빈 문자열 | "영화" 대체 텍스트로 인코딩 |
| year | 파싱 실패 | 0 처리 |

### 테스트 영화

`select_test_movies()`가 최신 연도에서 다양한 장르의 영화 5편을 자동 선정.
`config.py`의 `TEST_MOVIE_TITLES`에 직접 지정도 가능.

현재 테스트 영화 (자동 선정 예시):

| 영화 | 연도 | 장르 |
|------|------|------|
| 프로텍터 | 2025 | 범죄, 액션 |
| 킴~치! | 2025 | 코미디, 드라마 |
| 왜(歪): 더 카르텔 | 2025 | 사회 |
| 프로젝트 헤일메리 | 2026 | SF |
| 드림스 | 2025 | 드라마, 로맨스 |

---

## 핵심 수식

### 벡터 생성 (그룹별 L2 정규화 + 가중치)
```
g_norm = g / ||g||_2              # 장르 30D L2 정규화
k_norm = k / ||k||_2              # 키워드 80D L2 정규화
n_norm = n / ||n||_2              # 수치 5D L2 정규화
t_norm = t / ||t||_2              # 텍스트 384D L2 정규화

V = concat([g_norm x Wg, k_norm x Wk, n_norm x Wn, t_norm x Wt])  = 499D
```

### StandardScaler (표준화)
```
mu_i = mean(X_i),  sigma_i = std(X_i)
X_scaled = (X - mu) / sigma       -> 평균 0, 분산 1
```
- 용도: 피처 스케일 통일 (군집화, PCA에 사용)

### 코사인 유사도 (추천 핵심)
```
cos_sim(A, B) = (A . B) / (||A|| x ||B||)

GPU 배치: S = normalize(Q) x normalize(T)^T  ->  S (M x N)
```
- **raw 가중 벡터** 사용 (그룹별 가중치 효과가 직접 반영)
- 그룹별 기여도 분해: 장르/키워드/수치/텍스트 각각의 코사인 유사도를 개별 계산

### KMeans 군집화
```
반복 (최대 300회):
  1) dist[i,c] = ||X_i - centroid_c||_2      <- 유클리드 거리
  2) label_i = argmin_c(dist[i,c])            <- 가장 가까운 중심
  3) centroid_c = mean(X[label==c])           <- 중심 갱신
  4) 수렴: max|old - new| < 1e-6 -> 종료
```
- GPU seed=42 고정으로 재현성 보장, k=12

### PCA 차원 축소
```
X_c = X_scaled - mu                           <- 중심화
U, S, V^T = SVD(X_c)                          <- 특이값 분해
X_reduced = X_c x V^T[:n_comp]^T              <- 투영 (N x 3)
explained_ratio[i] = S_i^2 / sum(S^2)         <- 분산 설명 비율
```

### Spearman 순위 상관계수
```
rho = 1 - 6 x sum(d_i^2) / (n x (n^2 - 1))
```
- 민감도 분석에서 기준선 대비 순위 변동 측정
- 범위: -1(역순) ~ 0(무관) ~ 1(동일 순서)

---

## 평가 메트릭

### 정량적 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| avg_similarity | mean(sim_1..sim_K) | 추천 영화의 평균 코사인 유사도 |
| genre_precision@K | (장르 일치 수) / K | 쿼리와 장르가 겹치는 비율 |
| keyword_precision@K | (키워드 일치 수) / K | 쿼리와 키워드가 겹치는 비율 |
| diversity | mean(1 - sim(rec_i, rec_j)) | 추천 목록 내 다양성 |
| text_coherence | mean(cos_sim(query_text, rec_text)) | 줄거리 텍스트 유사도 |
| group_contribution | 그룹별 코사인 유사도 평균 | 장르/키워드/수치/텍스트 기여도 |

### 적합/부적합 판정

아래 조건 중 **하나라도** 해당하면 부적합:

| 조건 | 임계값 |
|------|--------|
| genre_precision | < 0.3 |
| avg_similarity | < 0.5 |
| 평균 3D 거리 | > 10.0 |
| Top1 직관성 | "낮음" |

### 현재 평가 결과 (5편 테스트)

| 지표 | 값 |
|------|-----|
| 평균 유사도 | 0.700 |
| 장르 정밀도 | 1.000 |
| 다양성 | 0.377 |
| 텍스트 일관성 | 0.486 |
| 적합 판정 | 5/5 적합 |

---

## 민감도 분석 (27조합)

장르/키워드/텍스트 가중치를 **상(1.5)/중(1.0)/하(0.5)** 3단계로 조합하여 3^3 = **27가지** 경우 분석.
수치 가중치는 0.5로 고정. 기준선: (중, 중, 중) = (1.0, 1.0, 1.0).

### 비교 메트릭 (기준선 대비)

| 메트릭 | 수식 | 범위 | 의미 |
|--------|------|------|------|
| 겹침률 | \|기준 ∩ 변형\| / 20 | 0~1 | Top-20 목록 동일 비율 |
| Spearman rho | 1 - 6*sum(d^2)/(n(n^2-1)) | -1~1 | 순위 상관 (1=동일 순서) |
| 평균 순위 변동 | mean(\|rank_base - rank_var\|) | 0~ | 평균 순위 이동 폭 |

### 대시보드 진행률 표시

민감도 분석은 27조합을 순차 실행하므로 시간이 소요됨.
대시보드의 민감도 탭에서 실행 버튼 클릭 시 **실시간 진행률 바** 표시 (예: "분석 중... 15/27 (56%)").

---

## 대시보드 (5개 탭)

`python main.py --dashboard` 실행 후 http://127.0.0.1:8050 접속.

| 탭 | 이름 | 기능 |
|----|------|------|
| 1 | 검색 및 추천 | 자유 텍스트 검색 -> Top-20 영화 카드 (포스터, 장르, 유사도 바) |
| 2 | 클러스터 시각화 | 3D 산점도 + 클러스터별 영화 수/주요 장르(색상 배지)/주요 키워드 테이블 |
| 3 | 파라미터 제어 | 4개 가중치 슬라이더 실시간 조정, 기준선 대비 진입/이탈 영화 비교 |
| 4 | 평가 | 전체 메트릭 카드 + 영화별 상세 차트 + 적합/부적합 배지 |
| 5 | 민감도 분석 | 27조합 실행 버튼 + 실시간 진행률 바 + 히트맵 + 상세 결과 테이블 |

---

## 실행 명령어

| 명령어 | 설명 |
|--------|------|
| `python main.py --build-cache` | 데이터 캐시 생성 (첫 실행 시 필수, 554 JSON 파싱) |
| `python main.py` | 전체 파이프라인 (6단계 + 시각화 6종) |
| `python main.py --sensitivity` | 27조합 민감도 분석 (CLI) |
| `python main.py --dashboard` | Dash 대시보드 (http://127.0.0.1:8050) |
| `python main.py --sweep` | 파라미터 스윕 비교 |
| `python main.py --search` | 인터랙티브 영화 검색 (CLI) |
| `python main.py --examples` | 검색 예시 실행 |

---

## 출력 파일 (results/)

| 파일 | 내용 |
|------|------|
| `summary.html` | **종합 분석 보고서** -- 원리, 개념, 수식, 흐름, 구현, 평가, 한계까지 체계적 정리 |
| `data_field_diagram.html` | KMDB 필드 -> 499D 임베딩 매핑 Sankey 다이어그램 |
| `embedding_3d.html` | 3D 임베딩 산점도 (클러스터 색상 + 추천 연결선) |
| `embedding_2d.html` | 2D PCA 산점도 (클러스터 라벨 + 축 해석) |
| `similarity_heatmap.html` | 테스트 영화 x 추천 영화 코사인 유사도 히트맵 |
| `evaluation_report.html` | 평가 차트 (유사도, 정밀도, 다양성, 일관성, 적합 판정) |
| `weight_impact.html` | 그룹별 유사도 기여도 누적 막대 차트 |
| `sensitivity_analysis.html` | 27조합 민감도 분석 (차트 + 상세 테이블 + 순위 비교) |

---

## 프로젝트 구조

```
Test-Recommendation_Cosine-Similarity/
|-- main.py                 # CLI 통합 엔트리포인트
|-- config.py               # 중앙 파라미터 관리 (장르 매핑, 키워드, 가중치, 임계값)
|-- data_loader.py          # Stage 1: 554 JSON 파싱, 장르 정규화, 파생 수치, pkl 캐시
|-- embedding.py            # Stage 2: 499D 하이브리드 벡터 (L2 정규화 + 가중치 + StandardScaler)
|-- clustering.py           # Stage 3: KMeans(k=12) GPU 군집화
|-- reduction.py            # Stage 4: PCA/t-SNE 차원 축소
|-- recommender.py          # Stage 5: 코사인 유사도 Top-20 + 그룹별 기여도 분해
|-- evaluator.py            # Stage 6: 정량/정성 평가 + 적합/부적합 판정
|-- visualizer.py           # Plotly 시각화 7종 + 종합 보고서
|-- sensitivity.py          # 27조합 민감도 분석
|-- search.py               # 자유 텍스트 검색 엔진 (장르/키워드 별칭 + sentence-transformers)
|-- dashboard.py            # Dash 대시보드 (한국어 UI, 5탭)
|-- requirements.txt        # 의존성 목록
|-- Data_new/movies/        # KMDB 원본 데이터 (554 JSON + 포스터 JPG)
|-- data/
|   |-- kmdb_processed.pkl  # 파싱 캐시 (19,354편)
|   +-- text_embeddings.npy # 텍스트 임베딩 캐시 (19,349 x 384)
|-- results/                # HTML 시각화 8종 (종합 보고서 + 시각화 7종)
|-- CLAUDE.md               # AI 어시스턴트 컨텍스트
+-- README.md               # 이 파일
```

---

## 설정 파라미터 (config.py)

| 구분 | 파라미터 | 기본값 | 설명 |
|------|---------|--------|------|
| 임베딩 | WEIGHT_GENRE | 1.0 | 장르 벡터 가중치 |
| | WEIGHT_KEYWORD | 1.0 | 키워드 벡터 가중치 |
| | WEIGHT_NUMERIC | 0.5 | 수치 벡터 가중치 |
| | WEIGHT_TEXT | 1.5 | 텍스트 벡터 가중치 |
| 텍스트 | TEXT_MODEL_NAME | paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers 모델 |
| | TEXT_EMBED_DIM | 384 | 텍스트 임베딩 차원 |
| | TEXT_BATCH_SIZE | 64 | 인코딩 배치 크기 |
| 군집화 | KMEANS_N_CLUSTERS | 12 | KMeans 클러스터 수 |
| 차원 축소 | PCA_COMPONENTS | 3 | PCA 출력 차원 |
| | USE_TSNE | False | t-SNE 사용 여부 |
| 추천 | TOP_K | 20 | 추천 영화 수 |
| 평가 | THRESHOLD_AVG_SIMILARITY | 0.5 | 유사도 최소 임계값 |
| | THRESHOLD_GENRE_PRECISION | 0.3 | 장르 정밀도 최소 임계값 |
| | THRESHOLD_3D_DISTANCE | 10.0 | 공간 거리 최대 임계값 |
| 민감도 | SENSITIVITY_LEVELS | 하0.5 / 중1.0 / 상1.5 | 가중치 3단계 레벨 |
| | SENSITIVITY_TOP_K | 20 | 비교용 Top-K 크기 |

---

## 기술 스택

| 분류 | 패키지 | 용도 |
|------|--------|------|
| 핵심 연산 | NumPy, scikit-learn | 벡터 연산, StandardScaler, KMeans |
| 텍스트 임베딩 | sentence-transformers | 한국어 줄거리 384D 벡터 인코딩 |
| GPU 가속 | PyTorch (CUDA) | 코사인 유사도, KMeans, PCA SVD |
| 시각화 | Plotly | 인터랙티브 HTML 차트 6종 |
| 대시보드 | Dash | 한국어 웹 UI (5탭) |
| 이미지 | Pillow | 포스터 썸네일 생성 |
| 기타 | tqdm | 진행률 표시 |

---

## 주의사항

- `sentence-transformers` 미설치 시 텍스트 임베딩이 0벡터로 대체되어 추천 품질 저하
- `tensorflow`가 설치된 환경에서는 protobuf 호환성 충돌로 sentence-transformers import 실패 가능 -> `pip uninstall tensorflow`로 해결
- `config.py`의 ALL_GENRES/ALL_KEYWORDS 변경 시 벡터 차원이 변경됨 -> 기존 캐시 삭제 필요
- 데이터 캐시(`data/kmdb_processed.pkl`) 삭제 시 554 JSON 재파싱 필요 (약 15초)
- 텍스트 임베딩 캐시(`data/text_embeddings.npy`)는 영화 수 변경 시 자동 재생성 (GPU 기준 약 20초)

---

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
KMDB 데이터는 [한국영화데이터베이스 이용약관](https://www.kmdb.or.kr)을 따릅니다.
