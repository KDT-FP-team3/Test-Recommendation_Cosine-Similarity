# AI 영화 추천 시스템 (Recommendation_AI)

코사인 유사도 기반 **콘텐츠 필터링(Content-Based Filtering)** 영화 추천 엔진.
TMDB/KOBIS에서 크롤링한 **10,857편(1980.01~2026.03)**의 영화를 54차원 벡터로 임베딩하여,
신작 영화 또는 자유 텍스트 검색어에 대해 가장 유사한 기존 영화 Top-K를 추천한다.

---

## 주요 특징

- **54차원 가중 임베딩**: 장르(17D) + 키워드(30D) + 수치특징(7D), 카테고리별 독립 가중치
- **GPU 가속**: PyTorch CUDA로 StandardScaler, 유사도 행렬, KMeans 연산 가속 (선택, CPU fallback 지원)
- **27조합 민감도 분석**: 가중치를 상(1.5)/중(1.0)/하(0.5)로 조합하여 추천 변화 정량 측정
- **인터랙티브 대시보드**: Dash 기반 실시간 파라미터 조정 + 5개 탭 (시각화/추천/평가/스윕/민감도)
- **종합 시각화**: Plotly 인터랙티브 차트 7개 HTML 생성
- **종합 보고서**: 14개 섹션의 summary.html (수식 체계, 연산 흐름도, 민감도 상세 분석 포함)

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 기본 파이프라인 실행 (6단계 + 시각화 5개)
python main.py

# 3. 민감도 분석 (27조합, 별도 실행)
python main.py --sensitivity

# 4. 결과 확인
# results/ 폴더의 HTML 파일을 브라우저에서 열기
# results/summary.html → 종합 보고서
```

---

## 아키텍처

### 6단계 파이프라인

```
Stage 1: 데이터 로드    (data_loader.py)   10,857편 검증/결측치 처리
    |
Stage 2: 임베딩 생성    (embedding.py)     54D 가중 벡터 + StandardScaler
    |
Stage 3: 군집화         (clustering.py)    KMeans(k=8) 클러스터 라벨 (GPU seed=42)
    |
Stage 4: 차원 축소      (reduction.py)     PCA 2D/3D 좌표 (시각화용)
    |
Stage 5: 추천           (recommender.py)   코사인 유사도 Top-K
    |
Stage 6: 평가           (evaluator.py)     정량/정성 메트릭 + 적합 판정
```

### 연산 흐름 (벡터 분기)

```
영화 데이터 (10,857편)
    |
    v
movie_to_vector() -----> raw_vectors (54D)  --+-- Wg, Wk, Wn 가중치 적용
    |                                          |
    v                                          v
StandardScaler --------> scaled_vectors    cosine_similarity (raw 벡터 사용)
    |          |                                |
    v          v                                v
 KMeans      PCA                          Top-K 추천 목록
 (scaled)   (scaled)                            |
    |          |                                v
    v          v                          평가 메트릭
 클러스터   2D/3D 좌표                   (유사도, 정밀도, 다양성)
 (0~7)     (시각화)
```

**핵심 설계**:
- **추천(Stage 5)**: raw 가중 벡터 간 코사인 유사도 (가중치 Wg, Wk, Wn 효과가 직접 반영)
- **군집화/PCA(Stage 3,4)**: scaled 벡터 사용 (피처 스케일 통일)
- **민감도 분석**: 가중치(Wg, Wk, Wn)를 변경하여 Stage 1~6 전체를 27회 반복

---

## 54차원 임베딩 벡터

```
V = [ genre_onehot(17) x Wg | keyword_binary(30) x Wk | numeric(7) x Wn ]
```

| 카테고리 | 차원 | 인코딩 | 가중치 | 기본값 |
|---------|------|--------|--------|--------|
| 장르 | 17D | 원-핫 (Action, Drama, Sci-Fi 등 17개) | WEIGHT_GENRE | 1.0 |
| 키워드 | 30D | 바이너리 (space, revenge, love 등 30개) | WEIGHT_KEYWORD | 1.0 |
| 수치 | 7D | 연속값 0.0~1.0 | WEIGHT_NUMERIC | 1.0 |

### 수치 특징 7가지

| 특징 | 범위 | 의미 |
|------|------|------|
| mood | 0.0~1.0 | 밝음(0) -> 어두움(1) |
| tempo | 0.0~1.0 | 느린 전개(0) -> 빠른 전개(1) |
| visual_style | 0.0~1.0 | 사실적(0) -> 환상적(1) |
| star_power | 0.0~1.0 | 배우 인지도 |
| critic_score | 0.0~1.0 | 평론가 점수 (결측 시 0.5) |
| audience_score | 0.0~1.0 | 관객 점수 (결측 시 0.5) |
| budget_scale | 0.0~1.0 | 저예산(0) -> 블록버스터(1) |

---

## 데이터

### 학습 데이터
- **출처**: TMDB API + KOBIS API
- **규모**: 10,857편 (1980년 1월 ~ 2026년 3월 한국 개봉 영화)
- **크롤링 필드**: 24개 (임베딩용 9개 + 식별자 2개 + 메타 1개 + 참조 12개)
- **저장**: `data/movies.json`
- **크롤링 재개**: `python crawler.py --resume` (체크포인트 기반 이어받기)

### 결측치 처리
| 필드 | 조건 | 처리 |
|------|------|------|
| critic_score | None | 0.5 (중립값) |
| audience_score | None | 0.5 (중립값) |
| runtime | 0 또는 None | None (0분은 불가하므로 결측) |
| budget_usd | 0.0 | budget_scale=0.10 (결측) |

### 테스트 영화 (2026년 개봉 예정 5편)

| 영화 | 장르 | 국가 |
|------|------|------|
| Iron Lung | Sci-Fi, Horror | US |
| HUMINT (휴민트) | Action, Thriller | KR |
| Nuremberg | Drama, History, Thriller | US |
| Peaky Blinders: The Immortal Man | Crime, Drama, History | UK |
| Ready or Not 2 | Comedy, Horror, Thriller | US |

> Heartman(하트맨, 2026-02-11 개봉)과 Project Hail Mary(프로젝트 헤일메리, 2026-03-15 개봉)는
> 이미 개봉되어 학습 데이터로 이동하였다.

---

## 핵심 수식

### 벡터 생성
```
V = concat([genre_onehot x Wg, keyword_binary x Wk, numeric x Wn])
    (17D)              (30D)                (7D)       = 54D
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
- 용도: 쿼리 영화와 학습 영화 간 유사도 계산, Top-K 순위 결정
- **raw 가중 벡터** 사용 (가중치 효과 직접 반영)

### KMeans 군집화
```
반복 (최대 300회):
  1) dist[i,c] = ||X_i - centroid_c||_2      <- 유클리드 거리
  2) label_i = argmin_c(dist[i,c])            <- 가장 가까운 중심
  3) centroid_c = mean(X[label==c])           <- 중심 갱신
  4) 수렴: max|old - new| < 1e-6 -> 종료
```
- GPU seed=42 고정으로 재현성 보장

### PCA 차원 축소
```
X_c = X_scaled - mu                           <- 중심화
U, S, V^T = SVD(X_c)                          <- 특이값 분해
X_reduced = X_c x V^T[:n_comp]^T              <- 투영 (N x 2 또는 N x 3)
explained_ratio[i] = S_i^2 / sum(S^2)         <- 분산 설명 비율
```
- 축 해석: 각 주성분에서 |가중치|가 큰 상위 피처 추출

### Spearman 순위 상관계수
```
rho = 1 - 6 x sum(d_i^2) / (n x (n^2 - 1))
```
- 공통 영화를 1~n으로 재순위(re-rank) 후 계산
- 범위: -1(역순) ~ 0(무관) ~ 1(동일 순서)

---

## 평가 메트릭

### 정량적 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| avg_similarity | mean(sim_1..sim_K) | 추천 영화의 평균 유사도 |
| genre_precision@K | (장르 일치 수) / K | 장르 기반 정밀도 |
| keyword_precision@K | (키워드 일치 수) / K | 키워드 기반 정밀도 |
| diversity | mean(1 - sim(rec_i, rec_j)) | 추천 목록 다양성 |
| std_similarity | std(전체 유사도) | 추천 일관성 |

### 적합/부적합 판정

아래 조건 중 **하나라도** 해당하면 부적합:
- genre_precision < 0.4
- avg_similarity < 0.6
- 평균 3D 거리 > 4.0
- Top1 직관성 = "낮음"

---

## 민감도 분석 (27조합)

장르/키워드/수치 가중치를 **상(1.5)/중(1.0)/하(0.5)** 3단계로 조합하여 3^3 = **27가지** 경우 분석.
각 조합마다 전체 파이프라인(Stage 1~6)을 실행하여 기준선(중,중,중) 대비 추천 변화를 측정.

### 분석 대상 (10개 쿼리)

**테스트 영화 5편** + **자유 텍스트 검색 5개**:
1. "어두운 분위기의 SF 우주 생존 영화"
2. "밝은 로맨틱 코미디 사랑 이야기"
3. "빠른 전개의 첩보 액션 스파이 스릴러"
4. "역사적 전쟁 드라마 심리극"
5. "화려한 블록버스터 판타지 모험 마법"

### 비교 메트릭 (기준선 대비)

| 메트릭 | 수식 | 범위 | 의미 |
|--------|------|------|------|
| 겹침률 | \|기준 ∩ 변형\| / 20 | 0~1 | Top-20 목록 동일 비율 |
| Spearman rho | 1 - 6*sum(d^2)/(n(n^2-1)) | -1~1 | 순위 상관 (1=동일 순서) |
| 순위 변동 | mean(\|rank_base - rank_var\|) | 0~ | 평균 순위 이동 폭 |
| 신뢰도 sigma | std(5편의 avg_similarity) | 0~ | 낮을수록 안정적 |

### 4개 분석 차트

| 차트 | 내용 |
|------|------|
| 평균 유사도 | 27조합을 유사도 내림차순 정렬. 어떤 가중치 조합이 가장 정확한 추천을 하는지 |
| Top-20 겹침률 | 기준선 대비 같은 영화가 포함된 비율. 가중치 변화에 목록이 얼마나 안정적인지 |
| Spearman 순위상관 | 공통 영화의 순서 유지 정도. 1.0에 가까울수록 순서 변동 없음 |
| 신뢰도 sigma | 테스트 5편 간 편차. 낮을수록 모든 영화에 균일한 추천 |

### 27조합 상세 비교표

- 셀 배경색: 녹색(양호) / 노랑(보통) / 빨강(미흡)
- 가중치 열: 진한 파랑(상1.5) / 중간 파랑(중1.0) / 연한 파랑(하0.5)
- 기준선 행: 노란 하이라이트

### 순위 비교 테이블

- 기준선(중,중,중) vs 최고 유사도 조합 vs 최저 유사도 조합의 Top-20 나란히 비교
- 각 영화에 클러스터 번호(C0~C7) 표시, 같은 클러스터는 동일 배경색
- 순위 변동: 상승/하락/NEW 표시

### 주요 발견

- **장르 가중치(Wg)**가 추천 정확도에 가장 큰 영향 (17D 원-핫이 코사인 유사도에 지배적)
- 가중치 소폭 변화 시 Top-20 목록의 **90% 이상 유지** (안정적)
- **수치 가중치(Wn) 증가가 가장 불안정**: 7D 수치 블록의 분산이 크므로 벡터 방향이 크게 변함
- **동시 증가(상,상,상)**가 최악: 등비적 스케일링으로 상대적 차이 변동, 겹침률 45%까지 하락

---

## 실행 명령어

| 명령어 | 설명 |
|--------|------|
| `python main.py` | 기본 파이프라인 (6단계 + 시각화 5개) |
| `python main.py --sensitivity` | 27조합 민감도 분석 (별도 실행) |
| `python main.py --dashboard` | Dash 대시보드 (http://127.0.0.1:8050) |
| `python main.py --sweep` | 파라미터 스윕 비교 (13개 조합) |
| `python main.py --search` | 인터랙티브 영화 검색 |
| `python main.py --examples` | 검색 예시 14개 실행 |
| `python main.py --diagram` | 데이터 필드 다이어그램만 생성 |
| `python crawler.py --resume` | TMDB/KOBIS 크롤링 (이어받기) |

---

## 출력 파일 (results/)

| 파일 | 내용 |
|------|------|
| `summary.html` | **종합 보고서** (14개 섹션: 수식 체계, 연산 흐름도, 민감도 상세 분석 포함) |
| `data_field_diagram.html` | 크롤링 24개 필드 -> 임베딩 54D 매핑 Sankey 다이어그램 |
| `embedding_3d.html` | 3D 임베딩 산점도 (클러스터 색상, 추천 연결선, 신작 마름모) |
| `embedding_2d.html` | 2D PCA 산점도 (클러스터 라벨: 상위 장르 + 영화 수 표시) |
| `similarity_heatmap.html` | 테스트 5편 x 학습 영화 코사인 유사도 히트맵 |
| `evaluation_report.html` | 평가 4개 차트 (부적합률, 영화별 메트릭, 적합/부적합, 전체 요약) |
| `sensitivity_analysis.html` | 27조합 분석 (4개 차트 + 색상 코딩 상세 비교표 + 클러스터별 순위 비교) |

---

## 프로젝트 구조

```
Recommendation_AI/
|-- main.py                 # CLI 엔트리포인트
|-- config.py               # 중앙 파라미터 관리
|-- crawler.py              # TMDB/KOBIS 크롤링 (1980~2026.03)
|-- data_loader.py          # Stage 1: 데이터 로드/검증
|-- embedding.py            # Stage 2: 54D 벡터 생성 + StandardScaler
|-- clustering.py           # Stage 3: KMeans/DBSCAN 군집화
|-- reduction.py            # Stage 4: PCA/t-SNE 차원 축소
|-- recommender.py          # Stage 5: 코사인 유사도 Top-K 추천
|-- evaluator.py            # Stage 6: 정량/정성 평가
|-- visualizer.py           # Plotly 시각화 7종
|-- sensitivity.py          # 27조합 민감도 분석
|-- search.py               # 자유 텍스트 검색 엔진
|-- dashboard.py            # Dash 대시보드 (5탭)
|-- requirements.txt        # 의존성 목록
|-- .env                    # TMDB API 키
|-- data/
|   |-- movies.json         # 학습 데이터 (10,857편)
|   +-- checkpoint.json     # 크롤링 체크포인트
|-- results/
|   |-- summary.html        # 종합 보고서 (14개 섹션)
|   |-- data_field_diagram.html
|   |-- embedding_3d.html
|   |-- embedding_2d.html
|   |-- similarity_heatmap.html
|   |-- evaluation_report.html
|   +-- sensitivity_analysis.html
|-- CLAUDE.md               # AI 어시스턴트 컨텍스트
+-- README.md               # 이 파일
```

---

## 설정 파라미터 (config.py)

| 구분 | 파라미터 | 기본값 | 설명 |
|------|---------|--------|------|
| 임베딩 | WEIGHT_GENRE | 1.0 | 장르 벡터 가중치 |
| | WEIGHT_KEYWORD | 1.0 | 키워드 벡터 가중치 |
| | WEIGHT_NUMERIC | 1.0 | 수치 벡터 가중치 |
| 군집화 | CLUSTER_METHOD | "kmeans" | kmeans 또는 dbscan |
| | KMEANS_N_CLUSTERS | 8 | KMeans 클러스터 수 |
| 차원 축소 | PCA_COMPONENTS | 3 | PCA 출력 차원 |
| | USE_TSNE | False | t-SNE 사용 여부 |
| 추천 | TOP_K | 5 | 추천 영화 수 |
| 평가 | THRESHOLD_AVG_SIMILARITY | 0.6 | 유사도 최소 임계값 |
| | THRESHOLD_GENRE_PRECISION | 0.4 | 장르 정밀도 최소 임계값 |
| | THRESHOLD_3D_DISTANCE | 4.0 | 공간 거리 최대 임계값 |
| 민감도 | SENSITIVITY_LEVELS | 하0.5 / 중1.0 / 상1.5 | 가중치 3단계 레벨 |
| | SENSITIVITY_TOP_K | 20 | 비교용 Top-K 크기 |

---

## 참고사항: STATIC 논문 적용 검토

### STATIC 논문 요약

[Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647) (arXiv:2602.22647, 2026)는 Google에서 발표한 논문으로, **LLM 기반 생성형 검색(Generative Retrieval)**에서 출력을 특정 아이템 집합으로 제한하는 **Constrained Decoding**을 효율화하는 기술이다.

STATIC(**S**parse **T**ransition Matrix-**A**ccelerated **T**rie **I**ndex for **C**onstrained Decoding)은 Trie(접두사 트리)를 CSR(Compressed Sparse Row) 희소 행렬로 평탄화하여, GPU/TPU에서 포인터 추적(pointer-chasing) 방식의 트리 탐색을 **벡터화된 희소 행렬 연산**으로 변환한다.

| 항목 | STATIC |
|------|--------|
| 핵심 기술 | Trie를 CSR 희소 행렬로 평탄화 -> GPU/TPU 벡터화 연산 |
| 적용 대상 | LLM이 아이템 ID를 토큰 단위로 생성할 때, 유효한 아이템만 출력하도록 제약 |
| 성과 | CPU 대비 948배, GPU 이진 탐색 대비 47~1,033배 속도 향상 |
| 오버헤드 | 스텝당 0.033ms, 전체 추론 시간의 0.25% |
| 배포 환경 | YouTube 등 수십억 사용자 규모 산업용 추천 시스템 |

### 현재 시스템 vs STATIC 비교

| 비교 항목 | 현재 시스템 (Recommendation_AI) | STATIC |
|----------|-------------------------------|--------|
| 추천 방식 | Content-Based Filtering (코사인 유사도) | LLM 기반 Generative Retrieval |
| 검색 방식 | 벡터 내적 -> Top-K 정렬 | LLM이 아이템 ID를 토큰 단위로 생성 |
| 아이템 표현 | 54D 연속 벡터 | 토큰 시퀀스 (아이템 ID를 문자열로 표현) |
| 필터링 | 없음 (전체 10,857편 대상) | Trie로 유효 아이템만 생성 제약 |
| LLM 사용 | 없음 (순수 수치 연산) | 필수 (LLM이 핵심) |
| 데이터 규모 | 10,857편 | 수억~수십억 아이템 |
| 해석 가능성 | 높음 (수식 기반, 파라미터 투명) | 낮음 (LLM 블랙박스) |
| 재현성 | seed=42 고정, 동일 결과 보장 | 확률적 생성, 매번 다를 수 있음 |

### STATIC 적용 시 장점

| 장점 | 설명 |
|------|------|
| 자연어 이해 | 현재 사전 매칭 방식 대신, LLM이 "어두운 SF 생존 영화"를 깊이 이해하여 추천 |
| 비정형 쿼리 처리 | "인터스텔라 같은데 좀 더 무서운 영화" 같은 복잡한 자연어 요청 가능 |
| 콜드스타트 개선 | 논문에서 콜드스타트 성능 개선 입증 -- 신작 영화(메타데이터 부족)에 유리 |
| 동적 비즈니스 제약 | "2020년 이후 한국 영화만" 같은 실시간 필터링을 Trie 제약으로 효율 구현 |

### STATIC 적용 시 단점

| 단점 | 설명 |
|------|------|
| 과잉 설계 | 10,857편에 LLM은 과도함. 현재 코사인 유사도가 밀리초 단위로 충분히 빠름 |
| LLM 의존성 증가 | GPT/Gemini 등 대형 모델 필요 -> 비용, 인프라, 레이턴시 급증 |
| 해석 불가 | 현재 시스템은 "장르 공유, 키워드 일치" 등 추천 이유를 명확히 설명. LLM은 블랙박스 |
| 재현성 상실 | 현재 시스템은 seed=42로 동일 결과 보장. LLM은 확률적 생성이라 매번 다름 |
| 민감도 분석 불가 | 현재 27조합 파라미터 분석이 가능한 이유는 수식이 명확하기 때문. LLM 내부는 분석 불가 |
| Trie 구축 비용 | 10,857편의 아이템 ID를 토큰화하여 Trie 구축 -> 이 규모에서는 오버헤드만 발생 |

### 적합성 판단

**STATIC은 현재 시스템에 적합하지 않다.** 이유는 다음과 같다:

| 판단 기준 | 현재 시스템 | STATIC 필요 여부 |
|----------|-----------|----------------|
| 데이터 규모 | 10,857편 | 불필요 (수억 이상에서 의미) |
| 속도 병목 | 없음 (밀리초 응답) | 불필요 |
| LLM 존재 | 없음 | **전제 조건 미충족** |
| 해석 가능성 | 높음 (수식 기반) | 적용 시 상실 |
| 파라미터 분석 | 27조합 가능 | 적용 시 불가 |

### 결론

STATIC은 **"이미 LLM을 사용하고 있는 대규모 추천 시스템"**에서 Constrained Decoding의 속도 병목을 해결하는 기술이다. 현재 시스템처럼 **LLM을 사용하지 않는 소규모 수치 연산 기반 시스템**에는 적용 전제 자체가 맞지 않는다.

만약 향후 LLM을 도입하여 자연어 기반 추천으로 확장한다면, 그때 STATIC의 Constrained Decoding이 유용할 수 있다.

### 출처

- [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators (arXiv:2602.22647, 2026)](https://arxiv.org/abs/2602.22647)
- [ResearchGate - Vectorizing the Trie (Google Research)](https://www.researchgate.net/publication/401279962)

---

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
TMDB 데이터는 [TMDB API 이용약관](https://www.themoviedb.org/documentation/api/terms-of-use)을 따릅니다.
