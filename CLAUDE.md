# CLAUDE.md -- KMDB 영화 추천 시스템

## 프로젝트 개요

코사인 유사도 기반 하이브리드(텍스트+메타데이터) 영화 추천 시스템.
KMDB(한국영화데이터베이스) **19,354편(1980~2026.03)** 학습, 자유 텍스트 검색 Top-20 추천.

## 기술 스택

Python 3.12 | NumPy, scikit-learn | sentence-transformers(paraphrase-multilingual-MiniLM-L12-v2) | Plotly, Dash | PyTorch CUDA | Pillow, tqdm

## 파이프라인 (6단계)

| 단계 | 파일 | 역할 |
|------|------|------|
| 1 | data_loader.py | 554 JSON 파싱, 장르 정규화(66->30), 파생 수치, pkl 캐시 |
| 2 | embedding.py | 499D 하이브리드 벡터 생성 (그룹별 L2 정규화 + 가중치) |
| 3 | clustering.py | KMeans(k=12) GPU 군집화 (seed=42) |
| 4 | reduction.py | PCA/t-SNE 차원 축소 (2D/3D 시각화용) |
| 5 | recommender.py | 코사인 유사도 Top-20 추천 + 그룹별 기여도 분해 |
| 6 | evaluator.py | 정량/정성 평가 + 텍스트 일관성 + 적합/부적합 판정 |

### 보조 모듈

| 파일 | 역할 |
|------|------|
| config.py | 중앙 파라미터 관리 (장르 매핑, 키워드, 가중치, 임계값) |
| visualizer.py | 시각화 6종 (Sankey, 2D/3D 산점도, 히트맵, 평가, 기여도) |
| sensitivity.py | 27조합 민감도 분석 (장르/키워드/텍스트 x 0.5/1.0/1.5) |
| search.py | 자유 텍스트 검색 엔진 (장르/키워드 별칭 파싱 + 텍스트 임베딩) |
| dashboard.py | Dash 대시보드 5탭 (검색/클러스터/파라미터/평가/민감도) |
| main.py | CLI 통합 엔트리포인트 |

### 데이터/출력 디렉토리

| 경로 | 내용 |
|------|------|
| Data_new/movies/ | KMDB 원본 (554 JSON + 19,791 포스터 JPG) |
| data/ | 캐시 (kmdb_processed.pkl, text_embeddings.npy) |
| results/ | HTML 시각화 6종 |

## 하이브리드 임베딩 (499D)

```
V = [L2norm(genre)*Wg | L2norm(keyword)*Wk | L2norm(numeric)*Wn | L2norm(text)*Wt]
     30D one-hot        80D binary           5D derived            384D sentence-transformers
```

| 그룹 | 차원 | 기본 가중치 | 설명 |
|------|------|------------|------|
| 장르 | 30D | 1.0 | KMDB 66종 -> 30개 표준 카테고리 원-핫 |
| 키워드 | 80D | 1.0 | 빈도 상위 80개 바이너리 |
| 수치 | 5D | 0.5 | runtime_norm, year_norm, keyword_richness, cast_size_norm, genre_count_norm |
| 텍스트 | 384D | 1.5 | paraphrase-multilingual-MiniLM-L12-v2 줄거리 임베딩 |

**정규화**: 그룹별 L2 정규화 후 가중치 곱 -> 연결(concatenate). 차원 수 차이(30D vs 384D) 편향 방지.

## 핵심 수식

| 연산 | 수식 | 용도 |
|------|------|------|
| L2 정규화 | g_norm = g / \|\|g\|\|_2 | 그룹별 차원 편향 방지 |
| 벡터 생성 | V = [g*Wg \| k*Wk \| n*Wn \| t*Wt] (499D) | 하이브리드 벡터 |
| 코사인 유사도 | cos(A,B) = A.B / (\|\|A\|\| x \|\|B\|\|) | 추천 순위 |
| KMeans | centroid = mean(X[label==c]) | 군집화 |
| PCA | X_reduced = (X-mu) @ V^T[:n] | 시각화 |
| Spearman rho | 1 - 6*sum(d^2) / (n*(n^2-1)) | 민감도 순위상관 |

## 실행 명령어

```bash
pip install -r requirements.txt
python main.py --build-cache         # 데이터 캐시 생성 (첫 실행)
python main.py                       # 전체 파이프라인 (6단계 + 시각화)
python main.py --sensitivity         # 27조합 민감도 분석
python main.py --dashboard           # 대시보드 (http://127.0.0.1:8050)
python main.py --sweep               # 파라미터 스윕 비교
python main.py --search              # 인터랙티브 검색
python main.py --examples            # 검색 예시
```

## 주의사항

- ALL_GENRES/ALL_KEYWORDS 변경 시 벡터 차원 변경됨 -> 캐시 삭제 필요
- GENRE_MAP에 없는 장르는 무시됨
- runtime=0 또는 None -> runtime_norm=0.5 (결측치 처리)
- text_embeddings.npy는 영화 수 변경 시 자동 재생성
- kmdb_processed.pkl 삭제 시 554 JSON 재파싱
- sentence-transformers 미설치 시 텍스트 임베딩이 0벡터로 대체됨 (추천 품질 저하)
- tensorflow가 설치된 환경에서는 protobuf 충돌로 sentence-transformers import 실패 가능 -> tensorflow 제거 필요

## 코드 수정 가이드

- **파라미터 변경**: config.py 수정 또는 대시보드 실시간 조정
- **장르 매핑 추가**: config.py GENRE_MAP에 {원본: 표준} 추가
- **키워드 목록 변경**: config.py ALL_KEYWORDS 수정 (중복 금지, 차원 변경 주의)
- **테스트 영화 변경**: config.py TEST_MOVIE_TITLES 지정 또는 자동 선정
- **가중치 변경**: config.py WEIGHT_GENRE/KEYWORD/NUMERIC/TEXT 또는 대시보드 슬라이더
- **텍스트 모델 변경**: config.py TEXT_MODEL_NAME (차원 변경 시 TEXT_EMBED_DIM도 수정)
- **평가 임계값 변경**: config.py THRESHOLD_GENRE_PRECISION/AVG_SIMILARITY/3D_DISTANCE
