# CLAUDE.md -- AI 영화 추천 시스템 프로젝트 컨텍스트

## 프로젝트 개요

코사인 유사도 기반 콘텐츠 필터링(Content-Based Filtering) 영화 추천 시스템.
TMDB/KOBIS 크롤링 데이터 **10,857편(1980~2026.03)**을 학습하고,
2026년 개봉 예정 **5편** + 자유 텍스트 검색 쿼리에 대해 Top-K 추천.

## 기술 스택

Python 3.12 | NumPy, scikit-learn | Plotly, Dash | PyTorch CUDA(선택) | Requests, tqdm

## 아키텍처

```
crawler.py          TMDB/KOBIS 크롤링 → data/movies.json (10,857편)
config.py           중앙 파라미터 관리
data_loader.py      Stage 1: 데이터 로드/검증/결측치 처리
embedding.py        Stage 2: 54D 가중 벡터 (17장르 + 30키워드 + 7수치) + StandardScaler
clustering.py       Stage 3: KMeans(k=8) 군집화 (GPU seed=42 결정성 보장)
reduction.py        Stage 4: PCA/t-SNE 차원 축소 (2D/3D)
recommender.py      Stage 5: 코사인 유사도 Top-K 추천
evaluator.py        Stage 6: 정량/정성 평가 + 적합/부적합 판정
visualizer.py       시각화 7종 (Sankey, 2D/3D, 히트맵, 평가, 민감도)
sensitivity.py      27조합 민감도 분석 (상1.5/중1.0/하0.5)³
search.py           자유 텍스트 → 54D 벡터 변환 → 추천
dashboard.py        Dash 대시보드 (5탭: 시각화/추천/평가/스윕/민감도)
main.py             CLI 통합 엔트리포인트
results/            HTML 출력 (summary.html 포함 7개)
```

## 핵심 수식

| 연산 | 수식 | 용도 |
|------|------|------|
| 벡터 생성 | V = [genre×Wg \| kw×Wk \| num×Wn] (54D) | 영화 → 벡터 |
| 표준화 | X_scaled = (X - μ) / σ | 스케일 통일 |
| 코사인 유사도 | cos(A,B) = A·B / (\|\|A\|\|×\|\|B\|\|) | 추천 순위 |
| KMeans | centroid = mean(X[label==c]) | 군집화 |
| PCA | X_reduced = (X-μ) @ Vᵀ[:n] | 시각화 |
| Spearman ρ | 1 - 6Σd²/(n(n²-1)) | 민감도 순위상관 |

## 실행 명령어

```bash
pip install -r requirements.txt
python main.py                    # 기본 파이프라인 (6단계 + 시각화 5개)
python main.py --sensitivity      # 27조합 민감도 분석
python main.py --dashboard        # 대시보드 (http://127.0.0.1:8050)
python main.py --sweep            # 파라미터 스윕 비교
python main.py --search           # 인터랙티브 검색
python main.py --examples         # 검색 예시 실행
python crawler.py --resume        # 데이터 크롤링 (이어받기)
```

## 출력 파일 (results/)

| 파일 | 내용 |
|------|------|
| summary.html | 종합 보고서 (14개 섹션, 수식/흐름도 포함) |
| data_field_diagram.html | 24개 필드 → 54D 매핑 Sankey |
| embedding_3d.html | 3D 산점도 (클러스터 + 추천 연결선) |
| embedding_2d.html | 2D 산점도 (클러스터 라벨 + PCA 축 해석) |
| similarity_heatmap.html | 테스트 5편 × 학습 영화 유사도 |
| evaluation_report.html | 평가 4개 차트 (막대 위 수치 표시) |
| sensitivity_analysis.html | 27조합 차트 + 상세 비교표 + 순위 비교 |

## 데이터 주의사항

- `config.py`의 ALL_GENRES/ALL_KEYWORDS 변경 시 벡터 차원이 변경됨
- 수치 특징은 0.0~1.0 범위 유지 필수
- runtime=0은 결측치(None)로 처리됨
- budget_usd=0.0은 budget_scale=0.10으로 처리 (결측)
- critic_score/audience_score None → 0.5 대체

## 코드 수정 가이드

- **파라미터 변경**: config.py 또는 대시보드 실시간 조정
- **테스트 영화 추가**: config.py TEST_MOVIES에 동일 dict 구조로 추가
- **민감도 레벨 변경**: config.py SENSITIVITY_LEVELS (상/중/하 값)
- **크롤링 범위 변경**: crawler.py START_DATE/END_DATE + YEAR_CHUNKS
