"""
main.py -- KMDB 영화 추천 시스템 파이프라인
====================================================================
6단계 파이프라인을 순서대로 실행한다:
  Stage 1: 데이터 로드        (data_loader.py)
  Stage 2: 임베딩 생성        (embedding.py)
  Stage 3: 군집화             (clustering.py)
  Stage 4: 차원 축소          (reduction.py)
  Stage 5: 추천 실행          (recommender.py)
  Stage 6: 평가 + 시각화      (evaluator.py + visualizer.py)

실행:
    python main.py                    기본 파이프라인
    python main.py --build-cache      데이터 캐시 생성만
    python main.py --dashboard        대시보드 실행
    python main.py --sensitivity      민감도 분석
    python main.py --search           인터랙티브 검색
    python main.py --examples         검색 예시 실행
"""

import argparse
import os
import sys

import config


def run_pipeline(params=None):
    """
    전체 파이프라인을 실행하고 결과를 반환한다.

    Parameters
    ----------
    params : dict, optional
        - weight_genre, weight_keyword, weight_numeric, weight_text
        - cluster_method, n_clusters
        - reduction_method, pca_components
        - top_k

    Returns
    -------
    dict
    """
    from data_loader import load_movies, select_test_movies
    from embedding import HybridMovieEmbedding
    from clustering import MovieClusterer
    from reduction import DimensionReducer
    from recommender import MovieRecommender
    from evaluator import RecommendationEvaluator
    import numpy as np

    p = params or {}

    # Stage 1: 데이터 로드
    print("\n[Stage 1] 데이터 로드")
    print("-" * 72)
    movies = load_movies()

    if not movies:
        print("[오류] 데이터가 없습니다.")
        sys.exit(1)

    test_titles = p.get("test_titles") or config.TEST_MOVIE_TITLES
    train_movies, test_movies = select_test_movies(movies, titles=test_titles or None)

    print(f"  학습: {len(train_movies):,}편 / 테스트: {len(test_movies)}편")
    for tm in test_movies:
        print(f"    - {tm['title']} ({tm['year']}) {tm['genres']}")

    # Stage 2: 임베딩
    print("\n[Stage 2] 임베딩 생성")
    print("-" * 72)
    emb = HybridMovieEmbedding(
        weight_genre=p.get("weight_genre"),
        weight_keyword=p.get("weight_keyword"),
        weight_numeric=p.get("weight_numeric"),
        weight_text=p.get("weight_text"),
    )
    emb.fit(train_movies)
    emb.transform(test_movies)

    # Stage 3: 군집화
    print("\n[Stage 3] 군집화")
    print("-" * 72)
    clusterer = MovieClusterer(
        method=p.get("cluster_method"),
        n_clusters=p.get("n_clusters"),
    )
    train_ids = [m["id"] for m in train_movies]
    X_train = emb.get_all_scaled_matrix(train_ids)
    clusters = clusterer.fit_predict(X_train)
    cluster_info = clusterer.get_cluster_info(clusters, train_movies)

    # Stage 4: 차원 축소
    print("\n[Stage 4] 차원 축소")
    print("-" * 72)
    reducer = DimensionReducer(
        method=p.get("reduction_method"),
        n_components=p.get("pca_components"),
    )
    train_coords = reducer.fit_transform(X_train)

    # 테스트 영화 좌표
    test_ids = [m["id"] for m in test_movies]
    if test_movies and reducer.method == "pca":
        X_test = emb.get_all_scaled_matrix(test_ids)
        test_coords = reducer.transform(X_test)
    else:
        test_coords = np.zeros((len(test_movies), reducer.n_components))

    # 좌표 매핑
    coords = {}
    for i, mid in enumerate(train_ids):
        coords[mid] = train_coords[i]
    for i, mid in enumerate(test_ids):
        coords[mid] = test_coords[i]

    # Stage 5: 추천
    print("\n[Stage 5] 추천")
    print("-" * 72)
    recommender = MovieRecommender(
        emb, train_movies, top_k=p.get("top_k")
    )
    recommendations = recommender.recommend_batch(test_movies)

    # 추천 결과 콘솔 출력
    for tm in test_movies:
        recommender.print_recommendations(tm, top_k=min(5, p.get("top_k", config.TOP_K)))

    # Stage 6: 평가
    print("\n[Stage 6] 평가")
    print("-" * 72)
    evaluator = RecommendationEvaluator(
        emb, train_movies, test_movies, coords
    )
    quant, qual, comparison = evaluator.print_report(recommendations)

    return {
        "embedding": emb,
        "clusterer": clusterer,
        "reducer": reducer,
        "recommender": recommender,
        "evaluator": evaluator,
        "recommendations": recommendations,
        "quant": quant,
        "qual": qual,
        "comparison": comparison,
        "train_movies": train_movies,
        "test_movies": test_movies,
        "coords": coords,
        "clusters": clusters,
        "cluster_info": cluster_info,
    }


def main():
    parser = argparse.ArgumentParser(description="KMDB 영화 추천 시스템")
    parser.add_argument("--build-cache", action="store_true",
                        help="데이터 캐시 생성만 수행")
    parser.add_argument("--dashboard", action="store_true",
                        help="대시보드 실행 (http://127.0.0.1:8050)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="27조합 민감도 분석")
    parser.add_argument("--search", action="store_true",
                        help="인터랙티브 검색 모드")
    parser.add_argument("--examples", action="store_true",
                        help="검색 예시 실행")
    parser.add_argument("--sweep", action="store_true",
                        help="파라미터 스윕 비교")
    args = parser.parse_args()

    # 캐시 생성만
    if args.build_cache:
        from data_loader import load_movies
        movies = load_movies(use_cache=False)
        print(f"\n캐시 생성 완료: {len(movies):,}편")
        return

    # 대시보드 모드
    if args.dashboard:
        from dashboard import create_app
        app = create_app()
        print(f"\n대시보드 시작: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
        app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)
        return

    # 기본 파이프라인 실행
    result = run_pipeline()

    # 시각화 생성
    print("\n[시각화] HTML 생성 중...")
    print("-" * 72)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    from visualizer import MovieVisualizer
    viz = MovieVisualizer(result)
    viz.generate_all()

    # 민감도 분석
    if args.sensitivity:
        print("\n[민감도 분석] 27조합 분석 중...")
        print("-" * 72)
        from sensitivity import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer()
        all_results, movie_analysis = analyzer.analyze_movies(
            result["embedding"], result["train_movies"], result["test_movies"]
        )
        _, text_analysis = analyzer.analyze_text_queries(
            result["embedding"], result["train_movies"]
        )
        analyzer.print_analysis(movie_analysis)
        if text_analysis:
            analyzer.print_analysis(text_analysis)

        # 민감도 분석 HTML 생성
        print("\n[민감도 시각화] sensitivity_analysis.html 생성 중...")
        viz.generate_sensitivity_html(
            all_results=all_results,
            analysis=movie_analysis,
            train_movies=result["train_movies"],
            test_movies=result["test_movies"],
            clusters=result["clusters"],
        )

    # 인터랙티브 검색
    if args.search:
        from search import run_interactive_search
        run_interactive_search(
            result["embedding"], result["train_movies"], result["test_movies"]
        )

    # 검색 예시
    if args.examples:
        from search import MovieSearchEngine
        engine = MovieSearchEngine(
            result["embedding"], result["train_movies"], result["test_movies"]
        )
        example_queries = [
            "어두운 분위기의 SF 우주 생존 영화",
            "밝은 로맨틱 코미디 사랑 이야기",
            "빠른 전개의 첩보 액션 스파이 스릴러",
            "역사적 전쟁 드라마",
            "판타지 모험 마법",
        ]
        for q in example_queries:
            print(f"\n{'='*60}")
            print(f"검색: {q}")
            print(f"{'='*60}")
            results, stype, parsed = engine.search(q, top_k=10)
            for r in results:
                print(f"  #{r['rank']:2d}  {r['title']} ({r['year']})  "
                      f"유사도: {r['similarity']:.4f}")

    # 파라미터 스윕
    if args.sweep:
        print("\n[파라미터 스윕] 비교 중...")
        print("-" * 72)
        sweep_configs = [
            {"weight_genre": 1.5, "weight_keyword": 1.0, "weight_text": 1.0},
            {"weight_genre": 1.0, "weight_keyword": 1.5, "weight_text": 1.0},
            {"weight_genre": 1.0, "weight_keyword": 1.0, "weight_text": 1.5},
            {"weight_genre": 0.5, "weight_keyword": 0.5, "weight_text": 2.0},
            {"weight_genre": 2.0, "weight_keyword": 0.5, "weight_text": 0.5},
        ]
        for sc in sweep_configs:
            print(f"\n  설정: {sc}")
            r = run_pipeline(sc)
            overall = r["quant"].get("overall", {})
            print(f"    평균 유사도: {overall.get('avg_similarity', 'N/A')}")
            print(f"    장르 정밀도: {overall.get('avg_genre_precision', 'N/A')}")

    print("\n완료.")


if __name__ == "__main__":
    main()
