"""
main.py -- 영화 추천 AI 시스템 파이프라인
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
    python main.py --dashboard        대시보드 실행
    python main.py --diagram          데이터 필드 다이어그램만
    python main.py --sweep            파라미터 스윕 비교
    python main.py --search           인터랙티브 검색 모드
    python main.py --examples         검색 예시 실행
"""

import argparse
import os
import sys

import config


def run_pipeline(params: dict = None) -> dict:
    """
    전체 파이프라인을 실행하고 결과를 반환한다.

    Args:
        params: 파라미터 dict (None이면 config.py 기본값)
            - weight_genre, weight_keyword, weight_numeric
            - cluster_method, n_clusters, dbscan_eps, dbscan_min_samples
            - reduction_method, pca_components, tsne_perplexity
            - top_k

    Returns:
        {"embedding", "clusterer", "reducer", "recommender", "evaluator",
         "recommendations", "quant", "qual", "comparison",
         "train_movies", "test_movies", "coords", "clusters"}
    """
    from data_loader import load_movies, load_test_movies
    from embedding import MovieEmbedding
    from clustering import MovieClusterer
    from reduction import DimensionReducer
    from recommender import MovieRecommender
    from evaluator import RecommendationEvaluator

    p = params or {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 1: 데이터 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 1] 데이터 로드")
    print("-" * 72)
    train_movies = load_movies()
    test_movies = load_test_movies()

    if not train_movies:
        print("[오류] 학습 데이터가 없습니다. 먼저 크롤링을 실행하세요: python crawler.py")
        sys.exit(1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 2: 임베딩
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 2] 임베딩 생성")
    print("-" * 72)
    emb = MovieEmbedding(
        weight_genre=p.get("weight_genre"),
        weight_keyword=p.get("weight_keyword"),
        weight_numeric=p.get("weight_numeric"),
    )
    emb.fit(train_movies)
    emb.transform(test_movies)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 3: 군집화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 3] 군집화")
    print("-" * 72)
    train_titles = [m["title"] for m in train_movies]
    X_scaled = emb.get_all_scaled_matrix(train_titles)

    clusterer = MovieClusterer(
        method=p.get("cluster_method"),
        n_clusters=p.get("n_clusters"),
        eps=p.get("dbscan_eps"),
        min_samples=p.get("dbscan_min_samples"),
    )
    cluster_labels = clusterer.fit_predict(X_scaled)

    # 클러스터 정보 출력
    cluster_info = clusterer.get_cluster_info(cluster_labels, train_movies)
    for name, info in list(cluster_info.items())[:5]:
        top_g = ", ".join(f"{g}({c})" for g, c in info["top_genres"][:3])
        print(f"  {name} ({info['count']}편): {top_g}")
    if len(cluster_info) > 5:
        print(f"  ... 외 {len(cluster_info) - 5}개 클러스터")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 4: 차원 축소
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 4] 차원 축소")
    print("-" * 72)

    use_tsne = p.get("use_tsne", config.USE_TSNE)
    n_comp = p.get("pca_components", config.PCA_COMPONENTS)
    method = "tsne" if use_tsne else "pca"

    reducer = DimensionReducer(
        method=method,
        n_components=n_comp,
        tsne_perplexity=p.get("tsne_perplexity"),
    )

    # 학습 데이터 축소
    X_train_reduced = reducer.fit_transform(X_scaled)

    # 좌표 저장
    coords = {}
    for i, title in enumerate(train_titles):
        coords[title] = X_train_reduced[i]

    # 테스트 데이터 축소 (PCA만 가능)
    if method == "pca":
        test_titles = [m["title"] for m in test_movies]
        X_test_scaled = emb.get_all_scaled_matrix(test_titles)
        X_test_reduced = reducer.transform(X_test_scaled)
        for i, title in enumerate(test_titles):
            coords[title] = X_test_reduced[i]

        # PCA 축 해석
        pca_interp = reducer.interpret_axes(emb.get_feature_names(), top_n=5)
        if pca_interp:
            print("\n  PCA 축 해석:")
            for axis_name, features in pca_interp.items():
                print(f"    {axis_name}:")
                for feat_name, weight in features[:3]:
                    bar = "#" * int(abs(weight) * 20)
                    sign = "+" if weight > 0 else "-"
                    print(f"      {sign} {feat_name:30s} {weight:+.3f}  {bar}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 5: 추천
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 5] 추천 실행")
    print("-" * 72)
    rec = MovieRecommender(
        emb, train_movies,
        top_k=p.get("top_k"),
    )

    recommendations = {}
    for movie in test_movies:
        results = rec.print_recommendations(movie)
        recommendations[movie["title"]] = results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 6: 평가
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[Stage 6] 평가")
    print("-" * 72)
    evaluator = RecommendationEvaluator(
        emb, train_movies, test_movies, coords=coords,
    )
    quant, qual, comparison = evaluator.print_report(recommendations)

    return {
        "embedding": emb,
        "clusterer": clusterer,
        "reducer": reducer,
        "recommender": rec,
        "evaluator": evaluator,
        "recommendations": recommendations,
        "quant": quant,
        "qual": qual,
        "comparison": comparison,
        "train_movies": train_movies,
        "test_movies": test_movies,
        "coords": coords,
        "clusters": cluster_labels,
        "cluster_info": cluster_info,
    }


def run_visualizations(result: dict):
    """파이프라인 결과에 대한 시각화 생성"""
    from visualizer import (
        create_3d_scatter, create_2d_scatter,
        create_similarity_heatmap, create_evaluation_charts,
        create_data_field_diagram,
    )

    print("\n시각화 생성")
    print("-" * 72)

    # 데이터 필드 다이어그램
    create_data_field_diagram()

    # 3D 시각화
    coords = result["coords"]
    n_comp = 3
    if coords:
        sample_coord = next(iter(coords.values()))
        n_comp = len(sample_coord)

    # 축 해석 (PCA인 경우)
    axis_labels = None
    reducer = result["reducer"]
    emb = result["embedding"]
    if reducer.method == "pca":
        pca_interp = reducer.interpret_axes(emb.get_feature_names(), top_n=3)
        if pca_interp:
            axis_labels = {}
            for axis_key, features in pca_interp.items():
                top_feats = [f"{name}({w:+.2f})" for name, w in features[:3]]
                axis_labels[axis_key] = " / ".join(top_feats)

    if n_comp == 3:
        create_3d_scatter(
            coords=coords,
            movies=result["train_movies"],
            clusters=result["clusters"],
            recommendations=result["recommendations"],
            test_movies=result["test_movies"],
            axis_labels=axis_labels,
        )

    # 2D 시각화 (PCA 2D 별도 계산)
    from reduction import DimensionReducer
    train_titles = [m["title"] for m in result["train_movies"]]
    X_scaled = emb.get_all_scaled_matrix(train_titles)

    reducer_2d = DimensionReducer(method="pca", n_components=2)
    X_2d = reducer_2d.fit_transform(X_scaled)
    coords_2d = {}
    for i, title in enumerate(train_titles):
        coords_2d[title] = X_2d[i]

    # 테스트 영화 2D
    test_titles = [m["title"] for m in result["test_movies"]]
    X_test_scaled = emb.get_all_scaled_matrix(test_titles)
    X_test_2d = reducer_2d.transform(X_test_scaled)
    for i, title in enumerate(test_titles):
        coords_2d[title] = X_test_2d[i]

    # 2D 축 해석
    axis_labels_2d = None
    pca_interp_2d = reducer_2d.interpret_axes(emb.get_feature_names(), top_n=3)
    if pca_interp_2d:
        axis_labels_2d = {}
        for axis_key, features in pca_interp_2d.items():
            top_feats = [f"{name}({w:+.2f})" for name, w in features[:3]]
            axis_labels_2d[axis_key] = " / ".join(top_feats)

    create_2d_scatter(
        coords=coords_2d,
        movies=result["train_movies"],
        clusters=result["clusters"],
        test_movies=result["test_movies"],
        method_name="PCA",
        axis_labels=axis_labels_2d,
        cluster_info=result.get("cluster_info"),
    )

    # 히트맵
    create_similarity_heatmap(
        emb, result["test_movies"], result["train_movies"],
    )

    # 평가 차트
    create_evaluation_charts(
        result["quant"], result["qual"], result["comparison"],
    )


def run_sweep():
    """파라미터 스윕: 다양한 파라미터 조합으로 성능 비교"""
    from visualizer import create_parameter_comparison

    print("\n" + "=" * 72)
    print("파라미터 스윕 시작")
    print("=" * 72)

    sweep_configs = [
        {"weight_genre": 1.0, "weight_keyword": 1.0, "weight_numeric": 1.0},
        {"weight_genre": 2.0, "weight_keyword": 1.0, "weight_numeric": 1.0},
        {"weight_genre": 1.0, "weight_keyword": 2.0, "weight_numeric": 1.0},
        {"weight_genre": 1.0, "weight_keyword": 1.0, "weight_numeric": 2.0},
        {"weight_genre": 1.5, "weight_keyword": 1.5, "weight_numeric": 0.5},
        {"weight_genre": 0.5, "weight_keyword": 0.5, "weight_numeric": 2.0},
        {"n_clusters": 4},
        {"n_clusters": 8},
        {"n_clusters": 12},
        {"n_clusters": 16},
        {"top_k": 3},
        {"top_k": 5},
        {"top_k": 10},
    ]

    sweep_results = []
    for i, sweep_params in enumerate(sweep_configs):
        label = ", ".join(f"{k}={v}" for k, v in sweep_params.items())
        print(f"\n--- 스윕 [{i+1}/{len(sweep_configs)}] {label} ---")

        result = run_pipeline(sweep_params)
        overall = result["quant"].get("overall", {})

        sweep_results.append({
            "params": sweep_params,
            "metrics": overall,
        })

        print(f"  결과: avg_sim={overall.get('avg_similarity', 0):.4f}, "
              f"genre_prec={overall.get('avg_genre_precision', 0):.4f}")

    # 비교 차트 생성
    create_parameter_comparison(sweep_results)

    print("\n" + "=" * 72)
    print(f"파라미터 스윕 완료: {len(sweep_configs)}개 조합 비교")
    print("=" * 72)


def run_sensitivity():
    """파라미터 민감도 분석: 가중치 변화에 따른 추천 목록 변화 시각화"""
    from sensitivity import SensitivityAnalyzer
    from visualizer import create_sensitivity_charts

    analyzer = SensitivityAnalyzer()
    results = analyzer.run()

    # 시각화 생성
    create_sensitivity_charts(results)

    print(f"\n생성된 파일:")
    print(f"   - {config.RESULTS_DIR}/sensitivity_analysis.html")
    print("브라우저에서 열어 확인하세요.\n")


def main():
    parser = argparse.ArgumentParser(description="영화 추천 AI 시스템")
    parser.add_argument("--dashboard", action="store_true",
                        help="대시보드 실행")
    parser.add_argument("--sensitivity", action="store_true",
                        help="파라미터 민감도 분석 실행")
    parser.add_argument("--diagram", action="store_true",
                        help="데이터 필드 다이어그램만 생성")
    parser.add_argument("--sweep", action="store_true",
                        help="파라미터 스윕 비교 실행")
    parser.add_argument("--search", action="store_true",
                        help="인터랙티브 검색 모드")
    parser.add_argument("--examples", action="store_true",
                        help="검색 예시 실행 (제목 7개 + 자유 텍스트 7개)")
    args = parser.parse_args()

    print("영화 추천 AI 시스템")
    print("=" * 72)

    # GPU 상태 출력
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA 활성화 - {gpu_name}")
        else:
            print("[GPU] CUDA 사용 불가 - CPU 모드")
    except ImportError:
        print("[GPU] PyTorch 미설치 - CPU 모드")

    # 대시보드 모드
    if args.dashboard:
        from dashboard import create_dashboard
        app = create_dashboard()
        print(f"\n대시보드 실행: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
        app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)
        return

    # 다이어그램만
    if args.diagram:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        from visualizer import create_data_field_diagram
        create_data_field_diagram()
        print(f"\n데이터 필드 다이어그램 생성 완료: {config.RESULTS_DIR}/data_field_diagram.html")
        return

    # 파라미터 스윕
    if args.sweep:
        run_sweep()
        return

    # 민감도 분석
    if args.sensitivity:
        run_sensitivity()
        return

    # 검색 모드
    if args.search or args.examples:
        result = run_pipeline()
        from search import MovieSearchEngine, run_examples
        engine = MovieSearchEngine(result["embedding"], result["train_movies"])
        if args.examples:
            run_examples(result["embedding"], result["train_movies"])
        else:
            engine.interactive()
        return

    # 기본 파이프라인
    result = run_pipeline()
    run_visualizations(result)

    # 완료 메시지
    print("\n" + "=" * 72)
    print("전체 파이프라인 완료!")
    print("=" * 72)
    print(f"\n생성된 파일 ({config.RESULTS_DIR}/):")
    print("   - data_field_diagram.html       - 데이터 필드 매핑 다이어그램")
    print("   - embedding_3d.html             - 3D 임베딩 시각화")
    print("   - embedding_2d.html             - 2D 임베딩 시각화")
    print("   - similarity_heatmap.html       - 유사도 히트맵")
    print("   - evaluation_report.html        - 평가 비교 차트")
    print("\n브라우저에서 HTML 파일을 열어 시각화를 확인하세요.")
    print("대시보드 실행:   python main.py --dashboard")
    print("검색 모드:       python main.py --search")
    print("검색 예시:       python main.py --examples")
    print("민감도 분석:     python main.py --sensitivity\n")


if __name__ == "__main__":
    main()
