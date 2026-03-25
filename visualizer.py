"""
visualizer.py -- 시각화 모듈
====================================================================
2D/3D 임베딩 산점도, 유사도 히트맵, 데이터 필드 다이어그램,
평가 비교 차트, 파라미터별 성능 비교 차트를 생성한다.

모든 출력은 results/ 폴더에 Plotly HTML로 저장된다.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from data_loader import get_data_field_info


def _ensure_results_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 1. 데이터 필드 다이어그램
# ═══════════════════════════════════════════════════════════════════

def create_data_field_diagram(save_path: str = None):
    """
    크롤링 24필드 -> 임베딩 54D 매핑을 Sankey 다이어그램으로 시각화.
    """
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "data_field_diagram.html")

    fields = get_data_field_info()

    # Sankey 노드 구성
    source_labels = list(fields.keys())
    category_labels = ["임베딩 사용 (54D)", "메타데이터", "식별자", "참고용"]
    target_labels = [
        f"장르 원-핫 ({len(config.ALL_GENRES)}D)",
        f"키워드 바이너리 ({len(config.ALL_KEYWORDS)}D)",
        f"수치 특징 ({len(config.NUMERIC_FEATURES)}D)",
    ]

    all_labels = source_labels + category_labels + target_labels
    n_src = len(source_labels)
    n_cat = len(category_labels)

    cat_idx_map = {
        "embedding": n_src + 0,
        "metadata": n_src + 1,
        "identifier": n_src + 2,
        "reference": n_src + 3,
    }
    target_idx_map = {
        "one-hot": n_src + n_cat + 0,
        "binary": n_src + n_cat + 1,
        "numeric": n_src + n_cat + 2,
    }

    cat_colors = {
        "embedding": "rgba(46, 204, 113, 0.6)",
        "metadata": "rgba(52, 152, 219, 0.4)",
        "identifier": "rgba(241, 196, 15, 0.4)",
        "reference": "rgba(149, 165, 166, 0.3)",
    }

    sources = []
    targets = []
    values = []
    link_colors = []

    # 링크: 필드 -> 카테고리
    for i, (field_name, info) in enumerate(fields.items()):
        cat = info["category"]
        cat_idx = cat_idx_map[cat]
        dim = max(info["embedding_dim"], 1)
        sources.append(i)
        targets.append(cat_idx)
        values.append(dim)
        link_colors.append(cat_colors.get(cat, "rgba(200,200,200,0.3)"))

    # 링크: 임베딩 카테고리 -> 인코딩 타입
    for field_name, info in fields.items():
        if info["category"] != "embedding":
            continue
        enc = info["encoding"]
        if enc in target_idx_map:
            sources.append(cat_idx_map["embedding"])
            targets.append(target_idx_map[enc])
            values.append(info["embedding_dim"])
            link_colors.append("rgba(46, 204, 113, 0.4)")

    # 노드 색상
    node_colors = []
    for field_name in source_labels:
        cat = fields[field_name]["category"]
        c = cat_colors.get(cat, "rgba(200,200,200,0.5)")
        node_colors.append(c.replace("0.3", "0.8").replace("0.4", "0.8").replace("0.6", "0.8"))
    node_colors.extend([
        "rgba(46, 204, 113, 0.9)",   # 임베딩
        "rgba(52, 152, 219, 0.9)",   # 메타데이터
        "rgba(241, 196, 15, 0.9)",   # 식별자
        "rgba(149, 165, 166, 0.9)",  # 참고용
    ])
    node_colors.extend([
        "rgba(231, 76, 60, 0.9)",    # 장르
        "rgba(155, 89, 182, 0.9)",   # 키워드
        "rgba(243, 156, 18, 0.9)",   # 수치
    ])

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels, color=node_colors,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
        ),
    ))

    total_dim = len(config.ALL_GENRES) + len(config.ALL_KEYWORDS) + len(config.NUMERIC_FEATURES)
    fig.update_layout(
        title=dict(
            text=f"크롤링 데이터 필드 -> 임베딩 벡터 ({total_dim}D) 매핑 다이어그램",
            font=dict(size=18),
        ),
        font=dict(size=12),
        width=1400, height=800,
    )

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  데이터 필드 다이어그램 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 2. 3D 임베딩 시각화
# ═══════════════════════════════════════════════════════════════════

def create_3d_scatter(
    coords: dict,
    movies: list[dict],
    clusters: np.ndarray = None,
    recommendations: dict = None,
    test_movies: list[dict] = None,
    axis_labels: dict = None,
    save_path: str = None,
):
    """
    3D 임베딩 산점도.

    Args:
        coords: {title: np.ndarray(3,)} 3D 좌표
        movies: 학습 영화 리스트
        clusters: (N,) 클러스터 라벨 (None이면 장르 기준 색상)
        recommendations: {query_title: [rec_dicts]} 추천 결과
        test_movies: 테스트 영화 리스트
        axis_labels: {"PC1 (X축)": "feature1/feature2/...", ...} 축 해석
        save_path: 저장 경로
    """
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "embedding_3d.html")

    fig = go.Figure()

    # 데이터 규모에 따른 마커 크기
    n = len(movies)
    if n > 5000:
        marker_size = 3
    elif n > 1000:
        marker_size = 4
    elif n > 100:
        marker_size = 5
    else:
        marker_size = 7

    # 축 범위 계산
    all_coords_list = [coords.get(m["title"]) for m in movies]
    if test_movies:
        all_coords_list += [coords.get(m["title"]) for m in test_movies]
    all_coords_list = [c for c in all_coords_list if c is not None]
    scale = float(np.abs(np.array(all_coords_list)).max()) if all_coords_list else 2.0

    _CLUSTER_COLORS = [
        "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
        "#8E44AD", "#E67E22", "#1ABC9C", "#34495E",
        "#00BCD4", "#795548", "#607D8B", "#FF5722",
        "#4CAF50", "#9C27B0", "#FF9800", "#009688",
        "#673AB7", "#CDDC39", "#03A9F4", "#FF5252",
    ]

    if clusters is not None:
        # 클러스터 기준 색상
        cluster_groups = {}
        for i, movie in enumerate(movies):
            label = int(clusters[i])
            if label not in cluster_groups:
                cluster_groups[label] = {"x": [], "y": [], "z": [], "text": []}
            coord = coords.get(movie["title"])
            if coord is not None:
                cluster_groups[label]["x"].append(coord[0])
                cluster_groups[label]["y"].append(coord[1])
                cluster_groups[label]["z"].append(coord[2])
                cluster_groups[label]["text"].append(
                    f"<b>{movie['title']}</b> ({movie['year']})<br>"
                    f"장르: {', '.join(movie['genres'])}<br>"
                    f"클러스터: {label}"
                )

        for label, data in sorted(cluster_groups.items()):
            name = f"Cluster {label}" if label >= 0 else "Noise"
            color = _CLUSTER_COLORS[label % len(_CLUSTER_COLORS)] if label >= 0 else "#999"
            fig.add_trace(go.Scatter3d(
                x=data["x"], y=data["y"], z=data["z"],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7,
                            line=dict(width=0)),
                text=data["text"], hoverinfo="text",
                name=name, legendgroup=name,
            ))
    else:
        # 장르 기준 색상
        genre_groups = {}
        for movie in movies:
            genre = movie["genres"][0] if movie["genres"] else "Drama"
            if genre not in genre_groups:
                genre_groups[genre] = {"x": [], "y": [], "z": [], "text": []}
            coord = coords.get(movie["title"])
            if coord is not None:
                genre_groups[genre]["x"].append(coord[0])
                genre_groups[genre]["y"].append(coord[1])
                genre_groups[genre]["z"].append(coord[2])
                genre_groups[genre]["text"].append(
                    f"<b>{movie['title']}</b> ({movie['year']})<br>"
                    f"장르: {', '.join(movie['genres'])}"
                )

        for genre, data in genre_groups.items():
            color = config.GENRE_COLORS.get(genre, "#CCC")
            fig.add_trace(go.Scatter3d(
                x=data["x"], y=data["y"], z=data["z"],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7,
                            line=dict(width=0)),
                text=data["text"], hoverinfo="text",
                name=genre, legendgroup=genre,
            ))

    # 테스트 영화: 다이아몬드 마커
    if test_movies:
        new_size = max(marker_size * 3, 12)
        for movie in test_movies:
            coord = coords.get(movie["title"])
            if coord is not None:
                hover = (
                    f"<b>[NEW] {movie['title']}</b> ({movie['year']})<br>"
                    f"장르: {', '.join(movie['genres'])}<br>"
                    f"키워드: {', '.join(movie['keywords'])}"
                )
                fig.add_trace(go.Scatter3d(
                    x=[coord[0]], y=[coord[1]], z=[coord[2]],
                    mode="markers+text",
                    marker=dict(
                        size=new_size, color="#FFD700", symbol="diamond",
                        opacity=1.0, line=dict(width=2, color="black"),
                    ),
                    text=[movie["title"]],
                    textposition="top center",
                    textfont=dict(size=10, color="black"),
                    hovertext=[hover], hoverinfo="text",
                    name=f"[NEW] {movie['title']}", showlegend=True,
                ))

    # 추천 연결선
    if recommendations:
        for query_title, recs in recommendations.items():
            q_coord = coords.get(query_title)
            if q_coord is None:
                continue
            for rec in recs[:3]:
                r_coord = coords.get(rec["title"])
                if r_coord is None:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=[q_coord[0], r_coord[0]],
                    y=[q_coord[1], r_coord[1]],
                    z=[q_coord[2], r_coord[2]],
                    mode="lines",
                    line=dict(color="black", width=3, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                ))

    # 좌표축 화살표
    arrow_len = scale * 1.2
    cone_size = arrow_len * 0.08
    for label, ux, uy, uz in [("X", 1, 0, 0), ("Y", 0, 1, 0), ("Z", 0, 0, 1)]:
        ax, ay, az = ux * arrow_len, uy * arrow_len, uz * arrow_len
        fig.add_trace(go.Scatter3d(
            x=[-ax, ax], y=[-ay, ay], z=[-az, az],
            mode="lines", line=dict(color="black", width=3),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[ax], y=[ay], z=[az], u=[ux], v=[uy], w=[uz],
            colorscale=[[0, "black"], [1, "black"]], showscale=False,
            sizemode="absolute", sizeref=cone_size, anchor="tail",
            hoverinfo="skip", showlegend=False,
        ))

    # 축 레이블
    x_title, y_title, z_title = "PC1", "PC2", "PC3"
    if axis_labels:
        keys = list(axis_labels.keys())
        if len(keys) >= 1:
            x_title = f"X: {axis_labels[keys[0]]}"
        if len(keys) >= 2:
            y_title = f"Y: {axis_labels[keys[1]]}"
        if len(keys) >= 3:
            z_title = f"Z: {axis_labels[keys[2]]}"

    fig.update_layout(
        title=dict(text="영화 임베딩 3D 시각화", font=dict(size=20)),
        scene=dict(
            xaxis=dict(title=dict(text=x_title, font=dict(size=11)),
                       backgroundcolor="rgb(240,240,240)", gridcolor="white"),
            yaxis=dict(title=dict(text=y_title, font=dict(size=11)),
                       backgroundcolor="rgb(230,230,250)", gridcolor="white"),
            zaxis=dict(title=dict(text=z_title, font=dict(size=11)),
                       backgroundcolor="rgb(245,245,220)", gridcolor="white"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        legend=dict(title="범례", font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=60, b=0),
        width=config.FIGURE_WIDTH, height=config.FIGURE_HEIGHT,
    )

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  3D 시각화 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 3. 2D 임베딩 시각화
# ═══════════════════════════════════════════════════════════════════

def create_2d_scatter(
    coords: dict,
    movies: list[dict],
    clusters: np.ndarray = None,
    test_movies: list[dict] = None,
    method_name: str = "PCA",
    axis_labels: dict = None,
    save_path: str = None,
    cluster_info: dict = None,
):
    """2D 임베딩 산점도."""
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "embedding_2d.html")

    fig = go.Figure()

    n = len(movies)
    marker_size = 4 if n > 1000 else (6 if n > 100 else 8)

    # 8개 클러스터가 명확히 구분되는 색상 팔레트
    _COLORS = [
        "#E74C3C",  # 0: 빨강
        "#3498DB",  # 1: 파랑
        "#2ECC71",  # 2: 초록
        "#F1C40F",  # 3: 노랑
        "#8E44AD",  # 4: 보라
        "#E67E22",  # 5: 주황
        "#1ABC9C",  # 6: 청록
        "#34495E",  # 7: 진회색 (기존 #E91E63 핑크→빨강과 혼동)
        "#00BCD4",  # 8: 하늘
        "#795548",  # 9: 갈색
        "#607D8B",  # 10: 회색
        "#FF5722",  # 11: 주홍
    ]

    if clusters is not None:
        cluster_groups = {}
        for i, movie in enumerate(movies):
            label = int(clusters[i])
            if label not in cluster_groups:
                cluster_groups[label] = {"x": [], "y": [], "text": []}
            coord = coords.get(movie["title"])
            if coord is not None:
                cluster_groups[label]["x"].append(coord[0])
                cluster_groups[label]["y"].append(coord[1])
                cluster_groups[label]["text"].append(
                    f"<b>{movie['title']}</b> ({movie['year']})<br>"
                    f"장르: {', '.join(movie['genres'])}"
                )

        for label, data in sorted(cluster_groups.items()):
            base_name = f"Cluster {label}" if label >= 0 else "Noise"
            # 클러스터 설명 라벨 생성 (상위 장르 + 영화 수)
            if cluster_info and base_name in cluster_info:
                info = cluster_info[base_name]
                top_genres = " / ".join(
                    g for g, _ in info.get("top_genres", [])[:3]
                )
                count = info.get("count", len(data["x"]))
                name = f"{base_name}: {top_genres} ({count}편)"
            else:
                name = base_name
            color = _COLORS[label % len(_COLORS)] if label >= 0 else "#999"
            fig.add_trace(go.Scatter(
                x=data["x"], y=data["y"], mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7),
                text=data["text"], hoverinfo="text", name=name,
            ))
    else:
        genre_groups = {}
        for movie in movies:
            genre = movie["genres"][0] if movie["genres"] else "Drama"
            if genre not in genre_groups:
                genre_groups[genre] = {"x": [], "y": [], "text": []}
            coord = coords.get(movie["title"])
            if coord is not None:
                genre_groups[genre]["x"].append(coord[0])
                genre_groups[genre]["y"].append(coord[1])
                genre_groups[genre]["text"].append(
                    f"<b>{movie['title']}</b><br>장르: {', '.join(movie['genres'])}"
                )

        for genre, data in genre_groups.items():
            color = config.GENRE_COLORS.get(genre, "#CCC")
            fig.add_trace(go.Scatter(
                x=data["x"], y=data["y"], mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7),
                text=data["text"], hoverinfo="text", name=genre,
            ))

    # 테스트 영화
    if test_movies:
        new_size = max(marker_size * 2, 12)
        for movie in test_movies:
            coord = coords.get(movie["title"])
            if coord is not None:
                fig.add_trace(go.Scatter(
                    x=[coord[0]], y=[coord[1]],
                    mode="markers+text",
                    marker=dict(
                        size=new_size, color="#FFD700", symbol="diamond",
                        line=dict(width=2, color="black"),
                    ),
                    text=[movie["title"]], textposition="top center",
                    textfont=dict(size=9),
                    name=f"[NEW] {movie['title']}",
                ))

    # 축 레이블
    x_title, y_title = "Dim 1", "Dim 2"
    if axis_labels:
        keys = list(axis_labels.keys())
        if len(keys) >= 1:
            x_title = f"X: {axis_labels[keys[0]]}"
        if len(keys) >= 2:
            y_title = f"Y: {axis_labels[keys[1]]}"

    fig.update_layout(
        title=f"영화 임베딩 2D 시각화 ({method_name})",
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        width=config.FIGURE_WIDTH,
        height=config.FIGURE_HEIGHT - 200,
    )

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  2D 시각화 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 4. 유사도 히트맵
# ═══════════════════════════════════════════════════════════════════

def create_similarity_heatmap(
    embedding,
    test_movies: list[dict],
    train_movies: list[dict],
    save_path: str = None,
):
    """신작 vs 학습 영화 간 코사인 유사도 히트맵."""
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "similarity_heatmap.html")

    test_titles = [m["title"] for m in test_movies]
    train_titles = [m["title"] for m in train_movies]

    # 배치 유사도 계산
    sim_matrix = embedding.compute_similarity_matrix(test_titles, train_titles)
    sim_matrix = np.round(sim_matrix, 3)

    # 학습 영화가 많으면 상위 N개만 표시
    if len(train_titles) > 50:
        top_n = 30
        top_indices = set()
        for row in sim_matrix:
            top_indices.update(np.argsort(row)[::-1][:top_n])
        top_indices = sorted(top_indices)
        train_titles_display = [train_titles[i] for i in top_indices]
        sim_display = sim_matrix[:, top_indices]
    else:
        train_titles_display = train_titles
        sim_display = sim_matrix

    fig = go.Figure(data=go.Heatmap(
        z=sim_display.tolist(),
        x=train_titles_display,
        y=test_titles,
        colorscale="YlOrRd",
        text=[[f"{v:.3f}" for v in row] for row in sim_display],
        texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate="신작: %{y}<br>기존: %{x}<br>유사도: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="코사인 유사도 히트맵 (신작 vs 학습 영화)",
        xaxis=dict(title="학습 영화", tickangle=45),
        yaxis=dict(title="신작 영화"),
        width=1400, height=500,
    )

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  히트맵 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 5. 평가 비교 차트
# ═══════════════════════════════════════════════════════════════════

def create_evaluation_charts(
    quant: dict,
    qual: dict,
    comparison: dict,
    save_path: str = None,
):
    """정량/정성 평가 비교 2x2 서브플롯 차트."""
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "evaluation_report.html")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "장르별 부적합 비율",
            "영화별 정량 메트릭",
            "적합 vs 부적합 메트릭 비교",
            "전체 메트릭 요약",
        ],
    )

    # (1,1) 장르별 부적합 비율
    genre_rates = comparison.get("genre_rates", {})
    if genre_rates:
        genres = list(genre_rates.keys())
        rates = list(genre_rates.values())
        colors = ["#E74C3C" if r > 0.5 else "#F39C12" if r > 0 else "#2ECC71"
                   for r in rates]
        fig.add_trace(go.Bar(
            x=genres, y=rates, marker_color=colors,
            text=[f"{r:.2f}" for r in rates], textposition="outside",
            textfont=dict(size=10),
            name="부적합률", showlegend=False,
        ), row=1, col=1)
        fig.update_yaxes(title_text="부적합률", range=[0, 1], row=1, col=1)

    # (1,2) 영화별 정량 메트릭
    per_query = quant.get("per_query", {})
    if per_query:
        titles = list(per_query.keys())
        short_titles = [t[:15] + "..." if len(t) > 18 else t for t in titles]
        sims = [per_query[t]["avg_similarity"] for t in titles]
        gp = [per_query[t]["genre_precision"] for t in titles]
        kp = [per_query[t]["keyword_precision"] for t in titles]

        fig.add_trace(go.Bar(x=short_titles, y=sims, name="유사도",
                             marker_color="#3498DB",
                             text=[f"{v:.2f}" for v in sims],
                             textposition="inside", insidetextanchor="end",
                             textfont=dict(size=8, color="white"),
                             ), row=1, col=2)
        fig.add_trace(go.Bar(x=short_titles, y=gp, name="장르정밀도",
                             marker_color="#2ECC71",
                             text=[f"{v:.2f}" for v in gp],
                             textposition="inside", insidetextanchor="end",
                             textfont=dict(size=8, color="white"),
                             ), row=1, col=2)
        fig.add_trace(go.Bar(x=short_titles, y=kp, name="키워드정밀도",
                             marker_color="#E67E22",
                             text=[f"{v:.2f}" for v in kp],
                             textposition="inside", insidetextanchor="end",
                             textfont=dict(size=8, color="white"),
                             ), row=1, col=2)
        fig.update_layout(barmode="group")

    # (2,1) 적합 vs 부적합 비교
    adequacy = comparison.get("adequacy", {})
    if adequacy and per_query:
        adeq_sims = [per_query[t]["avg_similarity"]
                     for t, a in adequacy.items() if a["adequate"] and t in per_query]
        inadeq_sims = [per_query[t]["avg_similarity"]
                       for t, a in adequacy.items() if not a["adequate"] and t in per_query]
        adeq_gp = [per_query[t]["genre_precision"]
                   for t, a in adequacy.items() if a["adequate"] and t in per_query]
        inadeq_gp = [per_query[t]["genre_precision"]
                     for t, a in adequacy.items() if not a["adequate"] and t in per_query]

        metrics_names = ["평균 유사도", "장르 정밀도"]
        adeq_vals = [
            float(np.mean(adeq_sims)) if adeq_sims else 0,
            float(np.mean(adeq_gp)) if adeq_gp else 0,
        ]
        inadeq_vals = [
            float(np.mean(inadeq_sims)) if inadeq_sims else 0,
            float(np.mean(inadeq_gp)) if inadeq_gp else 0,
        ]

        fig.add_trace(go.Bar(x=metrics_names, y=adeq_vals,
                             name="적합", marker_color="#2ECC71",
                             text=[f"{v:.2f}" for v in adeq_vals],
                             textposition="outside",
                             textfont=dict(size=10),
                             ), row=2, col=1)
        fig.add_trace(go.Bar(x=metrics_names, y=inadeq_vals,
                             name="부적합", marker_color="#E74C3C",
                             text=[f"{v:.2f}" for v in inadeq_vals],
                             textposition="outside",
                             textfont=dict(size=10),
                             ), row=2, col=1)

    # (2,2) 전체 메트릭 요약
    overall = quant.get("overall", {})
    if overall:
        metric_names = list(overall.keys())
        metric_vals = list(overall.values())
        fig.add_trace(go.Bar(
            x=metric_names, y=metric_vals,
            marker_color="#9B59B6", name="전체 평균", showlegend=False,
            text=[f"{v:.2f}" for v in metric_vals],
            textposition="outside",
            textfont=dict(size=10),
        ), row=2, col=2)
        fig.add_hline(y=config.THRESHOLD_AVG_SIMILARITY, line_dash="dash",
                      line_color="red", row=2, col=2,
                      annotation_text=f"유사도 임계값({config.THRESHOLD_AVG_SIMILARITY})")

    fig.update_layout(
        title="추천 평가 비교 분석", height=800, width=1400,
    )
    # 막대 위 텍스트가 잘리지 않도록 y축 상단 여유
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_yaxes(range=[0, 1.15], row=2, col=1)
    fig.update_yaxes(range=[0, 1.15], row=2, col=2)

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  평가 차트 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 6. 파라미터별 성능 비교 차트
# ═══════════════════════════════════════════════════════════════════

def create_parameter_comparison(
    sweep_results: list[dict],
    save_path: str = None,
):
    """
    파라미터 스윕 결과를 비교하는 차트.

    Args:
        sweep_results: [{"params": {...}, "metrics": {...}}, ...]
    """
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "parameter_comparison.html")

    if not sweep_results:
        print("  파라미터 스윕 결과가 없습니다.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "가중치별 평균 유사도",
            "클러스터 수별 다양성",
            "Top-K별 장르 정밀도",
            "전체 파라미터 비교",
        ],
    )

    run_ids = list(range(1, len(sweep_results) + 1))
    sims = [r["metrics"].get("avg_similarity", 0) for r in sweep_results]
    divs = [r["metrics"].get("avg_diversity", 0) for r in sweep_results]
    gps = [r["metrics"].get("avg_genre_precision", 0) for r in sweep_results]
    kps = [r["metrics"].get("avg_keyword_precision", 0) for r in sweep_results]

    hover_texts = []
    for r in sweep_results:
        p = r["params"]
        lines = [f"{k}: {v}" for k, v in p.items()]
        hover_texts.append("<br>".join(lines))

    fig.add_trace(go.Scatter(
        x=run_ids, y=sims, mode="lines+markers",
        name="평균 유사도", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#3498DB"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=run_ids, y=divs, mode="lines+markers",
        name="다양성", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#2ECC71"),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=run_ids, y=gps, mode="lines+markers",
        name="장르 정밀도", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#E67E22"),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=run_ids, y=kps, mode="lines+markers",
        name="키워드 정밀도", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#9B59B6"),
    ), row=2, col=2)

    fig.update_layout(title="파라미터별 성능 비교", height=800, width=1400)

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  파라미터 비교 차트 저장: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# 7. 민감도 분석 차트
# ═══════════════════════════════════════════════════════════════════

def create_sensitivity_charts(
    sensitivity_results: dict,
    save_path: str = None,
):
    """
    27조합 민감도 분석 결과 시각화.

    - 27조합 막대 차트 (겹침률, 순위상관, 유사도, 신뢰도)
    - 순위 비교 테이블 (기준선 vs 최고/최저 조합)
    """
    _ensure_results_dir()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "sensitivity_analysis.html")

    analysis = sensitivity_results["analysis"]
    rank_tables = sensitivity_results["rank_tables"]
    cluster_info = sensitivity_results.get("cluster_info", {})

    # 유사도 내림차순 정렬
    sorted_analysis = sorted(analysis, key=lambda x: -x["avg_similarity"])
    labels = [d["label"] for d in sorted_analysis]
    colors = ["#F39C12" if d["is_baseline"] else "#3498DB"
              for d in sorted_analysis]

    # ── 4개 서브플롯 ──
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "① 평균 유사도 (27조합, 유사도 내림차순)",
            "② Top-20 겹침률 (기준선 대비)",
            "③ Spearman 순위 상관계수 (기준선 대비)",
            "④ 신뢰도 σ (낮을수록 안정적)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # (1,1) 유사도
    fig.add_trace(go.Bar(
        x=labels,
        y=[d["avg_similarity"] for d in sorted_analysis],
        marker_color=colors,
        text=[f"{d['avg_similarity']:.3f}" for d in sorted_analysis],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="%{x}<br>유사도=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # (1,2) 겹침률
    fig.add_trace(go.Bar(
        x=labels,
        y=[d["overlap_ratio"] for d in sorted_analysis],
        marker_color=colors,
        text=[f"{d['overlap_ratio']:.0%}" for d in sorted_analysis],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="%{x}<br>겹침률=%{y:.1%}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    # (2,1) 순위 상관
    fig.add_trace(go.Bar(
        x=labels,
        y=[d["spearman_rho"] for d in sorted_analysis],
        marker_color=colors,
        text=[f"{d['spearman_rho']:.2f}" for d in sorted_analysis],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="%{x}<br>Spearman ρ=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    # (2,2) 신뢰도
    fig.add_trace(go.Bar(
        x=labels,
        y=[d["reliability_sim"] for d in sorted_analysis],
        marker_color=colors,
        text=[f"{d['reliability_sim']:.3f}" for d in sorted_analysis],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="%{x}<br>σ=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        title="파라미터 민감도 분석 — 27조합 (상/중/하)³ 비교<br>"
              "<sub>주황 = 기준선 (중,중,중), 파랑 = 변형</sub>",
        height=1000, width=1600,
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9))

    # ── 순위 비교 테이블 (HTML) ──
    table_html = _build_rank_comparison_html(rank_tables)

    # ── 27조합 상세 테이블 ──
    detail_html = _build_27combo_detail_html(sorted_analysis)

    chart_html = fig.to_html(include_plotlyjs=True, full_html=False)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>파라미터 민감도 분석 (27조합)</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #fafafa; }}
        .chart-section {{ margin-bottom: 40px; }}
        .detail-section {{ margin-top: 30px; }}
        .detail-section h2 {{ color: #2C3E50; border-bottom: 2px solid #8E44AD; padding-bottom: 8px; }}
        .detail-table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 15px 0; }}
        .detail-table th {{ background: #2C3E50; color: white; padding: 8px 10px; text-align: center; position: sticky; top: 0; }}
        .detail-table td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: center; color: #1a1a1a; }}
        .detail-table tr:nth-child(even) {{ background: #f2f2f2; }}
        .detail-table tr:hover {{ background: #EBF5FB; }}
        .detail-table .baseline {{ background: #FFF3CD !important; font-weight: bold; }}
        .rank-section {{ margin-top: 40px; }}
        .rank-section h2 {{ color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 8px; }}
        .query-title {{ color: #E74C3C; font-size: 18px; font-weight: bold; margin: 20px 0 10px; }}
        .rank-table {{ border-collapse: collapse; margin: 10px 0 30px; font-size: 13px; }}
        .rank-table th {{ background: #2C3E50; color: white; padding: 8px 12px; text-align: center; }}
        .rank-table td {{ padding: 6px 12px; border: 1px solid #ddd; }}
        .rank-table tr:nth-child(even) {{ background: #f2f2f2; }}
        .rank-up {{ color: #27AE60; font-weight: bold; }}
        .rank-down {{ color: #E74C3C; font-weight: bold; }}
        .rank-new {{ color: #8E44AD; font-weight: bold; }}
        .rank-same {{ color: #999; }}
        .tables-container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .table-block {{ flex: 1; min-width: 300px; }}
        .table-block h4 {{ margin: 5px 0; color: #34495E; }}
    </style>
</head>
<body>
    <div class="chart-section">
        {chart_html}
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:20px 0; font-size:13px; color:#444;">
            <div style="background:#f0f4f8; padding:14px 18px; border-radius:8px; border-left:4px solid #3498DB;">
                <b style="color:#2C3E50;">&#9312; 평균 유사도</b><br>
                27개 조합을 유사도가 높은 순서로 정렬한 막대 차트. 높을수록 추천 영화들이 쿼리와 유사하다는 뜻.
            </div>
            <div style="background:#f0f4f8; padding:14px 18px; border-radius:8px; border-left:4px solid #3498DB;">
                <b style="color:#2C3E50;">&#9313; Top-20 겹침률</b><br>
                기준선의 Top-20 목록과 얼마나 같은 영화가 포함되어 있는지의 비율. 100%이면 동일 목록.
            </div>
            <div style="background:#f0f4f8; padding:14px 18px; border-radius:8px; border-left:4px solid #3498DB;">
                <b style="color:#2C3E50;">&#9314; Spearman 순위 상관계수</b><br>
                공통으로 포함된 영화들의 순서가 얼마나 유지되는지 측정. 1.0이면 동일 순서, 0에 가까우면 순서가 크게 다름.
            </div>
            <div style="background:#f0f4f8; padding:14px 18px; border-radius:8px; border-left:4px solid #3498DB;">
                <b style="color:#2C3E50;">&#9315; 신뢰도 &sigma;</b><br>
                테스트 5편 간 유사도의 표준편차. 낮을수록 모든 영화에 대해 균일하게 좋은 추천을 한다는 뜻.
            </div>
        </div>
    </div>
    <div class="detail-section">
        <h2>27조합 상세 비교표 (장르, 키워드, 수치 = 상/중/하)</h2>
        {detail_html}
    </div>
    <div class="rank-section">
        <h2>순위 비교 — 기준선(중,중,중) vs 최고/최저 유사도 조합</h2>
        <p style="color:#666;">&#9650; = 순위 상승, &#9660; = 순위 하락, <span class="rank-new">NEW</span> = 새로 진입 &nbsp;|&nbsp; C# = 클러스터 번호</p>
        {_build_cluster_legend_html(cluster_info)}
        {table_html}
    </div>
</body>
</html>"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"  민감도 분석 차트 저장: {save_path}")
    return fig


def _val_color(val: float, low: float, high: float, reverse: bool = False) -> str:
    """값에 따라 녹색~빨강 배경색 반환. reverse=True면 낮을수록 녹색."""
    if high == low:
        return "background:transparent"
    ratio = (val - low) / (high - low)
    if reverse:
        ratio = 1.0 - ratio
    # 녹색(좋음) → 노랑(중간) → 빨강(나쁨)
    if ratio >= 0.5:
        # 녹색~노랑
        t = (ratio - 0.5) * 2
        r = int(255 * (1 - t))
        g = int(180 + 75 * t)
        return f"background:rgba({r},{g},80,0.25)"
    else:
        # 노랑~빨강
        t = ratio * 2
        g = int(180 * t)
        return f"background:rgba(255,{g},80,0.25)"


def _weight_bg(val: float) -> str:
    """가중치 값(0.5/1.0/1.5) → 파랑 계열 배경색."""
    if abs(val - 1.5) < 0.01:
        return "background:rgba(41,128,185,0.30)"   # 상: 진한 파랑
    elif abs(val - 1.0) < 0.01:
        return "background:rgba(133,193,233,0.25)"   # 중: 중간 파랑
    elif abs(val - 0.5) < 0.01:
        return "background:rgba(214,234,248,0.50)"   # 하: 연한 파랑
    return ""


def _build_27combo_detail_html(sorted_analysis: list) -> str:
    """27조합 상세 테이블 HTML (셀별 배경색 포함)."""
    if not sorted_analysis:
        return ""

    # 각 메트릭의 min/max 계산
    sims = [d["avg_similarity"] for d in sorted_analysis]
    gps = [d["genre_precision"] for d in sorted_analysis]
    ovls = [d["overlap_ratio"] for d in sorted_analysis]
    sprs = [d["spearman_rho"] for d in sorted_analysis]
    disps = [d["avg_rank_displacement"] for d in sorted_analysis]
    rels = [d["reliability_sim"] for d in sorted_analysis]

    rows = []
    for i, d in enumerate(sorted_analysis):
        cls = ' class="baseline"' if d["is_baseline"] else ""
        rows.append(
            f'<tr{cls}>'
            f'<td>{i+1}</td>'
            f'<td style="font-weight:bold">{d["label"]}</td>'
            f'<td style="{_weight_bg(d["weight_genre"])}">{d["weight_genre"]:.1f}</td>'
            f'<td style="{_weight_bg(d["weight_keyword"])}">{d["weight_keyword"]:.1f}</td>'
            f'<td style="{_weight_bg(d["weight_numeric"])}">{d["weight_numeric"]:.1f}</td>'
            f'<td style="{_val_color(d["avg_similarity"], min(sims), max(sims))}">{d["avg_similarity"]:.4f}</td>'
            f'<td style="{_val_color(d["genre_precision"], min(gps), max(gps))}">{d["genre_precision"]:.4f}</td>'
            f'<td style="{_val_color(d["overlap_ratio"], min(ovls), max(ovls))}">{d["overlap_ratio"]:.1%}</td>'
            f'<td style="{_val_color(d["spearman_rho"], min(sprs), max(sprs))}">{d["spearman_rho"]:.4f}</td>'
            f'<td style="{_val_color(d["avg_rank_displacement"], min(disps), max(disps), reverse=True)}">{d["avg_rank_displacement"]:.1f}</td>'
            f'<td style="{_val_color(d["reliability_sim"], min(rels), max(rels), reverse=True)}">{d["reliability_sim"]:.4f}</td>'
            f'</tr>'
        )
    return (
        '<table class="detail-table">'
        '<tr><th>#</th><th>조합</th><th>장르</th><th>키워드</th><th>수치</th>'
        '<th>유사도</th><th>장르정밀도</th><th>겹침률</th><th>순위상관 ρ</th>'
        '<th>순위변동</th><th>신뢰도σ</th></tr>'
        + "\n".join(rows)
        + '</table>'
        + '<p style="font-size:12px;color:#888;margin-top:8px;">'
        '가중치: '
        '<span style="background:rgba(41,128,185,0.30);padding:2px 10px;border-radius:3px;">상(1.5)</span> '
        '<span style="background:rgba(133,193,233,0.25);padding:2px 10px;border-radius:3px;">중(1.0)</span> '
        '<span style="background:rgba(214,234,248,0.50);padding:2px 10px;border-radius:3px;">하(0.5)</span> '
        '&nbsp;&nbsp;|&nbsp;&nbsp;메트릭: '
        '<span style="background:rgba(80,255,80,0.25);padding:2px 8px;border-radius:3px;">녹색=좋음</span> '
        '<span style="background:rgba(255,180,80,0.25);padding:2px 8px;border-radius:3px;">노랑=중간</span> '
        '<span style="background:rgba(255,0,80,0.25);padding:2px 8px;border-radius:3px;">빨강=나쁨</span> '
        '(순위변동·신뢰도σ는 낮을수록 녹색)</p>'
    )


_CLUSTER_BG = [
    "rgba(231,76,60,0.15)",   # 0: 빨강
    "rgba(52,152,219,0.15)",  # 1: 파랑
    "rgba(46,204,113,0.15)",  # 2: 초록
    "rgba(241,196,15,0.15)",  # 3: 노랑
    "rgba(142,68,173,0.15)",  # 4: 보라
    "rgba(230,126,34,0.15)",  # 5: 주황
    "rgba(26,188,156,0.15)",  # 6: 청록
    "rgba(52,73,94,0.12)",    # 7: 진회
    "rgba(0,188,212,0.15)",   # 8: 하늘
    "rgba(121,85,72,0.15)",   # 9: 갈색
    "rgba(96,125,139,0.15)",  # 10: 회색
    "rgba(255,87,34,0.15)",   # 11: 주홍
]


def _cluster_style(cluster_id: int) -> str:
    """클러스터 ID → 배경색 스타일."""
    if cluster_id < 0:
        return ""
    return f"background:{_CLUSTER_BG[cluster_id % len(_CLUSTER_BG)]}"


def _build_cluster_legend_html(cluster_info: dict) -> str:
    """클러스터 범례 HTML (C0~C7 + 상위 장르)."""
    items = []
    for i in range(max(8, len(cluster_info))):
        key = f"Cluster {i}"
        bg = _CLUSTER_BG[i % len(_CLUSTER_BG)]
        if key in cluster_info:
            info = cluster_info[key]
            genres = " / ".join(g for g, _ in info.get("top_genres", [])[:3])
            count = info.get("count", 0)
            label = f"C{i}: {genres} ({count}편)"
        else:
            label = f"C{i}"
        items.append(
            f'<span style="padding:3px 12px; border-radius:4px; '
            f'background:{bg}; white-space:nowrap;">{label}</span>'
        )
    return (
        '<div style="display:flex; gap:8px; flex-wrap:wrap; '
        'margin:10px 0 20px; font-size:12px;">'
        + "".join(items)
        + '</div>'
    )


def _build_rank_comparison_html(rank_tables: dict) -> str:
    """순위 비교 테이블 HTML (기준선 vs 최고/최저, 클러스터 정보 포함)."""
    html_parts = []

    for query_title, table_data in rank_tables.items():
        html_parts.append(f'<div class="query-title">{query_title}</div>')
        html_parts.append('<div class="tables-container">')

        # 기준선 테이블 (4열: 순위, 영화, 유사도, 클러스터)
        base_rows = table_data.get("baseline", [])
        html_parts.append('<div class="table-block">')
        html_parts.append('<h4>기준선 (중,중,중)</h4>')
        html_parts.append('<table class="rank-table">')
        html_parts.append('<tr><th>순위</th><th>영화</th><th>유사도</th><th>C#</th></tr>')
        for row in base_rows[:20]:
            rank, title, sim = row[0], row[1], row[2]
            cid = row[3] if len(row) > 3 else -1
            style = _cluster_style(cid)
            c_label = f"C{cid}" if cid >= 0 else "-"
            html_parts.append(
                f'<tr style="{style}">'
                f'<td>{rank}</td><td>{title}</td>'
                f'<td>{sim:.4f}</td><td>{c_label}</td></tr>'
            )
        html_parts.append('</table></div>')

        # 최고/최저 조합 테이블
        for tag, tag_label in [("best", "최고 유사도 조합"), ("worst", "최저 유사도 조합")]:
            rows = table_data.get("combos", {}).get(tag, [])
            if not rows:
                continue
            combo_label = rows[0][4] if len(rows[0]) > 4 else "?"
            html_parts.append('<div class="table-block">')
            html_parts.append(f'<h4>{tag_label} {combo_label}</h4>')
            html_parts.append('<table class="rank-table">')
            html_parts.append(
                '<tr><th>순위</th><th>영화</th><th>유사도</th><th>변동</th><th>C#</th></tr>'
            )
            for row in rows[:20]:
                rank, title, sim, change = row[0], row[1], row[2], row[3]
                cid = row[5] if len(row) > 5 else -1
                style = _cluster_style(cid)
                c_label = f"C{cid}" if cid >= 0 else "-"
                if change == "NEW":
                    cs = '<span class="rank-new">NEW</span>'
                elif isinstance(change, (int, float)):
                    if change > 0:
                        cs = f'<span class="rank-up">&#9650;{abs(int(change))}</span>'
                    elif change < 0:
                        cs = f'<span class="rank-down">&#9660;{abs(int(change))}</span>'
                    else:
                        cs = '<span class="rank-same">-</span>'
                else:
                    cs = str(change)
                html_parts.append(
                    f'<tr style="{style}">'
                    f'<td>{rank}</td><td>{title}</td>'
                    f'<td>{sim:.4f}</td><td>{cs}</td><td>{c_label}</td></tr>'
                )
            html_parts.append('</table></div>')

        html_parts.append('</div>')

    return "\n".join(html_parts)
