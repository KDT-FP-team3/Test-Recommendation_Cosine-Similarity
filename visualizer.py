"""
visualizer.py -- 시각화 모듈
====================================================================
파이프라인 결과를 받아 7종의 인터랙티브 Plotly HTML 시각화를 생성한다.

출력 파일 (results/):
  1. data_field_diagram.html  - Sankey: KMDB 필드 -> 4그룹 -> 499D
  2. embedding_3d.html        - 3D PCA 산점도 (클러스터 + 추천 연결선)
  3. embedding_2d.html        - 2D PCA 산점도 (클러스터 라벨 + 축 해석)
  4. similarity_heatmap.html  - 테스트 영화 vs 추천 영화 유사도 히트맵
  5. evaluation_report.html   - 평가 메트릭 막대 차트
  6. weight_impact.html       - 그룹 기여도 누적 막대 차트
  7. sensitivity_analysis.html - 민감도 분석 (기여도 차트 + 클러스터 정보 + 히트맵 + 상세 테이블)
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import config

_CLUSTER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
    "#9B59B6", "#E67E22", "#1ABC9C", "#34495E",
    "#FF5722", "#00BCD4", "#8BC34A", "#FF9800",
    "#607D8B", "#795548", "#009688", "#CDDC39",
    "#673AB7", "#03A9F4", "#F44336", "#4CAF50",
]

# 12개 클러스터용 무지개 색상 (최대 구분)
_RAINBOW_12 = [
    "#FF0000",  # C0  빨강
    "#FF7700",  # C1  주황
    "#FFCC00",  # C2  노랑
    "#88DD00",  # C3  연두
    "#00BB00",  # C4  초록
    "#00CCAA",  # C5  청록
    "#0099FF",  # C6  하늘
    "#0044FF",  # C7  파랑
    "#6600FF",  # C8  남색
    "#AA00FF",  # C9  보라
    "#FF00AA",  # C10 분홍
    "#FF3366",  # C11 진분홍
]


class MovieVisualizer:
    """파이프라인 결과를 HTML 시각화로 변환"""

    def __init__(self, pipeline_result):
        self.result = pipeline_result
        self.output_dir = config.RESULTS_DIR

    def _get_pc_group_labels(self):
        """각 PC 축의 주요 특징 그룹 기여도를 판별하여 라벨 생성."""
        reducer = self.result.get("reducer")
        emb = self.result.get("embedding")
        if reducer is None or emb is None:
            return {}

        components = reducer.get_components()
        if components is None:
            return {}

        # 그룹별 차원 경계
        g_end = emb.genre_dim
        k_end = g_end + emb.keyword_dim
        n_end = k_end + emb.numeric_dim
        t_end = n_end + emb.text_dim

        group_names = ["장르", "키워드", "수치", "텍스트"]
        group_ranges = [(0, g_end), (g_end, k_end), (k_end, n_end), (n_end, t_end)]

        labels = {}
        for i in range(min(len(components), 3)):
            comp = components[i]
            contributions = []
            for name, (start, end) in zip(group_names, group_ranges):
                contributions.append((name, np.sum(comp[start:end] ** 2)))

            total = sum(c for _, c in contributions)
            if total < 1e-10:
                continue

            parts = [f"{name} {val / total:.0%}"
                     for name, val in sorted(contributions, key=lambda x: x[1], reverse=True)]
            labels[i] = " / ".join(parts)

        return labels

    def generate_all(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._gen_sankey()
        self._gen_3d_scatter()
        self._gen_2d_scatter()
        self._gen_heatmap()
        self._gen_evaluation()
        self._gen_weight_impact()
        print(f"[visualizer] 시각화 {self.output_dir}/ 에 저장 완료")

    # 1. Sankey: KMDB 필드 -> 전처리 -> 그룹(L2+가중치) -> 499D
    def _gen_sankey(self):
        weights = config.get_effective_weights()
        wg = weights["genre"]
        wk = weights["keyword"]
        wn = weights["numeric"]
        wt = weights["text"]

        # 그룹 색상 팔레트
        C_GENRE = "#E74C3C"     # 빨강
        C_KW    = "#3498DB"     # 파랑
        C_NUM   = "#F1C40F"     # 노랑
        C_TEXT  = "#2ECC71"     # 초록
        C_META  = "#95A5A6"     # 회색

        def _alpha(hex_color, a=0.35):
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},{a})"

        # 차원 비율 압축 (384D가 전체를 지배하지 않도록 제곱근 스케일)
        import math
        def _scaled(dim):
            return max(math.sqrt(dim), 1.0)

        # ==============================================================
        # 노드 정의 (4단계)
        # ==============================================================
        node_labels = []
        node_colors = []
        node_x = []
        node_y = []
        node_hovers = []

        def add_node(label, color, x, y, hover=""):
            idx = len(node_labels)
            node_labels.append(label)
            node_colors.append(color)
            node_x.append(x)
            node_y.append(y)
            node_hovers.append(hover or label)
            return idx

        # --- Layer 1: KMDB 원본 필드 (x=0.01) ---
        n_genre    = add_node("genre\n(장르 66종)",    C_GENRE, 0.01, 0.01,
                              "KMDB 원본 장르 (66종)\n영화별 1~5개 장르 태그")
        n_keywords = add_node("keywords\n(키워드)",     C_KW,    0.01, 0.14,
                              "KMDB 키워드 텍스트\n영화별 다수 키워드")
        n_runtime  = add_node("runtime\n(러닝타임)",    C_NUM,   0.01, 0.30,
                              "상영 시간 (분)\n결측 시 0.5로 대체")
        n_year     = add_node("prodYear\n(제작연도)",   C_NUM,   0.01, 0.36,
                              "제작 연도 (1980~2026)")
        n_actors   = add_node("actors\n(출연진)",       C_NUM,   0.01, 0.42,
                              "출연 배우 목록\n→ 출연진 수 파생")
        n_plots    = add_node("plots\n(줄거리)",        C_TEXT,  0.01, 0.55,
                              "줄거리 텍스트\n→ 문장 임베딩 입력")
        n_title    = add_node("title (제목)",           C_META,  0.01, 0.74,
                              "영화 제목 (식별자)")
        n_nation   = add_node("nation (제작국)",        C_META,  0.01, 0.78,
                              "제작 국가")
        n_direct   = add_node("directors (감독)",       C_META,  0.01, 0.82,
                              "감독 정보")
        n_rating   = add_node("rating (등급)",          C_META,  0.01, 0.86,
                              "관람 등급")
        n_poster   = add_node("posters (포스터)",       C_META,  0.01, 0.90,
                              "포스터 이미지 URL")
        n_audi     = add_node("audiAcc (관객수)",       C_META,  0.01, 0.94,
                              "누적 관객수")
        n_awards   = add_node("Awards (수상)",          C_META,  0.01, 0.98,
                              "수상 정보")

        # --- Layer 2: 전처리/인코딩 방법 (x=0.35) ---
        n_proc_genre = add_node("장르 매핑\n(66→30 원-핫)",          C_GENRE, 0.35, 0.05,
                                "GENRE_MAP으로 66종→30종 표준화\n원-핫 인코딩 (30D)")
        n_proc_kw    = add_node("키워드 필터\n(상위 80 바이너리)",   C_KW,    0.35, 0.18,
                                "빈도 ≥93 기준 상위 80개 선별\n바이너리 인코딩 (80D)")
        n_proc_num   = add_node("수치 파생\n(Min-Max 정규화)",       C_NUM,   0.35, 0.36,
                                "5종 파생 수치 (0~1 범위)\n"
                                "runtime_norm, year_norm,\n"
                                "keyword_richness, cast_size_norm,\n"
                                "genre_count_norm")
        n_proc_text  = add_node("문장 임베딩\n(MiniLM-L12-v2)",     C_TEXT,  0.35, 0.55,
                                f"sentence-transformers\n{config.TEXT_MODEL_NAME.split('/')[-1]}\n→ {config.TEXT_EMBED_DIM}D 벡터")
        n_proc_meta  = add_node("메타데이터\n(비임베딩)",            C_META,  0.35, 0.86,
                                "임베딩에 포함되지 않는\n참조/표시용 정보 (7종)")

        # --- Layer 3: 특징 그룹 (L2 + 가중치) (x=0.65) ---
        n_grp_genre = add_node(f"장르 (30D)\nL2정규화 · ×{wg}",     C_GENRE, 0.65, 0.05,
                               f"장르 원-핫 벡터 (30D)\nL2 정규화 후 가중치 ×{wg} 적용")
        n_grp_kw    = add_node(f"키워드 (80D)\nL2정규화 · ×{wk}",   C_KW,    0.65, 0.18,
                               f"키워드 바이너리 벡터 (80D)\nL2 정규화 후 가중치 ×{wk} 적용")
        n_grp_num   = add_node(f"수치 (5D)\nL2정규화 · ×{wn}",      C_NUM,   0.65, 0.36,
                               f"파생 수치 벡터 (5D)\nL2 정규화 후 가중치 ×{wn} 적용")
        n_grp_text  = add_node(f"텍스트 (384D)\nL2정규화 · ×{wt}",  C_TEXT,  0.65, 0.55,
                               f"줄거리 임베딩 벡터 (384D)\nL2 정규화 후 가중치 ×{wt} 적용")

        # --- Layer 4: 최종 벡터 (x=0.99) ---
        n_final = add_node(f"499D\n하이브리드 벡터",                  "#2C3E50", 0.99, 0.30,
                           f"4그룹 Concatenation\n"
                           f"장르(30) + 키워드(80) + 수치(5) + 텍스트(384)\n"
                           f"= {config.TOTAL_DIM}D")
        n_meta_final = add_node("메타데이터\n(참조 정보)",            C_META, 0.99, 0.86,
                                "제목, 감독, 국가, 등급 등\n표시/검색용 (비임베딩)")

        # ==============================================================
        # 링크 정의 (Sankey 흐름 보존: 노드 입출력 합 일치 필수)
        # ==============================================================
        sources = []
        targets = []
        values = []
        link_colors = []
        link_hovers = []

        def add_link(src, tgt, vis_val, color, hover=""):
            """vis_val: 이미 스케일링된 시각적 밴드 폭"""
            sources.append(src)
            targets.append(tgt)
            values.append(vis_val)
            link_colors.append(_alpha(color, 0.35))
            link_hovers.append(hover)

        # 흐름 보존을 위한 스케일 값 사전 계산
        s1   = _scaled(1)      # 1.0   (1D 개별 수치)
        s30  = _scaled(30)     # 5.48  (장르 30D)
        s80  = _scaled(80)     # 8.94  (키워드 80D)
        s384 = _scaled(384)    # 19.6  (텍스트 384D)
        s_num  = 5 * s1        # 5.0   (수치 5 × 1D 입력 합)
        s_meta = 7 * s1        # 7.0   (메타데이터 7 × 1D 입력 합)

        # Layer 1 → Layer 2
        add_link(n_genre,    n_proc_genre, s30,  C_GENRE, "장르 태그 → 30개 표준 카테고리 원-핫 (30D)")
        add_link(n_genre,    n_proc_num,   s1,   C_NUM,   "장르 수 → genre_count_norm (1D)")
        add_link(n_keywords, n_proc_kw,    s80,  C_KW,    "키워드 → 상위 80개 바이너리 (80D)")
        add_link(n_keywords, n_proc_num,   s1,   C_NUM,   "키워드 수 → keyword_richness (1D)")
        add_link(n_runtime,  n_proc_num,   s1,   C_NUM,   "러닝타임 → runtime_norm (1D)")
        add_link(n_year,     n_proc_num,   s1,   C_NUM,   "제작연도 → year_norm (1D)")
        add_link(n_actors,   n_proc_num,   s1,   C_NUM,   "출연진 수 → cast_size_norm (1D)")
        add_link(n_plots,    n_proc_text,  s384, C_TEXT,  f"줄거리 → {config.TEXT_EMBED_DIM}D 문장 임베딩")

        for n_meta_src in [n_title, n_nation, n_direct, n_rating, n_poster, n_audi, n_awards]:
            add_link(n_meta_src, n_proc_meta, s1, C_META, "비임베딩 메타데이터 (1종)")

        # Layer 2 → Layer 3 (입력 합 = 출력)
        add_link(n_proc_genre, n_grp_genre, s30,   C_GENRE, "원-핫 30D → L2 정규화")
        add_link(n_proc_kw,    n_grp_kw,    s80,   C_KW,    "바이너리 80D → L2 정규화")
        add_link(n_proc_num,   n_grp_num,   s_num, C_NUM,   "수치 5D → L2 정규화")
        add_link(n_proc_text,  n_grp_text,  s384,  C_TEXT,  f"임베딩 {config.TEXT_EMBED_DIM}D → L2 정규화")

        # Layer 3 → Layer 4 (입력 합 = 출력)
        add_link(n_grp_genre, n_final, s30,   C_GENRE, f"장르 30D × {wg} → 연결")
        add_link(n_grp_kw,    n_final, s80,   C_KW,    f"키워드 80D × {wk} → 연결")
        add_link(n_grp_num,   n_final, s_num, C_NUM,   f"수치 5D × {wn} → 연결")
        add_link(n_grp_text,  n_final, s384,  C_TEXT,  f"텍스트 384D × {wt} → 연결")
        add_link(n_proc_meta, n_meta_final, s_meta, C_META, "7종 메타데이터 (비임베딩)")

        # ==============================================================
        # Figure 생성
        # ==============================================================
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label=node_labels,
                color=node_colors,
                x=node_x,
                y=node_y,
                pad=18,
                thickness=22,
                customdata=node_hovers,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=link_hovers,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        ))
        fig.update_layout(
            title=dict(
                text=(f"KMDB 데이터 파이프라인: 원본 필드 → 전처리 → L2 정규화 + 가중치 → "
                      f"{config.TOTAL_DIM}D 하이브리드 벡터"
                      f"<br><sub>가중치: 장르 ×{wg} | 키워드 ×{wk} | 수치 ×{wn} | 텍스트 ×{wt}</sub>"),
                font=dict(size=20),
            ),
            font=dict(size=16),
            width=config.FIGURE_WIDTH,
            height=max(config.FIGURE_HEIGHT, 1300),
            margin=dict(t=80, b=30, l=10, r=10),
        )
        pio.write_html(fig, os.path.join(self.output_dir, "data_field_diagram.html"))

    # 2. 3D 산점도
    def _gen_3d_scatter(self):
        coords = self.result["coords"]
        train_movies = self.result["train_movies"]
        test_movies = self.result["test_movies"]
        clusters = self.result["clusters"]
        recommendations = self.result["recommendations"]

        fig = go.Figure()

        # 학습 데이터 (클러스터별 색상)
        unique_clusters = sorted(set(clusters))
        for ci, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            indices = np.where(mask)[0]
            c_movies = [train_movies[i] for i in indices]
            c_ids = [m["id"] for m in c_movies]
            c_coords = np.array([coords[mid] for mid in c_ids if mid in coords])
            if len(c_coords) == 0:
                continue

            titles = [m["title"] for m in c_movies if m["id"] in coords]
            hover = [f"{t} ({m['year']})<br>장르: {', '.join(m.get('genres', [])[:3])}"
                     for t, m in zip(titles, c_movies) if m["id"] in coords]

            color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
            if c_coords.shape[1] >= 3:
                fig.add_trace(go.Scatter3d(
                    x=c_coords[:, 0], y=c_coords[:, 1], z=c_coords[:, 2],
                    mode="markers", marker=dict(size=3, color=color, opacity=0.5),
                    name=f"클러스터 {cluster_id}", text=hover, hoverinfo="text",
                ))

        # 테스트 영화
        for tm in test_movies:
            mid = tm["id"]
            if mid in coords and coords[mid].shape[0] >= 3:
                tc = coords[mid]
                fig.add_trace(go.Scatter3d(
                    x=[tc[0]], y=[tc[1]], z=[tc[2]],
                    mode="markers+text",
                    marker=dict(size=10, color="#FF0000", symbol="diamond"),
                    text=[tm["title"]], textposition="top center",
                    name=f"테스트: {tm['title']}",
                ))

                # 추천 연결선
                recs = recommendations.get(tm["title"], [])
                for r in recs[:5]:
                    rid = r.get("id")
                    if rid and rid in coords and coords[rid].shape[0] >= 3:
                        rc = coords[rid]
                        fig.add_trace(go.Scatter3d(
                            x=[tc[0], rc[0]], y=[tc[1], rc[1]], z=[tc[2], rc[2]],
                            mode="lines",
                            line=dict(color="#FF0000", width=2),
                            showlegend=False, hoverinfo="skip",
                        ))

        # PC 축별 주요 그룹 라벨 생성
        pc_labels = self._get_pc_group_labels()
        var_ratio = []
        reducer = self.result.get("reducer")
        if reducer is not None:
            var_ratio = reducer.get_explained_variance()

        def _make_3d_label(pc_idx, pc_name):
            parts = []
            if len(var_ratio) > pc_idx:
                parts.append(f"분산 {var_ratio[pc_idx]:.1%}")
            if pc_idx in pc_labels:
                parts.append(pc_labels[pc_idx])
            return f"{pc_name} ({', '.join(parts)})" if parts else pc_name

        fig.update_layout(
            title="3D 임베딩 공간 (PCA) -- 클러스터 및 추천 연결선",
            width=config.FIGURE_WIDTH, height=config.FIGURE_HEIGHT,
            scene=dict(
                xaxis_title=_make_3d_label(0, "PC1"),
                yaxis_title=_make_3d_label(1, "PC2"),
                zaxis_title=_make_3d_label(2, "PC3"),
            ),
        )
        pio.write_html(fig, os.path.join(self.output_dir, "embedding_3d.html"))

    # 3. 2D 산점도
    def _gen_2d_scatter(self):
        coords = self.result["coords"]
        train_movies = self.result["train_movies"]
        clusters = self.result["clusters"]
        cluster_info = self.result.get("cluster_info", {})

        # PCA 분산 비율 + 주요 그룹으로 축 라벨 생성
        reducer = self.result.get("reducer")
        pc_labels = self._get_pc_group_labels()
        x_label = "PC1"
        y_label = "PC2"
        if reducer is not None:
            var_ratio = reducer.get_explained_variance()
            if len(var_ratio) >= 2:
                x_parts = [f"분산 설명률 {var_ratio[0]:.1%}"]
                y_parts = [f"분산 설명률 {var_ratio[1]:.1%}"]
                if 0 in pc_labels:
                    x_parts.append(pc_labels[0])
                if 1 in pc_labels:
                    y_parts.append(pc_labels[1])
                x_label = f"PC1 ({', '.join(x_parts)})"
                y_label = f"PC2 ({', '.join(y_parts)})"
            elif pc_labels:
                if 0 in pc_labels:
                    x_label = f"PC1 ({pc_labels[0]})"
                if 1 in pc_labels:
                    y_label = f"PC2 ({pc_labels[1]})"

        fig = go.Figure()
        unique_clusters = sorted(set(clusters))
        annotations = []

        for ci, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            indices = np.where(mask)[0]
            c_movies = [train_movies[i] for i in indices]
            c_ids = [m["id"] for m in c_movies]
            c_coords = np.array([coords[mid][:2] for mid in c_ids if mid in coords])
            if len(c_coords) == 0:
                continue

            # 클러스터 정보에서 주요 장르 가져오기
            info_key = f"클러스터 {cluster_id}"
            cdata = cluster_info.get(info_key, {})
            top_genres = cdata.get("top_genres", [])
            genre_names = [g for g, _ in top_genres[:3]]
            genre_label = ", ".join(genre_names) if genre_names else "기타"
            cluster_label = f"클러스터 {cluster_id}: {genre_label}"

            # 무지개 색상 (클러스터 인덱스 기반, 최대 구분)
            rainbow_color = _RAINBOW_12[ci % len(_RAINBOW_12)]

            titles = [m["title"] for m in c_movies if m["id"] in coords]
            fig.add_trace(go.Scatter(
                x=c_coords[:, 0], y=c_coords[:, 1],
                mode="markers", marker=dict(size=4, color=rainbow_color, opacity=0.6),
                name=cluster_label, text=titles, hoverinfo="text",
            ))

            # 클러스터 중심에 라벨 표시
            cx, cy = float(np.mean(c_coords[:, 0])), float(np.mean(c_coords[:, 1]))
            # 배경색: 무지개 색상, 텍스트색: 명암 판단
            r_val = int(rainbow_color[1:3], 16)
            g_val = int(rainbow_color[3:5], 16)
            b_val = int(rainbow_color[5:7], 16)
            text_color = "#fff" if (r_val * 0.299 + g_val * 0.587 + b_val * 0.114) < 150 else "#222"

            count = cdata.get("count", len(c_movies))
            annotations.append(dict(
                x=cx, y=cy, text=f"<b>C{cluster_id}</b> {genre_label} ({count}편)",
                showarrow=False, font=dict(size=11, color=text_color),
                bgcolor=rainbow_color, borderpad=4, borderwidth=1,
                bordercolor=rainbow_color, opacity=0.9,
            ))

        fig.update_layout(
            title="2D 임베딩 공간 (PCA) -- 클러스터별 주요 장르",
            xaxis_title=x_label, yaxis_title=y_label,
            width=config.FIGURE_WIDTH, height=config.FIGURE_HEIGHT,
            annotations=annotations,
        )
        pio.write_html(fig, os.path.join(self.output_dir, "embedding_2d.html"))

    # 4. 유사도 히트맵
    def _gen_heatmap(self):
        recommendations = self.result["recommendations"]
        test_movies = self.result["test_movies"]

        if not test_movies or not recommendations:
            return

        test_titles = [tm["title"] for tm in test_movies]
        max_recs = min(20, max(len(recs) for recs in recommendations.values()) if recommendations else 0)

        z_data = []
        y_labels = []
        x_labels = []

        for tt in test_titles:
            recs = recommendations.get(tt, [])
            if not recs:
                continue
            y_labels.append(tt)
            row = [r["similarity"] for r in recs[:max_recs]]
            while len(row) < max_recs:
                row.append(0)
            z_data.append(row)
            if not x_labels:
                x_labels = [r["title"][:15] for r in recs[:max_recs]]

        if not z_data:
            return

        fig = go.Figure(go.Heatmap(
            z=z_data, x=x_labels, y=y_labels,
            colorscale="YlOrRd", text=np.round(z_data, 3), texttemplate="%{text}",
            hovertemplate="테스트: %{y}<br>추천: %{x}<br>유사도: %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title="테스트 영화 x 추천 결과 코사인 유사도 히트맵",
            xaxis_title="추천 영화", yaxis_title="테스트 영화",
            width=config.FIGURE_WIDTH, height=max(500, len(y_labels) * 100),
        )
        pio.write_html(fig, os.path.join(self.output_dir, "similarity_heatmap.html"))

    # 5. 평가 보고서
    def _gen_evaluation(self):
        quant = self.result.get("quant", {})
        per_query = quant.get("per_query", {})
        if not per_query:
            return

        titles = list(per_query.keys())
        metrics_names = ["avg_similarity", "genre_precision", "keyword_precision",
                         "diversity", "text_coherence"]
        metric_labels = ["평균 유사도", "장르 정밀도", "키워드 정밀도",
                         "다양성", "텍스트 일관성"]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metric_labels + ["적합/부적합 판정"],
        )

        colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F1C40F", "#9B59B6"]
        for i, (mn, ml) in enumerate(zip(metrics_names, metric_labels)):
            row, col = (i // 3) + 1, (i % 3) + 1
            vals = [per_query[t].get(mn, 0) for t in titles]
            short_titles = [t[:12] for t in titles]
            fig.add_trace(go.Bar(
                x=short_titles, y=vals, name=ml,
                marker_color=colors[i],
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ), row=row, col=col)

        # 적합/부적합
        comparison = self.result.get("comparison", {})
        adequacy = comparison.get("adequacy", {})
        adeq_vals = [1 if adequacy.get(t, {}).get("adequate", False) else 0 for t in titles]
        short_titles = [t[:12] for t in titles]
        fig.add_trace(go.Bar(
            x=short_titles, y=adeq_vals, name="적합 여부",
            marker_color=["#2ECC71" if v else "#E74C3C" for v in adeq_vals],
            text=["적합" if v else "부적합" for v in adeq_vals],
            textposition="outside",
        ), row=2, col=3)

        fig.update_layout(
            title="추천 평가 보고서",
            showlegend=False,
            width=config.FIGURE_WIDTH, height=config.FIGURE_HEIGHT,
        )
        pio.write_html(fig, os.path.join(self.output_dir, "evaluation_report.html"))

    # 6. 그룹 기여도
    def _gen_weight_impact(self):
        recommendations = self.result.get("recommendations", {})
        if not recommendations:
            return

        fig = go.Figure()
        group_names = {"genre": "장르", "keyword": "키워드",
                       "numeric": "수치", "text": "텍스트"}
        group_colors = {"genre": "#E74C3C", "keyword": "#3498DB",
                        "numeric": "#F1C40F", "text": "#2ECC71"}

        all_labels = []
        group_data = {g: [] for g in group_names}

        for qt, recs in recommendations.items():
            for r in recs[:10]:
                gs = r.get("group_similarity", {})
                label = f"{qt[:8]}->{r['title'][:10]}"
                all_labels.append(label)
                for g in group_names:
                    group_data[g].append(gs.get(g, 0))

        for g, gname in group_names.items():
            fig.add_trace(go.Bar(
                x=all_labels, y=group_data[g], name=gname,
                marker_color=group_colors[g],
            ))

        fig.update_layout(
            barmode="stack",
            title="추천별 그룹 유사도 기여도 분해",
            xaxis_title="쿼리 -> 추천", yaxis_title="코사인 유사도",
            width=config.FIGURE_WIDTH, height=config.FIGURE_HEIGHT,
            xaxis_tickangle=-45,
        )
        pio.write_html(fig, os.path.join(self.output_dir, "weight_impact.html"))

    # 7. 민감도 분석 HTML
    def generate_sensitivity_html(self, all_results, analysis,
                                   train_movies, test_movies, clusters,
                                   cluster_info=None):
        """
        81조합 민감도 분석 결과를 standalone HTML로 생성한다.

        Parameters
        ----------
        all_results : dict
            {combo_label: {query_title: [(movie_id, similarity), ...]}}
        analysis : list[dict]
            각 항목에 combo, wg, wk, wn, wt, query, overlap, spearman_rho,
            avg_displacement, avg_similarity, genre_precision 포함
        train_movies : list[dict]
        test_movies : list[dict]
        clusters : np.ndarray
            학습 데이터 클러스터 레이블
        cluster_info : dict, optional
            클러스터별 장르/키워드 정보 (clustering.get_cluster_info 결과)
        """
        import json as _json

        train_data = {m["id"]: m for m in train_movies}
        train_ids = [m["id"] for m in train_movies]
        # 클러스터 매핑: movie_id -> cluster_id
        cluster_map = {}
        for i, mid in enumerate(train_ids):
            if i < len(clusters):
                cluster_map[mid] = int(clusters[i])

        # 클러스터 정보 테이블 HTML
        cluster_info_html = ""
        if cluster_info:
            genre_colors = getattr(config, "GENRE_COLORS", {})
            cluster_info_html = '<div class="ci-wrap"><h4>클러스터 정보</h4><table class="ci-table"><thead><tr>'
            cluster_info_html += '<th>클러스터</th><th>영화 수</th><th>주요 장르</th><th>주요 키워드</th>'
            cluster_info_html += '</tr></thead><tbody>'
            for cid in range(config.KMEANS_N_CLUSTERS):
                key = f"클러스터 {cid}"
                cdata = cluster_info.get(key, {})
                if not cdata:
                    continue
                count = cdata.get("count", 0)
                cl_color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]

                # 장르 배지
                genre_cells = ""
                for g, cnt in cdata.get("top_genres", [])[:3]:
                    gc = genre_colors.get(g, "#95a5a6")
                    genre_cells += f'<span class="ci-genre" style="background:{gc};color:#fff">{g}({cnt})</span> '

                # 키워드
                kw_text = ", ".join(f"{k}({cnt})" for k, cnt in cdata.get("top_keywords", [])[:3])

                cluster_info_html += f'<tr>'
                cluster_info_html += f'<td><span class="cluster-badge" style="background:{cl_color};color:#fff">C{cid}</span></td>'
                cluster_info_html += f'<td>{count:,}</td>'
                cluster_info_html += f'<td class="ci-genres">{genre_cells}</td>'
                cluster_info_html += f'<td class="ci-kw">{kw_text}</td>'
                cluster_info_html += '</tr>'
            cluster_info_html += '</tbody></table></div>'

        # ---------------------------------------------------------------
        # 81조합별 집계 (테스트 영화 평균)
        # ---------------------------------------------------------------
        combo_agg = {}  # combo_label -> {metrics...}
        for item in analysis:
            label = item["combo"]
            if label not in combo_agg:
                combo_agg[label] = {
                    "wg": item["wg"], "wk": item["wk"],
                    "wn": item.get("wn", 0.5), "wt": item["wt"],
                    "avg_similarities": [], "genre_precisions": [],
                    "overlaps": [], "spearman_rhos": [],
                    "avg_displacements": [],
                }
            agg = combo_agg[label]
            agg["avg_similarities"].append(item.get("avg_similarity", 0))
            agg["genre_precisions"].append(item.get("genre_precision", 0))
            agg["overlaps"].append(item.get("overlap", 0))
            agg["spearman_rhos"].append(item.get("spearman_rho", 0))
            agg["avg_displacements"].append(item.get("avg_displacement", 0))

        combo_rows = []
        for label, agg in combo_agg.items():
            sims = agg["avg_similarities"]
            combo_rows.append({
                "combo": label,
                "wg": agg["wg"], "wk": agg["wk"], "wn": agg["wn"], "wt": agg["wt"],
                "avg_similarity": float(np.mean(sims)) if sims else 0,
                "genre_precision": float(np.mean(agg["genre_precisions"])) if agg["genre_precisions"] else 0,
                "overlap": float(np.mean(agg["overlaps"])) if agg["overlaps"] else 0,
                "spearman_rho": float(np.mean(agg["spearman_rhos"])) if agg["spearman_rhos"] else 0,
                "avg_displacement": float(np.mean(agg["avg_displacements"])) if agg["avg_displacements"] else 0,
                "sigma": float(np.std(sims)) if len(sims) > 1 else 0,
            })

        # 유사도 내림차순 정렬
        combo_rows.sort(key=lambda r: r["avg_similarity"], reverse=True)

        baseline_label = "(중,중,중,중)"

        # ---------------------------------------------------------------
        # Section 1: Plotly 차트 데이터
        # ---------------------------------------------------------------
        labels = [r["combo"] for r in combo_rows]
        avg_sims = [r["avg_similarity"] for r in combo_rows]
        overlaps = [r["overlap"] for r in combo_rows]
        rhos = [r["spearman_rho"] for r in combo_rows]
        sigmas = [r["sigma"] for r in combo_rows]

        bar_colors = ["#F39C12" if l == baseline_label else "#3498DB" for l in labels]

        chart_data = {
            "labels": labels,
            "avg_sims": [round(v, 4) for v in avg_sims],
            "overlaps": [round(v, 4) for v in overlaps],
            "rhos": [round(v, 4) for v in rhos],
            "sigmas": [round(v, 4) for v in sigmas],
            "bar_colors": bar_colors,
        }

        # ---------------------------------------------------------------
        # Section 2: Detail table HTML
        # ---------------------------------------------------------------
        def _level_label(val):
            for name, v in config.SENSITIVITY_LEVELS.items():
                if abs(v - val) < 0.01:
                    return name
            return f"{val:.1f}"

        def _weight_bg(val):
            if abs(val - 1.5) < 0.01:
                return "rgba(41,128,185,0.30)"    # 상: 파랑
            elif abs(val - 1.0) < 0.01:
                return "rgba(39,174,96,0.25)"      # 중: 초록
            else:
                return "rgba(230,126,34,0.30)"     # 하: 주황

        def _metric_color(val, low, high, invert=False):
            """green(good)=high -> yellow -> red(bad)=low.  invert: lower=better."""
            if high - low < 1e-10:
                return "rgba(255,255,255,0)"
            ratio = (val - low) / (high - low)
            ratio = max(0, min(1, ratio))
            if invert:
                ratio = 1 - ratio
            # ratio 1=good(green), 0=bad(red)
            if ratio >= 0.5:
                # green to yellow
                t = (ratio - 0.5) * 2
                r = int(255 * (1 - t))
                g = int(180 + 40 * t)
                return f"rgba({r},{g},80,0.25)"
            else:
                # yellow to red
                t = ratio * 2
                r = int(220 + 35 * (1 - t))
                g = int(180 * t)
                return f"rgba({r},{g},60,0.25)"

        # 범위 계산
        sim_vals = [r["avg_similarity"] for r in combo_rows]
        gp_vals = [r["genre_precision"] for r in combo_rows]
        ov_vals = [r["overlap"] for r in combo_rows]
        rho_vals = [r["spearman_rho"] for r in combo_rows]
        disp_vals = [r["avg_displacement"] for r in combo_rows]
        sig_vals = [r["sigma"] for r in combo_rows]

        sim_lo, sim_hi = (min(sim_vals), max(sim_vals)) if sim_vals else (0, 1)
        gp_lo, gp_hi = (min(gp_vals), max(gp_vals)) if gp_vals else (0, 1)
        ov_lo, ov_hi = (min(ov_vals), max(ov_vals)) if ov_vals else (0, 1)
        rho_lo, rho_hi = (min(rho_vals), max(rho_vals)) if rho_vals else (0, 1)
        disp_lo, disp_hi = (min(disp_vals), max(disp_vals)) if disp_vals else (0, 1)
        sig_lo, sig_hi = (min(sig_vals), max(sig_vals)) if sig_vals else (0, 1)

        def _level_span(val):
            """레벨값을 배경색 있는 span으로 변환."""
            label = _level_label(val)
            bg = _weight_bg(val)
            return f'<span style="background:{bg};padding:2px 6px;border-radius:3px">{label}</span>'

        table_rows_html = ""
        for idx, r in enumerate(combo_rows):
            is_baseline = r["combo"] == baseline_label
            row_class = ' class="baseline"' if is_baseline else ""

            combo_colored = f"({_level_span(r['wg'])},{_level_span(r['wk'])},{_level_span(r['wn'])},{_level_span(r['wt'])})"

            table_rows_html += f"""<tr{row_class}>
  <td>{idx+1}</td>
  <td class="combo-cell">{combo_colored}</td>
  <td style="background:{_weight_bg(r['wg'])}">{r['wg']:.1f}</td>
  <td style="background:{_weight_bg(r['wk'])}">{r['wk']:.1f}</td>
  <td style="background:{_weight_bg(r['wn'])}">{r['wn']:.1f}</td>
  <td style="background:{_weight_bg(r['wt'])}">{r['wt']:.1f}</td>
  <td style="background:{_metric_color(r['avg_similarity'], sim_lo, sim_hi)}">{r['avg_similarity']:.4f}</td>
  <td style="background:{_metric_color(r['genre_precision'], gp_lo, gp_hi)}">{r['genre_precision']:.2%}</td>
  <td style="background:{_metric_color(r['overlap'], ov_lo, ov_hi)}">{r['overlap']:.2%}</td>
  <td style="background:{_metric_color(r['spearman_rho'], rho_lo, rho_hi)}">{r['spearman_rho']:.4f}</td>
  <td style="background:{_metric_color(r['avg_displacement'], disp_lo, disp_hi, invert=True)}">{r['avg_displacement']:.2f}</td>
  <td style="background:{_metric_color(r['sigma'], sig_lo, sig_hi, invert=True)}">{r['sigma']:.4f}</td>
</tr>
"""

        # ---------------------------------------------------------------
        # Section 3: Rank comparison tables
        # ---------------------------------------------------------------
        # Find best/worst combo by avg_similarity
        best_combo = combo_rows[0]["combo"] if combo_rows else baseline_label
        worst_combo = combo_rows[-1]["combo"] if combo_rows else baseline_label

        rank_sections_html = ""
        for tm in test_movies:
            query_title = tm["title"]
            query_year = tm.get("year", "")

            # Get results for baseline, best, worst
            def _get_top_list(combo_label):
                combo_data = all_results.get(combo_label, {})
                items = combo_data.get(query_title, [])
                result_list = []
                for mid, sim in items[:20]:
                    m = train_data.get(mid, {})
                    cl = cluster_map.get(mid, -1)
                    result_list.append({
                        "id": mid,
                        "title": m.get("title", "?"),
                        "sim": round(sim, 4),
                        "cluster": cl,
                    })
                return result_list

            baseline_list = _get_top_list(baseline_label)
            best_list = _get_top_list(best_combo)
            worst_list = _get_top_list(worst_combo)

            baseline_id_set = {item["id"] for item in baseline_list}
            baseline_rank_map = {item["id"]: i for i, item in enumerate(baseline_list)}

            def _build_rank_table(title_label, items, compare_to_baseline=False):
                html = f'<div class="rank-table-wrap"><h4>{title_label}</h4>'
                html += '<table class="rank-table"><thead><tr>'
                html += '<th>#</th><th>영화 제목</th><th>유사도</th><th>클러스터</th>'
                if compare_to_baseline:
                    html += '<th>변동</th>'
                html += '</tr></thead><tbody>'
                for i, item in enumerate(items):
                    cl = item["cluster"]
                    cl_color = _CLUSTER_COLORS[cl % len(_CLUSTER_COLORS)] if cl >= 0 else "#ccc"
                    row_bg = f"rgba({int(cl_color[1:3],16)},{int(cl_color[3:5],16)},{int(cl_color[5:7],16)},0.12)"

                    change_cell = ""
                    if compare_to_baseline:
                        mid = item["id"]
                        if mid not in baseline_id_set:
                            change_cell = '<td class="change-new">NEW</td>'
                        else:
                            old_rank = baseline_rank_map.get(mid, 99)
                            diff = old_rank - i  # positive = moved up
                            if diff > 0:
                                change_cell = f'<td class="change-up">{chr(9650)}{diff}</td>'
                            elif diff < 0:
                                change_cell = f'<td class="change-down">{chr(9660)}{abs(diff)}</td>'
                            else:
                                change_cell = '<td class="change-same">-</td>'

                    html += f'<tr style="background:{row_bg}">'
                    html += f'<td>{i+1}</td>'
                    html += f'<td class="movie-title">{item["title"]}</td>'
                    html += f'<td>{item["sim"]:.4f}</td>'
                    html += f'<td><span class="cluster-badge" style="background:{cl_color};color:#fff">C{cl}</span></td>'
                    html += change_cell
                    html += '</tr>'
                html += '</tbody></table></div>'
                return html

            rank_sections_html += f'<div class="rank-section"><h3>{query_title} ({query_year})</h3>'
            rank_sections_html += '<div class="rank-tables-row">'
            rank_sections_html += _build_rank_table(
                f"기준선: {baseline_label}", baseline_list, compare_to_baseline=False)
            rank_sections_html += _build_rank_table(
                f"최고 유사도: {best_combo}", best_list, compare_to_baseline=True)
            rank_sections_html += _build_rank_table(
                f"최저 유사도: {worst_combo}", worst_list, compare_to_baseline=True)
            rank_sections_html += '</div></div>'

        # ---------------------------------------------------------------
        # Assemble full HTML
        # ---------------------------------------------------------------
        chart_json = _json.dumps(chart_data, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>민감도 분석 -- KMDB 영화 추천 시스템</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }}
h1 {{ text-align: center; margin: 20px 0 10px; font-size: 28px; }}
h2 {{ margin: 30px 0 15px; padding-bottom: 8px; border-bottom: 3px solid #3498db; font-size: 22px; }}
h3 {{ margin: 20px 0 10px; font-size: 18px; color: #2c3e50; }}
h4 {{ margin: 0 0 8px; font-size: 14px; color: #555; text-align: center; }}
.subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 14px; }}
.section {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
#chart-container {{ width: 100%; height: 700px; }}

/* Detail table */
.detail-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.detail-table th {{ background: #2c3e50; color: #fff; padding: 10px 8px; text-align: center; position: sticky; top: 0; }}
.detail-table td {{ padding: 8px 6px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
.detail-table tr:hover {{ background: rgba(52,152,219,0.08) !important; }}
.detail-table tr.baseline {{ background: rgba(243,156,18,0.15) !important; }}
.detail-table tr.baseline:hover {{ background: rgba(243,156,18,0.25) !important; }}
.combo-cell {{ text-align: left !important; font-family: monospace; font-size: 12px; white-space: nowrap; }}

/* Legend */
.legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px; padding: 12px; background: #ecf0f1; border-radius: 6px; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 20px; height: 14px; border-radius: 3px; border: 1px solid #bbb; }}

/* Cluster info table */
.ci-wrap {{ margin-bottom: 20px; }}
.ci-wrap h4 {{ margin: 0 0 8px; font-size: 15px; color: #2c3e50; }}
.ci-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }}
.ci-table th {{ background: #2c3e50; color: #fff; padding: 8px 6px; text-align: center; }}
.ci-table td {{ padding: 6px 5px; border-bottom: 1px solid #e0e0e0; text-align: center; }}
.ci-table tr:hover {{ background: rgba(52,152,219,0.06); }}
.ci-genres {{ text-align: left !important; }}
.ci-kw {{ text-align: left !important; font-size: 12px; color: #555; }}
.ci-genre {{ display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 11px; font-weight: bold; margin: 1px 2px; }}

/* Rank comparison */
.rank-section {{ margin-bottom: 30px; }}
.rank-tables-row {{ display: flex; gap: 15px; overflow-x: auto; }}
.rank-table-wrap {{ flex: 1; min-width: 320px; }}
.rank-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.rank-table th {{ background: #34495e; color: #fff; padding: 7px 5px; text-align: center; }}
.rank-table td {{ padding: 5px 4px; border-bottom: 1px solid #ddd; text-align: center; }}
.rank-table .movie-title {{ text-align: left; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.cluster-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
.change-up {{ color: #27ae60; font-weight: bold; }}
.change-down {{ color: #e74c3c; font-weight: bold; }}
.change-new {{ color: #8e44ad; font-weight: bold; }}
.change-same {{ color: #95a5a6; }}
</style>
</head>
<body>

<h1>민감도 분석 (81조합)</h1>
<p class="subtitle">가중치 조합 순서: (장르, 키워드, 수치, 텍스트) &mdash; 각 <span style="background:rgba(41,128,185,0.30);padding:2px 6px;border-radius:3px">상(1.5)</span> / <span style="background:rgba(39,174,96,0.25);padding:2px 6px;border-radius:3px">중(1.0)</span> / <span style="background:rgba(230,126,34,0.30);padding:2px 6px;border-radius:3px">하(0.5)</span></p>

<!-- Section 1: Charts -->
<div class="section">
<h2>1. 주요 지표 차트</h2>
<div id="chart-container"></div>
</div>

<!-- Section 2: Detail table -->
<div class="section">
<h2>2. 81조합 상세 테이블</h2>
<div style="overflow-x:auto;">
<table class="detail-table">
<thead>
<tr>
  <th>#</th><th>조합<br><span style="font-weight:normal;font-size:11px">(장르,키워드,수치,텍스트)</span></th><th>장르</th><th>키워드</th><th>수치</th><th>텍스트</th>
  <th>유사도</th><th>장르정밀도</th><th>겹침률</th><th>순위상관 &rho;</th>
  <th>순위변동</th><th>신뢰도 &sigma;</th>
</tr>
</thead>
<tbody>
{table_rows_html}
</tbody>
</table>
</div>

<div class="legend">
  <strong>범례:</strong>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(41,128,185,0.30)"></div> 상 (1.5) 파랑</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(39,174,96,0.25)"></div> 중 (1.0) 초록</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(230,126,34,0.30)"></div> 하 (0.5) 주황</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(243,156,18,0.15);border-color:#F39C12"></div> 기준선 행</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(80,200,80,0.25)"></div> 지표: 좋음</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(255,200,60,0.25)"></div> 지표: 보통</div>
  <div class="legend-item"><div class="legend-swatch" style="background:rgba(230,80,60,0.25)"></div> 지표: 나쁨</div>
</div>
</div>

<!-- Section 3: Rank comparison -->
<div class="section">
<h2>3. 순위 비교 (기준선 vs 최고/최저 유사도 조합)</h2>
{cluster_info_html}
{rank_sections_html}
</div>

<script>
(function() {{
  var data = {chart_json};
  var labels = data.labels;
  var colors = data.bar_colors;

  // Chart 1: avg similarity
  var trace1 = {{
    x: labels, y: data.avg_sims, type: 'bar',
    marker: {{ color: colors }},
    text: data.avg_sims.map(function(v){{ return v.toFixed(4); }}),
    textposition: 'outside', textfont: {{ size: 9 }},
  }};

  // Chart 2: overlap rate
  var trace2 = {{
    x: labels, y: data.overlaps.map(function(v){{ return v*100; }}), type: 'bar',
    marker: {{ color: colors }},
    text: data.overlaps.map(function(v){{ return (v*100).toFixed(1)+'%'; }}),
    textposition: 'outside', textfont: {{ size: 9 }},
  }};

  // Chart 3: spearman rho
  var trace3 = {{
    x: labels, y: data.rhos, type: 'bar',
    marker: {{ color: colors }},
    text: data.rhos.map(function(v){{ return v.toFixed(4); }}),
    textposition: 'outside', textfont: {{ size: 9 }},
  }};

  // Chart 4: sigma
  var trace4 = {{
    x: labels, y: data.sigmas, type: 'bar',
    marker: {{ color: colors }},
    text: data.sigmas.map(function(v){{ return v.toFixed(4); }}),
    textposition: 'outside', textfont: {{ size: 9 }},
  }};

  var layout = {{
    grid: {{ rows: 2, columns: 2, pattern: 'independent' }},
    annotations: [
      {{ text: '<b>평균 유사도</b>', xref: 'x domain', yref: 'y domain', x: 0.5, y: 1.12, showarrow: false, font: {{ size: 15 }}, xanchor: 'center' }},
      {{ text: '<b>Top-20 겹침률 (%)</b>', xref: 'x2 domain', yref: 'y2 domain', x: 0.5, y: 1.12, showarrow: false, font: {{ size: 15 }}, xanchor: 'center' }},
      {{ text: '<b>Spearman 순위상관계수</b>', xref: 'x3 domain', yref: 'y3 domain', x: 0.5, y: 1.12, showarrow: false, font: {{ size: 15 }}, xanchor: 'center' }},
      {{ text: '<b>신뢰도 &sigma; (낮을수록 안정)</b>', xref: 'x4 domain', yref: 'y4 domain', x: 0.5, y: 1.12, showarrow: false, font: {{ size: 15 }}, xanchor: 'center' }},
    ],
    showlegend: false,
    height: 700,
    margin: {{ t: 60, b: 120, l: 60, r: 30 }},
    xaxis:  {{ tickangle: -60, tickfont: {{ size: 8 }}, anchor: 'y' }},
    xaxis2: {{ tickangle: -60, tickfont: {{ size: 8 }}, anchor: 'y2' }},
    xaxis3: {{ tickangle: -60, tickfont: {{ size: 8 }}, anchor: 'y3' }},
    xaxis4: {{ tickangle: -60, tickfont: {{ size: 8 }}, anchor: 'y4' }},
    yaxis:  {{ title: '유사도', anchor: 'x' }},
    yaxis2: {{ title: '겹침률 (%)', anchor: 'x2' }},
    yaxis3: {{ title: 'Spearman ρ', anchor: 'x3' }},
    yaxis4: {{ title: 'σ', anchor: 'x4' }},
  }};

  // Assign traces to subplots
  trace1.xaxis = 'x'; trace1.yaxis = 'y';
  trace2.xaxis = 'x2'; trace2.yaxis = 'y2';
  trace3.xaxis = 'x3'; trace3.yaxis = 'y3';
  trace4.xaxis = 'x4'; trace4.yaxis = 'y4';

  Plotly.newPlot('chart-container', [trace1, trace2, trace3, trace4], layout, {{ responsive: true }});
}})();
</script>

</body>
</html>"""

        # Write file
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "sensitivity_analysis.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[visualizer] 민감도 분석 HTML 저장: {out_path}")
