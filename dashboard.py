"""
dashboard.py -- Plotly Dash 인터랙티브 파라미터 대시보드
====================================================================
브라우저에서 파라미터를 실시간 조정하고 추천 결과를 확인한다.

[레이아웃]
  좌측: 파라미터 패널 (슬라이더 + 직접 입력)
  우측: 결과 탭 (3D/2D 시각화, 추천 테이블, 평가 차트, 스윕 비교)

[실행]
    python main.py --dashboard
    또는
    python dashboard.py
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

try:
    from dash import Dash, html, dcc, callback_context, dash_table
    from dash.dependencies import Input, Output, State
    HAS_DASH = True
except ImportError:
    HAS_DASH = False


def create_dashboard():
    """Dash 앱 생성 및 반환"""
    if not HAS_DASH:
        print("[오류] dash 패키지가 설치되어 있지 않습니다.")
        print("  설치: pip install dash")
        raise ImportError("dash 패키지 필요")

    app = Dash(__name__)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 레이아웃
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    app.layout = html.Div([
        # 제목
        html.H2("영화 추천 AI - 파라미터 대시보드",
                 style={"textAlign": "center", "padding": "10px",
                        "backgroundColor": "#2C3E50", "color": "white",
                        "margin": "0"}),

        html.Div([
            # ── 좌측: 파라미터 패널 ──
            html.Div([
                html.H4("파라미터 설정", style={"marginBottom": "15px"}),

                # 특징 가중치
                html.Label("장르 가중치", style={"fontWeight": "bold"}),
                html.Div([
                    dcc.Slider(id="slider-wg", min=0, max=3, step=0.1,
                               value=config.WEIGHT_GENRE,
                               marks={i: str(i) for i in range(4)},
                               tooltip={"placement": "bottom"}),
                    dcc.Input(id="input-wg", type="number", value=config.WEIGHT_GENRE,
                              step=0.1, min=0, max=3,
                              style={"width": "70px", "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                html.Label("키워드 가중치", style={"fontWeight": "bold", "marginTop": "10px"}),
                html.Div([
                    dcc.Slider(id="slider-wk", min=0, max=3, step=0.1,
                               value=config.WEIGHT_KEYWORD,
                               marks={i: str(i) for i in range(4)},
                               tooltip={"placement": "bottom"}),
                    dcc.Input(id="input-wk", type="number", value=config.WEIGHT_KEYWORD,
                              step=0.1, min=0, max=3,
                              style={"width": "70px", "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                html.Label("수치 가중치", style={"fontWeight": "bold", "marginTop": "10px"}),
                html.Div([
                    dcc.Slider(id="slider-wn", min=0, max=3, step=0.1,
                               value=config.WEIGHT_NUMERIC,
                               marks={i: str(i) for i in range(4)},
                               tooltip={"placement": "bottom"}),
                    dcc.Input(id="input-wn", type="number", value=config.WEIGHT_NUMERIC,
                              step=0.1, min=0, max=3,
                              style={"width": "70px", "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                html.Hr(),

                # 군집화
                html.Label("군집화 방법", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="dd-cluster", options=[
                    {"label": "KMeans", "value": "kmeans"},
                    {"label": "DBSCAN", "value": "dbscan"},
                ], value=config.CLUSTER_METHOD, clearable=False),

                html.Label("KMeans 클러스터 수", style={"fontWeight": "bold", "marginTop": "10px"}),
                html.Div([
                    dcc.Slider(id="slider-k", min=2, max=20, step=1,
                               value=config.KMEANS_N_CLUSTERS,
                               marks={i: str(i) for i in [2, 5, 8, 10, 15, 20]},
                               tooltip={"placement": "bottom"}),
                    dcc.Input(id="input-k", type="number",
                              value=config.KMEANS_N_CLUSTERS,
                              step=1, min=2, max=20,
                              style={"width": "70px", "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                html.Hr(),

                # 차원 축소
                html.Label("차원 축소", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="dd-reduction", options=[
                    {"label": "PCA (3D)", "value": "pca3"},
                    {"label": "PCA (2D)", "value": "pca2"},
                    {"label": "t-SNE (2D)", "value": "tsne2"},
                ], value="pca3", clearable=False),

                html.Hr(),

                # 추천
                html.Label("Top-K 추천 수", style={"fontWeight": "bold"}),
                html.Div([
                    dcc.Slider(id="slider-topk", min=1, max=20, step=1,
                               value=config.TOP_K,
                               marks={i: str(i) for i in [1, 3, 5, 10, 15, 20]},
                               tooltip={"placement": "bottom"}),
                    dcc.Input(id="input-topk", type="number", value=config.TOP_K,
                              step=1, min=1, max=20,
                              style={"width": "70px", "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                html.Hr(),

                # 실행 버튼
                html.Button("파이프라인 실행", id="btn-run",
                             style={"width": "100%", "padding": "12px",
                                    "fontSize": "16px", "fontWeight": "bold",
                                    "backgroundColor": "#27AE60", "color": "white",
                                    "border": "none", "borderRadius": "5px",
                                    "cursor": "pointer", "marginTop": "10px"}),

                # 상태 표시
                html.Div(id="status-text",
                          style={"marginTop": "10px", "color": "#666",
                                 "fontSize": "13px"}),

            ], style={"width": "300px", "padding": "15px",
                       "backgroundColor": "#F8F9FA", "borderRight": "2px solid #DDD",
                       "overflowY": "auto", "height": "calc(100vh - 60px)"}),

            # ── 우측: 결과 영역 ──
            html.Div([
                dcc.Tabs(id="result-tabs", value="tab-viz", children=[
                    dcc.Tab(label="시각화", value="tab-viz"),
                    dcc.Tab(label="추천 결과", value="tab-rec"),
                    dcc.Tab(label="평가", value="tab-eval"),
                    dcc.Tab(label="스윕 비교", value="tab-sweep"),
                    dcc.Tab(label="민감도 분석", value="tab-sensitivity"),
                ]),
                html.Div(id="tab-content",
                          style={"padding": "10px", "overflowY": "auto"}),
            ], style={"flex": "1", "height": "calc(100vh - 60px)",
                       "overflowY": "auto"}),

        ], style={"display": "flex", "height": "calc(100vh - 60px)"}),

        # 히든 저장소: 파이프라인 결과
        dcc.Store(id="store-result"),
        dcc.Store(id="store-sweep", data=[]),
        dcc.Store(id="store-sensitivity", data=None),

    ], style={"fontFamily": "sans-serif", "margin": "0", "padding": "0"})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 콜백: 슬라이더 <-> 입력 동기화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _sync_callback(slider_id, input_id):
        @app.callback(
            Output(slider_id, "value"),
            Output(input_id, "value"),
            Input(slider_id, "value"),
            Input(input_id, "value"),
        )
        def sync(slider_val, input_val):
            ctx = callback_context
            if not ctx.triggered:
                return slider_val, slider_val
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == slider_id:
                return slider_val, slider_val
            else:
                return input_val, input_val

    _sync_callback("slider-wg", "input-wg")
    _sync_callback("slider-wk", "input-wk")
    _sync_callback("slider-wn", "input-wn")
    _sync_callback("slider-k", "input-k")
    _sync_callback("slider-topk", "input-topk")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 콜백: 파이프라인 실행
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @app.callback(
        Output("store-result", "data"),
        Output("store-sweep", "data"),
        Output("status-text", "children"),
        Input("btn-run", "n_clicks"),
        State("slider-wg", "value"),
        State("slider-wk", "value"),
        State("slider-wn", "value"),
        State("dd-cluster", "value"),
        State("slider-k", "value"),
        State("dd-reduction", "value"),
        State("slider-topk", "value"),
        State("store-sweep", "data"),
        prevent_initial_call=True,
    )
    def run_pipeline_callback(n_clicks, wg, wk, wn, cluster_method,
                               n_clusters, reduction, topk, sweep_history):
        from main import run_pipeline

        # 파라미터 구성
        use_tsne = "tsne" in reduction
        n_comp = 2 if reduction.endswith("2") else 3

        params = {
            "weight_genre": wg,
            "weight_keyword": wk,
            "weight_numeric": wn,
            "cluster_method": cluster_method,
            "n_clusters": n_clusters,
            "use_tsne": use_tsne,
            "pca_components": n_comp,
            "top_k": topk,
        }

        try:
            result = run_pipeline(params)

            # 직렬화 가능한 형태로 변환
            store_data = {
                "params": params,
                "quant": result["quant"],
                "qual": result["qual"],
                "comparison": result["comparison"],
                "recommendations": {},
                "coords": {},
                "clusters": result["clusters"].tolist(),
                "train_titles": [m["title"] for m in result["train_movies"]],
                "test_titles": [m["title"] for m in result["test_movies"]],
                "train_genres": {m["title"]: m["genres"] for m in result["train_movies"]},
                "test_genres": {m["title"]: m["genres"] for m in result["test_movies"]},
                "train_years": {m["title"]: m["year"] for m in result["train_movies"]},
            }

            # 좌표 직렬화
            for title, coord in result["coords"].items():
                store_data["coords"][title] = coord.tolist()

            # 추천 결과 직렬화
            for qt, recs in result["recommendations"].items():
                store_data["recommendations"][qt] = recs

            # 스윕 이력 추가
            overall = result["quant"].get("overall", {})
            sweep_entry = {"params": params, "metrics": overall}
            if sweep_history is None:
                sweep_history = []
            sweep_history.append(sweep_entry)

            status = (f"실행 완료 (유사도: {overall.get('avg_similarity', 0):.4f}, "
                      f"장르정밀도: {overall.get('avg_genre_precision', 0):.4f})")
            return store_data, sweep_history, status

        except Exception as e:
            return None, sweep_history or [], f"오류: {str(e)}"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 콜백: 탭 컨텐츠 렌더링
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @app.callback(
        Output("tab-content", "children"),
        Input("result-tabs", "value"),
        Input("store-result", "data"),
        Input("store-sweep", "data"),
        Input("store-sensitivity", "data"),
    )
    def render_tab(tab, result_data, sweep_data, sensitivity_data):
        if tab == "tab-sensitivity":
            return _render_sensitivity_tab(sensitivity_data)

        if result_data is None:
            return html.Div("파라미터를 설정하고 '파이프라인 실행' 버튼을 클릭하세요.",
                           style={"padding": "40px", "textAlign": "center",
                                  "color": "#999", "fontSize": "16px"})

        if tab == "tab-viz":
            return _render_viz_tab(result_data)
        elif tab == "tab-rec":
            return _render_rec_tab(result_data)
        elif tab == "tab-eval":
            return _render_eval_tab(result_data)
        elif tab == "tab-sweep":
            return _render_sweep_tab(sweep_data)

        return html.Div()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 콜백: 민감도 분석 실행
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @app.callback(
        Output("store-sensitivity", "data"),
        Output("sensitivity-status", "children"),
        Input("btn-sensitivity", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_sensitivity_callback(n_clicks):
        if not n_clicks:
            return None, ""
        try:
            from sensitivity import SensitivityAnalyzer
            from visualizer import create_sensitivity_charts

            analyzer = SensitivityAnalyzer()
            results = analyzer.run()
            create_sensitivity_charts(results)

            # 직렬화 가능한 데이터로 변환
            serializable = _serialize_sensitivity(results)
            return serializable, f"완료! {len(config.SENSITIVITY_SWEEP_VALUES)}개 값 × 3개 파라미터 분석"
        except Exception as e:
            return None, f"오류: {str(e)}"

    return app


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 렌더링 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _render_viz_tab(data):
    """시각화 탭"""
    coords = data.get("coords", {})
    clusters = data.get("clusters", [])
    train_titles = data.get("train_titles", [])
    train_genres = data.get("train_genres", {})
    train_years = data.get("train_years", {})
    test_titles = data.get("test_titles", [])
    test_genres = data.get("test_genres", {})

    if not coords:
        return html.Div("좌표 데이터가 없습니다.")

    sample = list(coords.values())[0]
    n_dim = len(sample)

    _COLORS = [
        "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
        "#8E44AD", "#E67E22", "#1ABC9C", "#34495E",
        "#00BCD4", "#795548", "#607D8B", "#FF5722",
    ]

    if n_dim == 3:
        fig = go.Figure()

        # 클러스터별 학습 영화
        cluster_groups = {}
        for i, title in enumerate(train_titles):
            if i < len(clusters):
                label = clusters[i]
            else:
                label = 0
            if label not in cluster_groups:
                cluster_groups[label] = {"x": [], "y": [], "z": [], "text": []}
            coord = coords.get(title)
            if coord:
                cluster_groups[label]["x"].append(coord[0])
                cluster_groups[label]["y"].append(coord[1])
                cluster_groups[label]["z"].append(coord[2])
                genres = train_genres.get(title, [])
                year = train_years.get(title, "")
                cluster_groups[label]["text"].append(
                    f"<b>{title}</b> ({year})<br>"
                    f"장르: {', '.join(genres)}<br>클러스터: {label}"
                )

        marker_size = 3 if len(train_titles) > 5000 else (4 if len(train_titles) > 1000 else 6)
        for label, gdata in sorted(cluster_groups.items()):
            color = _COLORS[label % len(_COLORS)] if label >= 0 else "#999"
            fig.add_trace(go.Scatter3d(
                x=gdata["x"], y=gdata["y"], z=gdata["z"],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7),
                text=gdata["text"], hoverinfo="text",
                name=f"Cluster {label}",
            ))

        # 테스트 영화
        for title in test_titles:
            coord = coords.get(title)
            if coord:
                fig.add_trace(go.Scatter3d(
                    x=[coord[0]], y=[coord[1]], z=[coord[2]],
                    mode="markers+text",
                    marker=dict(size=12, color="#FFD700", symbol="diamond",
                                line=dict(width=2, color="black")),
                    text=[title], textposition="top center",
                    name=f"[NEW] {title}",
                ))

        fig.update_layout(
            height=700, margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgb(240,240,240)"),
                yaxis=dict(backgroundcolor="rgb(230,230,250)"),
                zaxis=dict(backgroundcolor="rgb(245,245,220)"),
            ),
        )
    else:
        # 2D
        fig = go.Figure()
        cluster_groups = {}
        for i, title in enumerate(train_titles):
            label = clusters[i] if i < len(clusters) else 0
            if label not in cluster_groups:
                cluster_groups[label] = {"x": [], "y": [], "text": []}
            coord = coords.get(title)
            if coord:
                cluster_groups[label]["x"].append(coord[0])
                cluster_groups[label]["y"].append(coord[1])
                genres = train_genres.get(title, [])
                cluster_groups[label]["text"].append(
                    f"<b>{title}</b><br>장르: {', '.join(genres)}"
                )

        marker_size = 4 if len(train_titles) > 1000 else 6
        for label, gdata in sorted(cluster_groups.items()):
            color = _COLORS[label % len(_COLORS)] if label >= 0 else "#999"
            fig.add_trace(go.Scatter(
                x=gdata["x"], y=gdata["y"], mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.7),
                text=gdata["text"], hoverinfo="text",
                name=f"Cluster {label}",
            ))

        for title in test_titles:
            coord = coords.get(title)
            if coord:
                fig.add_trace(go.Scatter(
                    x=[coord[0]], y=[coord[1]],
                    mode="markers+text",
                    marker=dict(size=12, color="#FFD700", symbol="diamond",
                                line=dict(width=2, color="black")),
                    text=[title], textposition="top center",
                    name=f"[NEW] {title}",
                ))

        fig.update_layout(height=600)

    return dcc.Graph(figure=fig)


def _render_rec_tab(data):
    """추천 결과 탭"""
    recommendations = data.get("recommendations", {})
    if not recommendations:
        return html.Div("추천 결과가 없습니다.")

    tables = []
    for query_title, recs in recommendations.items():
        tables.append(html.H4(f"입력: {query_title}",
                               style={"marginTop": "20px"}))

        rows = []
        for r in recs:
            rows.append({
                "순위": r["rank"],
                "제목": r["title"],
                "연도": r["year"],
                "유사도": f"{r['similarity']:.4f}",
                "공통장르": ", ".join(r.get("shared_genres", [])),
                "이유": r.get("explanation", ""),
            })

        tables.append(dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "순위", "id": "순위"},
                {"name": "제목", "id": "제목"},
                {"name": "연도", "id": "연도"},
                {"name": "유사도", "id": "유사도"},
                {"name": "공통장르", "id": "공통장르"},
                {"name": "이유", "id": "이유"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px",
                         "fontSize": "13px"},
            style_header={"backgroundColor": "#2C3E50", "color": "white",
                           "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": "#F8F9FA"},
            ],
        ))

    return html.Div(tables)


def _render_eval_tab(data):
    """평가 탭"""
    quant = data.get("quant", {})
    comparison = data.get("comparison", {})

    if not quant:
        return html.Div("평가 결과가 없습니다.")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["영화별 유사도", "적합/부적합 현황"],
    )

    per_query = quant.get("per_query", {})
    if per_query:
        titles = list(per_query.keys())
        short = [t[:12] + "..." if len(t) > 15 else t for t in titles]
        sims = [per_query[t]["avg_similarity"] for t in titles]
        gps = [per_query[t]["genre_precision"] for t in titles]

        fig.add_trace(go.Bar(x=short, y=sims, name="유사도",
                             marker_color="#3498DB"), row=1, col=1)
        fig.add_trace(go.Bar(x=short, y=gps, name="장르정밀도",
                             marker_color="#2ECC71"), row=1, col=1)

    adequacy = comparison.get("adequacy", {})
    if adequacy:
        adeq = sum(1 for a in adequacy.values() if a["adequate"])
        inadeq = len(adequacy) - adeq
        fig.add_trace(go.Bar(
            x=["적합", "부적합"], y=[adeq, inadeq],
            marker_color=["#2ECC71", "#E74C3C"],
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(height=400, barmode="group")

    # 적합/부적합 상세
    detail_rows = []
    for title, result in adequacy.items():
        status = "적합" if result["adequate"] else "부적합"
        reasons = "; ".join(result["reasons"]) if result["reasons"] else "-"
        detail_rows.append({"영화": title, "판정": status, "사유": reasons})

    return html.Div([
        dcc.Graph(figure=fig),
        html.H4("적합/부적합 상세", style={"marginTop": "20px"}),
        dash_table.DataTable(
            data=detail_rows,
            columns=[
                {"name": "영화", "id": "영화"},
                {"name": "판정", "id": "판정"},
                {"name": "사유", "id": "사유"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px",
                         "fontSize": "13px"},
            style_header={"backgroundColor": "#2C3E50", "color": "white"},
            style_data_conditional=[
                {"if": {"filter_query": '{판정} = "부적합"'},
                 "backgroundColor": "#FDEDEC"},
            ],
        ),
    ])


def _render_sweep_tab(sweep_data):
    """스윕 비교 탭"""
    if not sweep_data:
        return html.Div("파이프라인을 여러 번 실행하면 결과를 비교할 수 있습니다.",
                        style={"padding": "40px", "textAlign": "center",
                               "color": "#999"})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["실행별 유사도 추이", "실행별 정밀도 추이"],
    )

    run_ids = list(range(1, len(sweep_data) + 1))
    sims = [r["metrics"].get("avg_similarity", 0) for r in sweep_data]
    gps = [r["metrics"].get("avg_genre_precision", 0) for r in sweep_data]

    hover_texts = []
    for r in sweep_data:
        p = r["params"]
        lines = [f"{k}: {v}" for k, v in p.items()]
        hover_texts.append("<br>".join(lines))

    fig.add_trace(go.Scatter(
        x=run_ids, y=sims, mode="lines+markers",
        name="평균 유사도", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#3498DB", size=10),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=run_ids, y=gps, mode="lines+markers",
        name="장르 정밀도", text=hover_texts, hoverinfo="text+y",
        marker=dict(color="#2ECC71", size=10),
    ), row=1, col=2)

    fig.update_layout(height=400)
    fig.update_xaxes(title_text="실행 번호", dtick=1)

    # 실행 이력 테이블
    history_rows = []
    for i, r in enumerate(sweep_data):
        p = r["params"]
        m = r["metrics"]
        history_rows.append({
            "#": i + 1,
            "장르가중": p.get("weight_genre", "-"),
            "키워드가중": p.get("weight_keyword", "-"),
            "수치가중": p.get("weight_numeric", "-"),
            "클러스터": p.get("n_clusters", "-"),
            "Top-K": p.get("top_k", "-"),
            "유사도": f"{m.get('avg_similarity', 0):.4f}",
            "장르정밀도": f"{m.get('avg_genre_precision', 0):.4f}",
        })

    return html.Div([
        dcc.Graph(figure=fig),
        html.H4("실행 이력", style={"marginTop": "20px"}),
        dash_table.DataTable(
            data=history_rows,
            columns=[
                {"name": "#", "id": "#"},
                {"name": "장르가중", "id": "장르가중"},
                {"name": "키워드가중", "id": "키워드가중"},
                {"name": "수치가중", "id": "수치가중"},
                {"name": "클러스터", "id": "클러스터"},
                {"name": "Top-K", "id": "Top-K"},
                {"name": "유사도", "id": "유사도"},
                {"name": "장르정밀도", "id": "장르정밀도"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px",
                         "fontSize": "13px"},
            style_header={"backgroundColor": "#2C3E50", "color": "white"},
        ),
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 민감도 분석 탭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _render_sensitivity_tab(sensitivity_data):
    """민감도 분석 탭 렌더링"""
    from dash import html, dcc, dash_table

    # 실행 버튼 (항상 표시)
    controls = html.Div([
        html.Button(
            "민감도 분석 실행 (3×7 파라미터 스윕)",
            id="btn-sensitivity",
            style={
                "padding": "12px 24px", "fontSize": "15px",
                "fontWeight": "bold", "backgroundColor": "#8E44AD",
                "color": "white", "border": "none", "borderRadius": "5px",
                "cursor": "pointer", "marginBottom": "10px",
            },
        ),
        html.Div(id="sensitivity-status",
                  style={"color": "#666", "fontSize": "13px",
                         "marginBottom": "15px"}),
        html.P("장르·키워드·수치 가중치를 0.0~3.0으로 변화시키며 "
               "기준선(1.0, 1.0, 1.0) 대비 추천 목록의 변화를 분석합니다.",
               style={"color": "#888", "fontSize": "12px"}),
    ])

    if sensitivity_data is None:
        return html.Div([
            controls,
            html.Div("'민감도 분석 실행' 버튼을 클릭하세요.",
                     style={"padding": "40px", "textAlign": "center",
                            "color": "#999", "fontSize": "16px"}),
        ])

    analysis = sensitivity_data.get("analysis", [])
    if not analysis:
        return html.Div([controls, html.Div("분석 결과가 없습니다.")])

    # 유사도 내림차순 정렬
    sorted_data = sorted(analysis, key=lambda x: -x["avg_similarity"])
    labels = [d["label"] for d in sorted_data]
    colors = ["#F39C12" if d["is_baseline"] else "#3498DB" for d in sorted_data]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[d["avg_similarity"] for d in sorted_data],
        marker_color=colors,
        text=[f"{d['avg_similarity']:.3f}" for d in sorted_data],
        textposition="outside", textfont=dict(size=8),
    ))
    fig.update_layout(
        title="27조합 평균 유사도 (주황=기준선)",
        height=400, xaxis_tickangle=45,
    )

    # 요약 테이블
    summary_rows = []
    for d in sorted_data:
        summary_rows.append({
            "조합": d["label"],
            "장르": d["weight_genre"],
            "키워드": d["weight_keyword"],
            "수치": d["weight_numeric"],
            "겹침률": f"{d['overlap_ratio']:.1%}",
            "순위상관": f"{d['spearman_rho']:.4f}",
            "유사도": f"{d['avg_similarity']:.4f}",
            "장르정밀도": f"{d['genre_precision']:.4f}",
            "신뢰도σ": f"{d['reliability_sim']:.4f}",
        })

    return html.Div([
        controls,
        dcc.Graph(figure=fig),
        html.H4("27조합 상세 결과", style={"marginTop": "20px"}),
        dash_table.DataTable(
            data=summary_rows,
            columns=[
                {"name": c, "id": c}
                for c in ["조합", "장르", "키워드", "수치", "겹침률",
                          "순위상관", "유사도", "장르정밀도", "신뢰도σ"]
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "6px",
                         "fontSize": "12px"},
            style_header={"backgroundColor": "#2C3E50", "color": "white",
                           "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": '{조합} = "(중,중,중)"'},
                 "backgroundColor": "#FFF3CD", "fontWeight": "bold"},
            ],
        ),
    ])


def _serialize_sensitivity(results: dict) -> dict:
    """민감도 분석 결과를 JSON 직렬화 가능한 형태로 변환 (27조합)."""
    analysis = [
        {
            "label": d["label"],
            "weight_genre": float(d["weight_genre"]),
            "weight_keyword": float(d["weight_keyword"]),
            "weight_numeric": float(d["weight_numeric"]),
            "overlap_ratio": float(d["overlap_ratio"]),
            "spearman_rho": float(d["spearman_rho"]),
            "avg_rank_displacement": float(d["avg_rank_displacement"]),
            "avg_similarity": float(d["avg_similarity"]),
            "genre_precision": float(d["genre_precision"]),
            "reliability_sim": float(d["reliability_sim"]),
            "is_baseline": d["is_baseline"],
        }
        for d in results["analysis"]
    ]
    return {"analysis": analysis}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 직접 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    app = create_dashboard()
    print(f"대시보드 실행: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=True)
