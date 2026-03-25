"""
dashboard.py -- Dash 기반 영화 추천 시스템 대시보드
====================================================================
5개 탭으로 구성된 대화형 대시보드:
  탭 1: 검색 및 추천
  탭 2: 클러스터 시각화
  탭 3: 파라미터 제어
  탭 4: 평가
  탭 5: 민감도 분석

실행:
    python main.py --dashboard
"""

import os
import io
import base64
import threading
import numpy as np

import dash
from dash import html, dcc, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

_CLUSTER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
    "#9B59B6", "#E67E22", "#1ABC9C", "#34495E",
    "#FF5722", "#00BCD4", "#8BC34A", "#FF9800",
]

# ------------------------------------------------------------------
# 포스터 썸네일 헬퍼
# ------------------------------------------------------------------
def _get_poster_base64(poster_path, width=120, height=170):
    if not poster_path or not os.path.exists(poster_path):
        return None
    if "_noimage" in poster_path.lower():
        return None
    try:
        from PIL import Image
        img = Image.open(poster_path)
        img.thumbnail((width, height))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


# ------------------------------------------------------------------
# 카드 컴포넌트
# ------------------------------------------------------------------
def _make_movie_card(rec, idx):
    poster_src = _get_poster_base64(rec.get("poster_path"))
    poster_el = html.Img(
        src=poster_src,
        style={"width": "100px", "height": "140px", "objectFit": "cover",
               "borderRadius": "4px", "backgroundColor": "#eee"},
    ) if poster_src else html.Div(
        rec.get("title", "")[:6],
        style={"width": "100px", "height": "140px", "backgroundColor": "#ddd",
               "display": "flex", "alignItems": "center", "justifyContent": "center",
               "borderRadius": "4px", "fontSize": "12px", "color": "#666"},
    )

    sim = rec.get("similarity", 0)
    sim_pct = max(0, min(100, sim * 100))
    genres = rec.get("genres", [])

    return html.Div(style={
        "border": "1px solid #e0e0e0", "borderRadius": "8px", "padding": "12px",
        "display": "flex", "gap": "12px", "backgroundColor": "#fff",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }, children=[
        poster_el,
        html.Div(style={"flex": "1", "minWidth": "0"}, children=[
            html.Div(f"#{idx}", style={"fontSize": "11px", "color": "#999"}),
            html.Div(rec.get("title", ""), style={
                "fontWeight": "bold", "fontSize": "14px", "marginBottom": "2px",
                "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
            }),
            html.Div(f"{rec.get('title_eng', '')} ({rec.get('year', '')})",
                      style={"fontSize": "11px", "color": "#777", "marginBottom": "4px"}),
            html.Div(style={"display": "flex", "gap": "4px", "flexWrap": "wrap",
                             "marginBottom": "4px"}, children=[
                html.Span(g, style={
                    "fontSize": "10px", "padding": "1px 6px", "borderRadius": "10px",
                    "backgroundColor": config.GENRE_COLORS.get(g, "#bbb"),
                    "color": "#fff",
                }) for g in genres[:4]
            ]),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                html.Div(style={
                    "flex": "1", "height": "6px", "backgroundColor": "#eee",
                    "borderRadius": "3px", "overflow": "hidden",
                }, children=[
                    html.Div(style={
                        "width": f"{sim_pct}%", "height": "100%",
                        "backgroundColor": "#3498DB", "borderRadius": "3px",
                    }),
                ]),
                html.Span(f"{sim:.4f}", style={"fontSize": "12px", "fontWeight": "bold",
                                                  "color": "#3498DB", "minWidth": "50px"}),
            ]),
            html.Div(rec.get("explanation", ""),
                      style={"fontSize": "10px", "color": "#888", "marginTop": "2px"}),
        ]),
    ])


# ------------------------------------------------------------------
# 앱 생성
# ------------------------------------------------------------------
def create_app():
    from main import run_pipeline
    from search import MovieSearchEngine

    print("[dashboard] 파이프라인 실행 중...")
    result = run_pipeline()
    print("[dashboard] 파이프라인 완료")

    engine = MovieSearchEngine(
        result["embedding"], result["train_movies"], result["test_movies"]
    )

    # 기준선 추천 결과 저장
    baseline_recs = {}
    for tm in result["test_movies"]:
        recs = result["recommendations"].get(tm["title"], [])
        baseline_recs[tm["title"]] = [r["id"] for r in recs[:config.TOP_K]]

    app = dash.Dash(__name__, title="영화 추천 시스템",
                    suppress_callback_exceptions=True)

    # ==================================================================
    # 레이아웃
    # ==================================================================

    header = html.Div(style={
        "backgroundColor": "#2c3e50", "color": "#fff", "padding": "16px 24px",
        "fontSize": "20px", "fontWeight": "bold",
    }, children="KMDB 영화 추천 시스템 -- 코사인 유사도 기반 하이브리드 추천")

    tabs = dcc.Tabs(id="main-tabs", value="tab-search", children=[
        dcc.Tab(label="검색 및 추천", value="tab-search"),
        dcc.Tab(label="클러스터 시각화", value="tab-cluster"),
        dcc.Tab(label="파라미터 제어", value="tab-params"),
        dcc.Tab(label="평가", value="tab-eval"),
        dcc.Tab(label="민감도 분석", value="tab-sensitivity"),
    ])

    app.layout = html.Div(style={"fontFamily": "Malgun Gothic, sans-serif",
                                   "backgroundColor": "#f5f5f5"}, children=[
        header,
        tabs,
        html.Div(id="tab-content", style={"padding": "20px"}),
    ])

    # ==================================================================
    # 탭 1: 검색
    # ==================================================================
    def _tab_search():
        return html.Div([
            html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "20px"}, children=[
                dcc.Input(
                    id="search-input", type="text",
                    placeholder="영화 제목 또는 키워드를 입력하세요...",
                    style={"flex": "1", "padding": "12px", "fontSize": "16px",
                           "border": "2px solid #3498DB", "borderRadius": "6px",
                           "height": "48px", "lineHeight": "24px"},
                ),
                html.Button("검색", id="search-btn", style={
                    "padding": "12px 24px", "backgroundColor": "#3498DB",
                    "color": "#fff", "border": "none", "borderRadius": "6px",
                    "fontSize": "16px", "cursor": "pointer",
                }),
            ]),
            html.Div(id="search-info", style={"marginBottom": "10px", "color": "#666"}),
            html.Div(id="search-results", style={
                "display": "grid", "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "12px",
            }),
        ])

    # ==================================================================
    # 탭 2: 클러스터
    # ==================================================================
    def _tab_cluster():
        coords = result["coords"]
        train_movies = result["train_movies"]
        clusters = result["clusters"]

        fig = go.Figure()
        unique_clusters = sorted(set(clusters))
        for ci, cid in enumerate(unique_clusters):
            mask = clusters == cid
            indices = np.where(mask)[0]
            c_movies = [train_movies[i] for i in indices]
            c_ids = [m["id"] for m in c_movies]
            c_coords = np.array([coords[mid][:3] for mid in c_ids
                                 if mid in coords and len(coords[mid]) >= 3])
            if len(c_coords) == 0:
                continue
            titles = [m["title"] for m in c_movies if m["id"] in coords]
            color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=c_coords[:, 0], y=c_coords[:, 1], z=c_coords[:, 2],
                mode="markers", marker=dict(size=2, color=color, opacity=0.5),
                name=f"클러스터 {cid}", text=titles, hoverinfo="text",
            ))

        for tm in result["test_movies"]:
            mid = tm["id"]
            if mid in coords and len(coords[mid]) >= 3:
                tc = coords[mid]
                fig.add_trace(go.Scatter3d(
                    x=[tc[0]], y=[tc[1]], z=[tc[2]],
                    mode="markers+text",
                    marker=dict(size=8, color="#FF0000", symbol="diamond"),
                    text=[tm["title"]], name=f"테스트: {tm['title']}",
                ))

        fig.update_layout(
            title="3D 임베딩 공간 -- 클러스터 분포",
            height=700,
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        )

        # 클러스터 정보 테이블 (장르별 배경색)
        cluster_info = result.get("cluster_info", {})

        def _genre_badge(name, count):
            bg = config.GENRE_COLORS.get(name, "#BDBDBD")
            # 밝은 배경엔 검정, 어두운 배경엔 흰색
            r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
            text_color = "#fff" if (r * 0.299 + g * 0.587 + b * 0.114) < 150 else "#222"
            return html.Span(
                f"{name}({count})",
                style={
                    "backgroundColor": bg, "color": text_color,
                    "padding": "2px 8px", "borderRadius": "4px",
                    "marginRight": "6px", "fontSize": "13px",
                    "display": "inline-block", "marginBottom": "2px",
                },
            )

        table_header = html.Thead(html.Tr([
            html.Th(c, style={"backgroundColor": "#2c3e50", "color": "#fff",
                               "fontWeight": "bold", "padding": "8px", "textAlign": "left"})
            for c in ["클러스터", "영화 수", "주요 장르", "주요 키워드"]
        ]))

        table_rows = []
        for cname, cdata in cluster_info.items():
            genre_badges = [_genre_badge(g, c) for g, c in cdata.get("top_genres", [])[:3]]
            kw_str = ", ".join(f"{k}({c})" for k, c in cdata.get("top_keywords", [])[:3])
            row_style = {"padding": "8px", "fontSize": "13px", "borderBottom": "1px solid #eee"}
            table_rows.append(html.Tr([
                html.Td(cname, style=row_style),
                html.Td(str(cdata.get("count", 0)), style=row_style),
                html.Td(genre_badges, style=row_style),
                html.Td(kw_str, style=row_style),
            ]))

        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("클러스터 정보", style={"marginTop": "20px"}),
            html.Table([table_header, html.Tbody(table_rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    # ==================================================================
    # 탭 3: 파라미터 제어
    # ==================================================================
    def _tab_params():
        return html.Div([
            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                             "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div([
                    html.Label("장르 가중치", style={"fontWeight": "bold"}),
                    dcc.Slider(id="w-genre", min=0, max=3, step=0.1,
                               value=config.WEIGHT_GENRE,
                               marks={0: "0", 1: "1", 2: "2", 3: "3"}),
                ]),
                html.Div([
                    html.Label("키워드 가중치", style={"fontWeight": "bold"}),
                    dcc.Slider(id="w-keyword", min=0, max=3, step=0.1,
                               value=config.WEIGHT_KEYWORD,
                               marks={0: "0", 1: "1", 2: "2", 3: "3"}),
                ]),
                html.Div([
                    html.Label("수치 가중치", style={"fontWeight": "bold"}),
                    dcc.Slider(id="w-numeric", min=0, max=3, step=0.1,
                               value=config.WEIGHT_NUMERIC,
                               marks={0: "0", 1: "1", 2: "2", 3: "3"}),
                ]),
                html.Div([
                    html.Label("텍스트 가중치", style={"fontWeight": "bold"}),
                    dcc.Slider(id="w-text", min=0, max=3, step=0.1,
                               value=config.WEIGHT_TEXT,
                               marks={0: "0", 1: "1", 2: "2", 3: "3"}),
                ]),
            ]),
            html.Button("가중치 적용", id="apply-weights-btn", style={
                "padding": "10px 24px", "backgroundColor": "#E74C3C",
                "color": "#fff", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer", "marginBottom": "20px",
            }),
            html.Div(id="weight-info", style={"marginBottom": "10px", "color": "#666"}),
            html.Div(id="weight-results"),
        ])

    # ==================================================================
    # 탭 4: 평가
    # ==================================================================
    def _tab_eval():
        quant = result.get("quant", {})
        overall = quant.get("overall", {})
        per_query = quant.get("per_query", {})
        comparison = result.get("comparison", {})

        # 전체 평균 카드
        overall_cards = []
        metric_display = [
            ("평균 유사도", overall.get("avg_similarity", 0)),
            ("장르 정밀도", overall.get("avg_genre_precision", 0)),
            ("키워드 정밀도", overall.get("avg_keyword_precision", 0)),
            ("다양성", overall.get("avg_diversity", 0)),
            ("텍스트 일관성", overall.get("avg_text_coherence", 0)),
        ]
        for label, val in metric_display:
            overall_cards.append(html.Div(style={
                "backgroundColor": "#fff", "borderRadius": "8px", "padding": "16px",
                "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            }, children=[
                html.Div(label, style={"fontSize": "12px", "color": "#888"}),
                html.Div(f"{val:.4f}", style={"fontSize": "24px", "fontWeight": "bold",
                                                "color": "#2c3e50"}),
            ]))

        # 영화별 차트
        if per_query:
            titles = list(per_query.keys())
            short_titles = [t[:12] for t in titles]
            metrics = ["avg_similarity", "genre_precision", "diversity", "text_coherence"]
            labels = ["평균 유사도", "장르 정밀도", "다양성", "텍스트 일관성"]
            colors = ["#3498DB", "#E74C3C", "#F1C40F", "#2ECC71"]

            fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
            for i, (mn, ml) in enumerate(zip(metrics, labels)):
                r, c = (i // 2) + 1, (i % 2) + 1
                vals = [per_query[t].get(mn, 0) for t in titles]
                fig.add_trace(go.Bar(
                    x=short_titles, y=vals, marker_color=colors[i],
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                ), row=r, col=c)
            fig.update_layout(showlegend=False, height=600)
        else:
            fig = go.Figure()

        # 적합/부적합 목록
        adequacy = comparison.get("adequacy", {})
        adequacy_items = []
        for title, adeq in adequacy.items():
            status = "적합" if adeq["adequate"] else "부적합"
            color = "#2ECC71" if adeq["adequate"] else "#E74C3C"
            reasons = adeq.get("reasons", [])
            adequacy_items.append(html.Div(style={
                "padding": "8px 12px", "borderLeft": f"4px solid {color}",
                "marginBottom": "8px", "backgroundColor": "#fff",
            }, children=[
                html.Span(f"[{status}] ", style={"color": color, "fontWeight": "bold"}),
                html.Span(title),
                html.Div(", ".join(reasons) if reasons else "",
                          style={"fontSize": "11px", "color": "#888"}),
            ]))

        return html.Div([
            html.H4("전체 평균 지표"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                             "gap": "12px", "marginBottom": "20px"}, children=overall_cards),
            html.H4("영화별 상세 지표"),
            dcc.Graph(figure=fig),
            html.H4("적합/부적합 판정"),
            html.Div(children=adequacy_items),
        ])

    # ==================================================================
    # 탭 5: 민감도
    # ==================================================================
    _sensitivity_progress = {"current": 0, "total": 27, "running": False, "done": False}

    def _tab_sensitivity():
        return html.Div([
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "20px",
                             "marginBottom": "20px"}, children=[
                html.Button("민감도 분석 실행 (27조합)", id="run-sensitivity-btn", style={
                    "padding": "10px 24px", "backgroundColor": "#9B59B6",
                    "color": "#fff", "border": "none", "borderRadius": "6px",
                    "fontSize": "14px", "cursor": "pointer", "whiteSpace": "nowrap",
                }),
                html.Div(style={"flex": "1"}, children=[
                    html.Div(id="progress-text", style={
                        "fontSize": "13px", "color": "#555", "marginBottom": "4px",
                    }, children="분석 실행 버튼을 클릭하세요."),
                    html.Div(style={
                        "width": "100%", "height": "22px", "backgroundColor": "#ecf0f1",
                        "borderRadius": "11px", "overflow": "hidden",
                    }, children=[
                        html.Div(id="progress-bar", style={
                            "width": "0%", "height": "100%",
                            "backgroundColor": "#9B59B6", "borderRadius": "11px",
                            "transition": "width 0.3s ease",
                        }),
                    ]),
                ]),
            ]),
            dcc.Interval(id="progress-interval", interval=500, disabled=True),
            html.Div(id="sensitivity-status", style={"display": "none"}),
            html.Div(id="sensitivity-results"),
        ])

    # ==================================================================
    # 콜백
    # ==================================================================

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "tab-search":
            return _tab_search()
        elif tab == "tab-cluster":
            return _tab_cluster()
        elif tab == "tab-params":
            return _tab_params()
        elif tab == "tab-eval":
            return _tab_eval()
        elif tab == "tab-sensitivity":
            return _tab_sensitivity()
        return html.Div()

    @app.callback(
        [Output("search-results", "children"),
         Output("search-info", "children")],
        Input("search-btn", "n_clicks"),
        State("search-input", "value"),
        prevent_initial_call=True,
    )
    def do_search(n_clicks, query):
        if not query or not query.strip():
            return [], ""
        results, stype, parsed = engine.search(query.strip(), top_k=config.TOP_K)
        info_parts = [f"검색 방식: {stype}"]
        if stype == "text" and parsed.get("genres"):
            info_parts.append(f"파싱 장르: {', '.join(parsed['genres'])}")
        if stype == "text" and parsed.get("keywords"):
            info_parts.append(f"파싱 키워드: {', '.join(parsed['keywords'])}")
        info_parts.append(f"결과: {len(results)}편")

        cards = [_make_movie_card(r, r["rank"]) for r in results]
        return cards, " | ".join(info_parts)

    @app.callback(
        [Output("weight-results", "children"),
         Output("weight-info", "children")],
        Input("apply-weights-btn", "n_clicks"),
        [State("w-genre", "value"), State("w-keyword", "value"),
         State("w-numeric", "value"), State("w-text", "value")],
        prevent_initial_call=True,
    )
    def apply_weights(n_clicks, wg, wk, wn, wt):
        emb = result["embedding"]
        new_raw = emb.rebuild_with_weights(w_genre=wg, w_keyword=wk,
                                            w_numeric=wn, w_text=wt)
        old_raw = emb.raw_vectors
        emb.raw_vectors = new_raw

        train_ids = [m["id"] for m in result["train_movies"]]
        children = []

        for tm in result["test_movies"]:
            tid = tm["id"]
            if tid not in new_raw:
                continue
            ranked = emb.compute_similarity_to_train(new_raw[tid], train_ids)
            new_ids = [mid for mid, _ in ranked if mid != tid][:config.TOP_K]
            base_ids = baseline_recs.get(tm["title"], [])

            overlap = len(set(new_ids) & set(base_ids))
            overlap_pct = overlap / config.TOP_K * 100 if config.TOP_K > 0 else 0

            entered = set(new_ids) - set(base_ids)
            exited = set(base_ids) - set(new_ids)

            id_to_movie = {m["id"]: m for m in result["train_movies"]}
            entered_names = [id_to_movie[i]["title"] for i in entered if i in id_to_movie][:5]
            exited_names = [id_to_movie[i]["title"] for i in exited if i in id_to_movie][:5]

            children.append(html.Div(style={
                "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                "marginBottom": "12px", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
            }, children=[
                html.H4(f"{tm['title']} ({tm['year']})", style={"margin": "0 0 8px 0"}),
                html.Div(f"기준선 대비 오버랩: {overlap}/{config.TOP_K} ({overlap_pct:.0f}%)",
                          style={"fontWeight": "bold", "marginBottom": "6px"}),
                html.Div(f"진입: {', '.join(entered_names)}" if entered_names else "진입: 없음",
                          style={"fontSize": "12px", "color": "#27AE60"}),
                html.Div(f"이탈: {', '.join(exited_names)}" if exited_names else "이탈: 없음",
                          style={"fontSize": "12px", "color": "#E74C3C"}),
            ]))

        emb.raw_vectors = old_raw

        info = f"가중치: 장르={wg:.1f}, 키워드={wk:.1f}, 수치={wn:.1f}, 텍스트={wt:.1f}"
        return children, info

    # -- 민감도: 버튼 클릭 -> 백그라운드 분석 시작 + Interval 활성화
    @app.callback(
        Output("progress-interval", "disabled"),
        Input("run-sensitivity-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def start_sensitivity(n_clicks):
        if _sensitivity_progress["running"]:
            return dash.no_update

        _sensitivity_progress["current"] = 0
        _sensitivity_progress["total"] = 27
        _sensitivity_progress["running"] = True
        _sensitivity_progress["done"] = False
        _sensitivity_progress["analysis"] = None

        def _run():
            from sensitivity import SensitivityAnalyzer
            analyzer = SensitivityAnalyzer()

            def _on_progress(cur, total):
                _sensitivity_progress["current"] = cur
                _sensitivity_progress["total"] = total

            _, analysis = analyzer.analyze_movies(
                result["embedding"], result["train_movies"], result["test_movies"],
                progress_callback=_on_progress,
            )
            _sensitivity_progress["analysis"] = analysis
            _sensitivity_progress["running"] = False
            _sensitivity_progress["done"] = True

        threading.Thread(target=_run, daemon=True).start()
        return False  # Interval 활성화

    # -- 민감도: Interval 폴링 -> 진행률 바 + 완료 시 결과 렌더링
    @app.callback(
        [Output("progress-bar", "style"),
         Output("progress-text", "children"),
         Output("sensitivity-results", "children"),
         Output("sensitivity-status", "children"),
         Output("progress-interval", "disabled", allow_duplicate=True)],
        Input("progress-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_progress(n):
        cur = _sensitivity_progress["current"]
        total = _sensitivity_progress["total"]
        pct = int(cur / total * 100) if total > 0 else 0

        bar_style = {
            "width": f"{pct}%", "height": "100%",
            "backgroundColor": "#9B59B6", "borderRadius": "11px",
            "transition": "width 0.3s ease",
        }

        if not _sensitivity_progress["done"]:
            return (
                bar_style,
                f"분석 중... {cur}/{total} ({pct}%)",
                dash.no_update,
                dash.no_update,
                False,  # Interval 유지
            )

        # 완료
        bar_style["width"] = "100%"
        bar_style["backgroundColor"] = "#27AE60"
        analysis = _sensitivity_progress.get("analysis", [])

        if not analysis:
            return bar_style, "분석 완료 (결과 없음)", [], "", True

        # 히트맵 생성
        queries = list(set(a["query"] for a in analysis))
        combos = list(set(a["combo"] for a in analysis))
        combos.sort()

        z_data = []
        for q in queries:
            row = []
            for combo in combos:
                match = [a for a in analysis if a["query"] == q and a["combo"] == combo]
                row.append(match[0]["overlap"] if match else 0)
            z_data.append(row)

        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=[c[:20] for c in combos],
            y=[q[:15] for q in queries],
            colorscale="YlOrRd",
            text=np.round(z_data, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="27조합 Top-20 오버랩 비율 히트맵",
            xaxis_title="가중치 조합", yaxis_title="쿼리",
            height=max(400, len(queries) * 80),
            xaxis_tickangle=-45,
        )

        table_data = []
        for a in analysis[:50]:
            table_data.append({
                "조합": a["combo"],
                "쿼리": a["query"][:20],
                "오버랩": f"{a['overlap']:.2%}",
                "Spearman": f"{a['spearman_rho']:.4f}",
            })

        results_children = [
            dcc.Graph(figure=fig),
            html.H4("상세 결과 (상위 50건)", style={"marginTop": "20px"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in ["조합", "쿼리", "오버랩", "Spearman"]],
                data=table_data,
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
                style_header={"backgroundColor": "#2c3e50", "color": "#fff"},
                page_size=20,
            ),
        ]

        return (
            bar_style,
            f"분석 완료: {len(analysis)}개 조합-쿼리 결과",
            results_children,
            "",
            True,  # Interval 중지
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)
