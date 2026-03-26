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
# 그룹 기여도 바 렌더링
# ------------------------------------------------------------------
def _render_group_sim_bars(gs):
    """장르/키워드/텍스트 기여도를 수평 바로 렌더링."""
    if not gs:
        return html.Div()
    bars = []
    groups = [
        ("genre", "장르", "#E74C3C"),
        ("keyword", "키워드", "#3498DB"),
        ("text", "텍스트", "#2ECC71"),
    ]
    for key, label, color in groups:
        val = gs.get(key, 0)
        pct = max(0, min(100, val * 100))
        bars.append(html.Div(style={
            "display": "flex", "alignItems": "center", "gap": "4px",
            "marginBottom": "2px",
        }, children=[
            html.Span(label, style={
                "fontSize": "9px", "color": "#666", "width": "36px",
                "textAlign": "right",
            }),
            html.Div(style={
                "flex": "1", "height": "4px", "backgroundColor": "#eee",
                "borderRadius": "2px", "overflow": "hidden",
            }, children=[
                html.Div(style={
                    "width": f"{pct}%", "height": "100%",
                    "backgroundColor": color, "borderRadius": "2px",
                }),
            ]),
            html.Span(f"{val:.2f}", style={
                "fontSize": "9px", "color": color, "width": "30px",
            }),
        ]))
    return html.Div(style={"marginTop": "4px"}, children=bars)


# ------------------------------------------------------------------
# 카드 컴포넌트
# ------------------------------------------------------------------
def _make_movie_card(rec, idx):
    is_source = rec.get("is_source", False)

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

    # 원본 영화 스타일
    if is_source:
        card_style = {
            "border": "2px solid #E74C3C", "borderRadius": "8px", "padding": "12px",
            "display": "flex", "gap": "12px",
            "backgroundColor": "#FFF5F5",
            "boxShadow": "0 2px 8px rgba(231,76,60,0.15)",
        }
        rank_el = html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
            html.Span("★ 원본 영화", style={
                "fontSize": "11px", "fontWeight": "bold", "color": "#fff",
                "backgroundColor": "#E74C3C", "padding": "2px 8px",
                "borderRadius": "10px",
            }),
        ])
        sim_bar = html.Div("검색된 원본 영화", style={
            "fontSize": "12px", "color": "#E74C3C", "fontWeight": "bold",
            "marginTop": "4px",
        })
    else:
        card_style = {
            "border": "1px solid #e0e0e0", "borderRadius": "8px", "padding": "12px",
            "display": "flex", "gap": "12px", "backgroundColor": "#fff",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        }
        rank_el = html.Div(f"#{idx}", style={"fontSize": "11px", "color": "#999"})
        sim_bar = html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
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
        ])

    group_bars = _render_group_sim_bars(rec.get("group_similarity"))

    return html.Div(style=card_style, children=[
        poster_el,
        html.Div(style={"flex": "1", "minWidth": "0"}, children=[
            rank_el,
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
            sim_bar,
            group_bars,
            html.Div(rec.get("explanation", ""),
                      style={"fontSize": "10px", "color": "#888", "marginTop": "2px"}),
        ]),
    ])


# ------------------------------------------------------------------
# 기여도 예시 패널
# ------------------------------------------------------------------
def _build_contribution_examples_panel(results):
    """추천 결과에서 장르/키워드/텍스트가 주도적인 예시 쌍을 선정한다."""
    candidates = [r for r in results if not r.get("is_source") and r.get("group_similarity")]
    if not candidates:
        return html.Div()

    genre_dominant = []
    keyword_dominant = []
    text_dominant = []

    for r in candidates:
        gs = r["group_similarity"]
        g, k, t = gs.get("genre", 0), gs.get("keyword", 0), gs.get("text", 0)
        if g > k and g > t and g > 0.1:
            genre_dominant.append((r, g))
        if k > g and k > t and k > 0.1:
            keyword_dominant.append((r, k))
        if t > g and t > k and t > 0.1:
            text_dominant.append((r, t))

    genre_dominant.sort(key=lambda x: -x[1])
    keyword_dominant.sort(key=lambda x: -x[1])
    text_dominant.sort(key=lambda x: -x[1])

    def _example_card(title, color, items, factor_label):
        if not items:
            return html.Div(style={
                "flex": "1", "padding": "12px", "borderRadius": "8px",
                "backgroundColor": "#fff", "border": f"2px solid {color}",
            }, children=[
                html.Div(title, style={"fontWeight": "bold", "color": color, "marginBottom": "6px"}),
                html.Div("해당 요인 주도 예시 없음", style={"fontSize": "12px", "color": "#999"}),
            ])
        entries = []
        for r, val in items[:3]:
            entries.append(html.Div(style={
                "display": "flex", "justifyContent": "space-between",
                "padding": "4px 0", "borderBottom": "1px solid #f0f0f0",
            }, children=[
                html.Span(r["title"][:20], style={"fontSize": "12px"}),
                html.Span(f"{factor_label}={val:.3f}", style={
                    "fontSize": "11px", "fontWeight": "bold", "color": color,
                }),
            ]))
        return html.Div(style={
            "flex": "1", "padding": "12px", "borderRadius": "8px",
            "backgroundColor": "#fff", "border": f"2px solid {color}",
        }, children=[
            html.Div(title, style={"fontWeight": "bold", "color": color, "marginBottom": "6px"}),
            html.Div(children=entries),
        ])

    return html.Div(style={"marginTop": "20px"}, children=[
        html.H4("기여도별 예시", style={"marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            _example_card("장르 주도", "#E74C3C", genre_dominant, "장르"),
            _example_card("키워드 주도", "#3498DB", keyword_dominant, "키워드"),
            _example_card("텍스트 주도", "#2ECC71", text_dominant, "텍스트"),
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

    # 상태 관리
    _eff = config.get_effective_weights()
    _weight_state = {
        "previous": _eff.copy(),
        "current": _eff.copy(),
    }

    _sim_state = {
        "running": False,
        "done": False,
        "progress": 0,
        "total": 50,
        "best_weights": None,
        "best_score": 0,
        "best_metrics": None,
        "confidence": 0,
        "accuracy": 0,
        "history": [],
    }

    # ==================================================================
    # 레이아웃 -- 모든 탭을 미리 렌더링, display로 토글 (상태 보존)
    # ==================================================================

    header = html.Div(style={
        "backgroundColor": "#2c3e50", "color": "#fff", "padding": "16px 24px",
        "fontSize": "20px", "fontWeight": "bold",
        "position": "sticky", "top": "0", "zIndex": "1000",
    }, children="KMDB 영화 추천 시스템 -- 코사인 유사도 기반 하이브리드 추천")

    tabs = dcc.Tabs(id="main-tabs", value="tab-search", style={
        "position": "sticky", "top": "56px", "zIndex": "999",
        "backgroundColor": "#fff",
    }, children=[
        dcc.Tab(label="검색 및 추천", value="tab-search"),
        dcc.Tab(label="클러스터 시각화", value="tab-cluster"),
        dcc.Tab(label="파라미터 제어", value="tab-params"),
        dcc.Tab(label="평가", value="tab-eval"),
        dcc.Tab(label="민감도 분석", value="tab-sensitivity"),
    ])

    # -- 탭 1: 검색
    tab_search_content = html.Div(id="tab-search-div", style={"padding": "20px"}, children=[
        html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "20px"}, children=[
            dcc.Input(
                id="search-input", type="text",
                placeholder="영화 제목 또는 키워드를 입력하세요...",
                debounce=True,
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
            "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "12px",
        }),
        html.Div(id="contribution-examples"),
    ])

    # -- 탭 2: 클러스터 (정적 콘텐츠, 미리 생성)
    def _build_cluster_content():
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

        cluster_info = result.get("cluster_info", {})

        def _genre_badge(name, count):
            bg = config.GENRE_COLORS.get(name, "#BDBDBD")
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

        return [
            dcc.Graph(figure=fig),
            html.H4("클러스터 정보", style={"marginTop": "20px"}),
            html.Table([table_header, html.Tbody(table_rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ]

    tab_cluster_content = html.Div(id="tab-cluster-div",
                                    style={"padding": "20px", "display": "none"},
                                    children=_build_cluster_content())

    # -- 탭 3: 파라미터 제어
    eff = config.get_effective_weights()
    tab_params_content = html.Div(id="tab-params-div", style={"padding": "20px", "display": "none"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                         "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div([
                html.Label("장르 가중치", style={"fontWeight": "bold"}),
                dcc.Slider(id="w-genre", min=0, max=3, step=0.1,
                           value=eff["genre"],
                           marks={0: "0", 1: "1", 2: "2", 3: "3"}),
            ]),
            html.Div([
                html.Label("키워드 가중치", style={"fontWeight": "bold"}),
                dcc.Slider(id="w-keyword", min=0, max=3, step=0.1,
                           value=eff["keyword"],
                           marks={0: "0", 1: "1", 2: "2", 3: "3"}),
            ]),
            html.Div([
                html.Label("수치 가중치", style={"fontWeight": "bold"}),
                dcc.Slider(id="w-numeric", min=0, max=3, step=0.1,
                           value=eff["numeric"],
                           marks={0: "0", 1: "1", 2: "2", 3: "3"}),
            ]),
            html.Div([
                html.Label("텍스트 가중치", style={"fontWeight": "bold"}),
                dcc.Slider(id="w-text", min=0, max=3, step=0.1,
                           value=eff["text"],
                           marks={0: "0", 1: "1", 2: "2", 3: "3"}),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "20px",
                         "flexWrap": "wrap"}, children=[
            html.Button("가중치 적용", id="apply-weights-btn", style={
                "padding": "10px 24px", "backgroundColor": "#E74C3C",
                "color": "#fff", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer",
            }),
            html.Button("초기화", id="reset-weights-btn", style={
                "padding": "10px 24px", "backgroundColor": "#3498DB",
                "color": "#fff", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer",
            }),
            html.Button("되돌리기", id="undo-weights-btn", style={
                "padding": "10px 24px", "backgroundColor": "#F1C40F",
                "color": "#333", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer",
            }),
            html.Button("시뮬레이션", id="sim-weights-btn", style={
                "padding": "10px 24px", "backgroundColor": "#9B59B6",
                "color": "#fff", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer",
            }),
            html.Button("저장 및 반영", id="save-sim-btn", style={
                "padding": "10px 24px", "backgroundColor": "#27AE60",
                "color": "#fff", "border": "none", "borderRadius": "6px",
                "fontSize": "14px", "cursor": "pointer",
            }),
        ]),
        html.Div(id="sim-progress-area", style={"marginBottom": "16px"}, children=[
            html.Div(id="sim-progress-text", style={
                "fontSize": "13px", "color": "#555", "marginBottom": "4px",
            }),
            html.Div(style={
                "width": "100%", "height": "22px", "backgroundColor": "#ecf0f1",
                "borderRadius": "11px", "overflow": "hidden",
            }, children=[
                html.Div(id="sim-progress-bar", style={
                    "width": "0%", "height": "100%",
                    "backgroundColor": "#9B59B6", "borderRadius": "11px",
                    "transition": "width 0.3s ease",
                }),
            ]),
        ]),
        dcc.Interval(id="sim-interval", interval=500, disabled=True),
        html.Div(id="sim-results"),
        html.Div(id="weight-info", style={"marginBottom": "10px", "color": "#666"}),
        html.Div(id="weight-results"),
    ])

    # -- 탭 4: 평가 (정적)
    def _build_eval_content():
        quant = result.get("quant", {})
        overall = quant.get("overall", {})
        per_query = quant.get("per_query", {})
        comparison = result.get("comparison", {})

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

        return [
            html.H4("전체 평균 지표"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                             "gap": "12px", "marginBottom": "20px"}, children=overall_cards),
            html.H4("영화별 상세 지표"),
            dcc.Graph(figure=fig),
            html.H4("적합/부적합 판정"),
            html.Div(children=adequacy_items),
        ]

    tab_eval_content = html.Div(id="tab-eval-div",
                                 style={"padding": "20px", "display": "none"},
                                 children=_build_eval_content())

    # -- 탭 5: 민감도
    _sensitivity_progress = {"current": 0, "total": 81, "running": False, "done": False}

    tab_sensitivity_content = html.Div(id="tab-sensitivity-div", style={"padding": "20px", "display": "none"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "20px",
                         "marginBottom": "20px"}, children=[
            html.Button("민감도 분석 실행 (81조합)", id="run-sensitivity-btn", style={
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

    app.layout = html.Div(style={"fontFamily": "Malgun Gothic, sans-serif",
                                   "backgroundColor": "#f5f5f5"}, children=[
        header,
        tabs,
        # 모든 탭 콘텐츠를 동시에 배치, display로 토글
        tab_search_content,
        tab_cluster_content,
        tab_params_content,
        tab_eval_content,
        tab_sensitivity_content,
    ])

    # ==================================================================
    # 콜백
    # ==================================================================

    # -- 탭 전환: display 토글
    @app.callback(
        [Output("tab-search-div", "style"),
         Output("tab-cluster-div", "style"),
         Output("tab-params-div", "style"),
         Output("tab-eval-div", "style"),
         Output("tab-sensitivity-div", "style")],
        Input("main-tabs", "value"),
    )
    def toggle_tabs(tab):
        styles = []
        for t in ["tab-search", "tab-cluster", "tab-params", "tab-eval", "tab-sensitivity"]:
            if t == tab:
                styles.append({"padding": "20px", "display": "block"})
            else:
                styles.append({"padding": "20px", "display": "none"})
        return styles

    # -- 검색 콜백 (버튼 클릭 또는 Enter 키)
    @app.callback(
        [Output("search-results", "children"),
         Output("search-info", "children"),
         Output("contribution-examples", "children")],
        [Input("search-btn", "n_clicks"),
         Input("search-input", "n_submit")],
        State("search-input", "value"),
        prevent_initial_call=True,
    )
    def do_search(n_clicks, n_submit, query):
        if not query or not query.strip():
            return [], "", html.Div()
        results, stype, parsed = engine.search(query.strip(), top_k=config.TOP_K)
        info_parts = [f"검색 방식: {stype}"]
        if stype == "title":
            source = next((r for r in results if r.get("is_source")), None)
            if source:
                info_parts.append(f"매칭 영화: {source['title']} ({source['year']})")
        if stype == "text" and parsed.get("genres"):
            info_parts.append(f"파싱 장르: {', '.join(parsed['genres'])}")
        if stype == "text" and parsed.get("keywords"):
            info_parts.append(f"파싱 키워드: {', '.join(parsed['keywords'])}")
        if parsed.get("corrections"):
            corrections = parsed["corrections"]
            corr_str = ", ".join(f"{c['original']}→{c['corrected']}" for c in corrections)
            info_parts.append(f"오타 교정: {corr_str}")
        info_parts.append(f"결과: {len(results)}편")

        cards = [_make_movie_card(r, r["rank"]) for r in results]
        examples_panel = _build_contribution_examples_panel(results)
        return cards, " | ".join(info_parts), examples_panel

    # -- 가중치 적용
    @app.callback(
        [Output("weight-results", "children"),
         Output("weight-info", "children")],
        Input("apply-weights-btn", "n_clicks"),
        [State("w-genre", "value"), State("w-keyword", "value"),
         State("w-numeric", "value"), State("w-text", "value")],
        prevent_initial_call=True,
    )
    def apply_weights(n_clicks, wg, wk, wn, wt):
        _weight_state["previous"] = _weight_state["current"].copy()
        _weight_state["current"] = {
            "genre": wg, "keyword": wk, "numeric": wn, "text": wt,
        }

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

    # -- 초기화 버튼
    @app.callback(
        [Output("w-genre", "value", allow_duplicate=True),
         Output("w-keyword", "value", allow_duplicate=True),
         Output("w-numeric", "value", allow_duplicate=True),
         Output("w-text", "value", allow_duplicate=True)],
        Input("reset-weights-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_weights(n_clicks):
        if os.path.exists(config.SAVED_WEIGHTS_PATH):
            os.remove(config.SAVED_WEIGHTS_PATH)
        return config.WEIGHT_GENRE, config.WEIGHT_KEYWORD, config.WEIGHT_NUMERIC, config.WEIGHT_TEXT

    # -- 되돌리기 버튼
    @app.callback(
        [Output("w-genre", "value", allow_duplicate=True),
         Output("w-keyword", "value", allow_duplicate=True),
         Output("w-numeric", "value", allow_duplicate=True),
         Output("w-text", "value", allow_duplicate=True)],
        Input("undo-weights-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def undo_weights(n_clicks):
        prev = _weight_state["previous"]
        return prev["genre"], prev["keyword"], prev["numeric"], prev["text"]

    # -- 시뮬레이션 시작
    @app.callback(
        Output("sim-interval", "disabled"),
        Input("sim-weights-btn", "n_clicks"),
        [State("w-genre", "value"), State("w-keyword", "value"),
         State("w-numeric", "value"), State("w-text", "value")],
        prevent_initial_call=True,
    )
    def start_simulation(n_clicks, wg, wk, wn, wt):
        if _sim_state["running"]:
            return dash.no_update

        _sim_state["running"] = True
        _sim_state["done"] = False
        _sim_state["progress"] = 0
        _sim_state["total"] = 50
        _sim_state["best_weights"] = None
        _sim_state["best_score"] = 0
        _sim_state["best_metrics"] = None
        _sim_state["confidence"] = 0
        _sim_state["accuracy"] = 0
        _sim_state["history"] = []

        initial_weights = {"genre": wg, "keyword": wk, "numeric": wn, "text": wt}

        def _run():
            from optimizer import WeightOptimizer
            optimizer = WeightOptimizer(
                result["embedding"],
                result["train_movies"],
                result["test_movies"],
                result.get("coords", {}),
            )

            def _on_progress(cur, total, best_score):
                _sim_state["progress"] = cur
                _sim_state["total"] = total

            opt_result = optimizer.optimize(
                initial_weights=initial_weights,
                progress_callback=_on_progress,
            )

            _sim_state["best_weights"] = opt_result["best_weights"]
            _sim_state["best_score"] = opt_result["best_score"]
            _sim_state["best_metrics"] = opt_result["best_metrics"]
            _sim_state["confidence"] = opt_result["confidence"]
            _sim_state["accuracy"] = opt_result["accuracy"]
            _sim_state["history"] = opt_result["history"]
            _sim_state["running"] = False
            _sim_state["done"] = True

        threading.Thread(target=_run, daemon=True).start()
        return False  # Interval 활성화

    # -- 시뮬레이션 폴링 (진행 중에도 실시간 업데이트)
    @app.callback(
        [Output("sim-progress-bar", "style"),
         Output("sim-progress-text", "children"),
         Output("sim-results", "children"),
         Output("sim-interval", "disabled", allow_duplicate=True)],
        Input("sim-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_simulation(n):
        cur = _sim_state["progress"]
        total = _sim_state["total"]
        pct = int(cur / total * 100) if total > 0 else 0

        bar_style = {
            "width": f"{pct}%", "height": "100%",
            "backgroundColor": "#9B59B6", "borderRadius": "11px",
            "transition": "width 0.3s ease",
        }

        # 현재까지의 히스토리 차트 (진행 중에도 표시)
        history = _sim_state.get("history", [])
        bw = _sim_state.get("best_weights") or {}
        bm = _sim_state.get("best_metrics") or {}
        best_score = _sim_state.get("best_score", 0)
        conf = _sim_state.get("confidence", 0)
        acc = _sim_state.get("accuracy", 0)

        result_children = []

        # 현재 최적 가중치 (진행 중에도 표시)
        if bw:
            result_children.append(html.Div(style={
                "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "12px", "marginBottom": "16px",
            }, children=[
                html.Div(style={
                    "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                    "textAlign": "center", "border": "2px solid #9B59B6",
                }, children=[
                    html.Div("장르", style={"fontSize": "11px", "color": "#888"}),
                    html.Div(f"{bw.get('genre', 0):.1f}", style={
                        "fontSize": "24px", "fontWeight": "bold", "color": "#9B59B6",
                    }),
                ]) for _ in [0]  # placeholder
            ] + [
                html.Div(style={
                    "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                    "textAlign": "center", "border": "2px solid #9B59B6",
                }, children=[
                    html.Div(lbl, style={"fontSize": "11px", "color": "#888"}),
                    html.Div(f"{bw.get(key, 0):.1f}", style={
                        "fontSize": "24px", "fontWeight": "bold", "color": "#9B59B6",
                    }),
                ]) for key, lbl in [("keyword", "키워드"), ("numeric", "수치"), ("text", "텍스트")]
            ]))

        # 점수/신뢰도/정확도
        if best_score > 0:
            conf_color = "#27AE60" if conf >= 0.5 else "#E74C3C"
            acc_color = "#27AE60" if acc >= 0.5 else "#E74C3C"
            result_children.append(html.Div(style={
                "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                "gap": "12px", "marginBottom": "16px",
            }, children=[
                html.Div(style={
                    "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                    "textAlign": "center",
                }, children=[
                    html.Div("복합 점수", style={"fontSize": "11px", "color": "#888"}),
                    html.Div(f"{best_score:.4f}", style={
                        "fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50",
                    }),
                ]),
                html.Div(style={
                    "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                    "textAlign": "center",
                }, children=[
                    html.Div("신뢰도", style={"fontSize": "11px", "color": "#888"}),
                    html.Div(f"{conf:.1%}", style={
                        "fontSize": "20px", "fontWeight": "bold", "color": conf_color,
                    }),
                ]),
                html.Div(style={
                    "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                    "textAlign": "center",
                }, children=[
                    html.Div("정확도", style={"fontSize": "11px", "color": "#888"}),
                    html.Div(f"{acc:.1%}", style={
                        "fontSize": "20px", "fontWeight": "bold", "color": acc_color,
                    }),
                ]),
            ]))

        # 메트릭 상세
        if bm:
            result_children.append(html.Div(style={
                "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                "marginBottom": "16px",
            }, children=[
                html.Div("최적화 메트릭 상세", style={"fontWeight": "bold", "marginBottom": "8px"}),
                html.Div(f"평균 유사도: {bm.get('avg_similarity', 0):.4f}  |  "
                          f"장르 정밀도: {bm.get('genre_precision', 0):.4f}  |  "
                          f"텍스트 일관성: {bm.get('text_coherence', 0):.4f}  |  "
                          f"다양성: {bm.get('diversity', 0):.4f}",
                          style={"fontSize": "13px", "color": "#555"}),
            ]))

        # 히스토리 차트 (진행 중에도 실시간 표시)
        if history and len(history) > 1:
            iters = [h["iteration"] for h in history]
            scores = [h["score"] for h in history]
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Scatter(
                x=iters, y=scores, mode="lines+markers",
                line=dict(color="#9B59B6"), marker=dict(size=4),
                name="복합 점수",
            ))
            hist_fig.update_layout(
                title="시뮬레이션 탐색 히스토리",
                xaxis_title="반복", yaxis_title="복합 점수",
                height=300,
            )
            result_children.append(dcc.Graph(figure=hist_fig))

        if not _sim_state["done"]:
            return (
                bar_style,
                f"시뮬레이션 중... {cur}/{total} ({pct}%)",
                result_children,
                False,
            )

        # 완료
        bar_style["width"] = "100%"
        bar_style["backgroundColor"] = "#27AE60"

        if conf >= 0.5 and acc >= 0.5:
            result_children.append(html.Div(
                "✓ 신뢰도와 정확도가 충분합니다. '저장 및 반영' 버튼으로 슬라이더에 적용할 수 있습니다.",
                style={"color": "#27AE60", "fontWeight": "bold", "marginTop": "8px"},
            ))
        else:
            result_children.append(html.Div(
                "✗ 신뢰도 또는 정확도가 부족합니다. 저장 및 반영이 제한됩니다.",
                style={"color": "#E74C3C", "fontWeight": "bold", "marginTop": "8px"},
            ))

        return (
            bar_style,
            f"시뮬레이션 완료: {len(history)-1}회 반복",
            result_children,
            True,
        )

    # -- 저장 및 반영
    @app.callback(
        [Output("w-genre", "value", allow_duplicate=True),
         Output("w-keyword", "value", allow_duplicate=True),
         Output("w-numeric", "value", allow_duplicate=True),
         Output("w-text", "value", allow_duplicate=True)],
        Input("save-sim-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_simulation_weights(n_clicks):
        if not _sim_state["done"] or not _sim_state["best_weights"]:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        conf = _sim_state["confidence"]
        acc = _sim_state["accuracy"]
        if conf < 0.5 or acc < 0.5:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        bw = _sim_state["best_weights"]
        config.save_weights(
            weights=bw,
            source="simulation",
            confidence=_sim_state["confidence"],
            accuracy=_sim_state["accuracy"],
            score=_sim_state["best_score"],
        )
        return bw["genre"], bw["keyword"], bw["numeric"], bw["text"]

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
        _sensitivity_progress["total"] = 81
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

    # -- 민감도: Interval 폴링
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
                False,
            )

        bar_style["width"] = "100%"
        bar_style["backgroundColor"] = "#27AE60"
        analysis = _sensitivity_progress.get("analysis", [])

        if not analysis:
            return bar_style, "분석 완료 (결과 없음)", [], "", True

        # ---- 기여도 순위 계산 ----
        from sensitivity import SensitivityAnalyzer
        _analyzer = SensitivityAnalyzer()
        importances = _analyzer.compute_contribution_importance(analysis)

        imp_colors = ["#E74C3C", "#3498DB", "#F1C40F", "#2ECC71"]
        imp_labels = [imp["label"] for imp in importances]
        imp_values = [imp["importance"] for imp in importances]

        imp_fig = go.Figure()
        imp_fig.add_trace(go.Bar(
            x=imp_labels, y=imp_values,
            marker_color=imp_colors[:len(importances)],
            text=[f"{v:.4f}" for v in imp_values],
            textposition="outside",
        ))
        imp_fig.update_layout(
            title="파라미터별 기여도 (Contribution Importance)",
            xaxis_title="파라미터", yaxis_title="중요도 (Marginal Range)",
            height=350,
        )

        imp_cards = []
        for rank, imp in enumerate(importances):
            color = imp_colors[rank] if rank < len(imp_colors) else "#999"
            levels_text = " | ".join(f"{k}: {v:.4f}" for k, v in imp["level_means"].items())
            imp_cards.append(html.Div(style={
                "backgroundColor": "#fff", "borderRadius": "8px", "padding": "12px",
                "border": f"2px solid {color}",
            }, children=[
                html.Div(f"#{rank+1} {imp['label']}", style={
                    "fontWeight": "bold", "fontSize": "16px", "color": color,
                }),
                html.Div(f"중요도: {imp['importance']:.4f}", style={"fontSize": "14px"}),
                html.Div(f"최적 수준: {imp['best_level']} ({imp['best_value']:.1f})",
                          style={"fontSize": "12px", "color": "#555"}),
                html.Div(levels_text, style={"fontSize": "11px", "color": "#888"}),
            ]))

        # ---- 히트맵 ----
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
            title="81조합 Top-20 오버랩 비율 히트맵",
            xaxis_title="가중치 조합", yaxis_title="쿼리",
            height=max(400, len(queries) * 80),
            xaxis_tickangle=-90,
        )

        # ---- 상세 테이블 (유사도 내림차순 + 색상) ----
        # 수준별 배경색: 상=파랑, 중=초록, 하=주황
        _LEVEL_BG = {"상": "rgba(41,128,185,0.25)", "중": "rgba(39,174,96,0.20)", "하": "rgba(230,126,34,0.25)"}

        sorted_analysis = sorted(analysis, key=lambda a: a.get("avg_similarity", 0), reverse=True)
        table_data = []
        for a in sorted_analysis[:50]:
            combo_str = a["combo"]
            table_data.append({
                "조합": combo_str,
                "쿼리": a["query"][:20],
                "평균유사도": round(a.get("avg_similarity", 0), 4),
                "오버랩": round(a.get("overlap", 0), 4),
                "Spearman": round(a.get("spearman_rho", 0), 4),
            })

        # 메트릭 범위 계산 (조건부 스타일용)
        all_sims = [a.get("avg_similarity", 0) for a in sorted_analysis[:50]] or [0]
        all_overlaps = [a.get("overlap", 0) for a in sorted_analysis[:50]] or [0]
        all_rhos = [a.get("spearman_rho", 0) for a in sorted_analysis[:50]] or [0]

        sim_lo, sim_hi = min(all_sims), max(all_sims)
        ov_lo, ov_hi = min(all_overlaps), max(all_overlaps)
        rho_lo, rho_hi = min(all_rhos), max(all_rhos)

        def _metric_style_conditions(col_id, lo, hi, good_color, bad_color):
            """DataTable style_data_conditional 항목 생성."""
            if hi <= lo:
                return []
            mid = (lo + hi) / 2
            q1 = lo + (hi - lo) * 0.25
            q3 = lo + (hi - lo) * 0.75
            return [
                {"if": {"column_id": col_id, "filter_query": f"{{{col_id}}} >= {q3}"},
                 "backgroundColor": good_color, "fontWeight": "bold"},
                {"if": {"column_id": col_id, "filter_query": f"{{{col_id}}} >= {mid} && {{{col_id}}} < {q3}"},
                 "backgroundColor": "rgba(39,174,96,0.12)"},
                {"if": {"column_id": col_id, "filter_query": f"{{{col_id}}} >= {q1} && {{{col_id}}} < {mid}"},
                 "backgroundColor": "rgba(241,196,15,0.15)"},
                {"if": {"column_id": col_id, "filter_query": f"{{{col_id}}} < {q1}"},
                 "backgroundColor": bad_color},
            ]

        style_conditions = []
        style_conditions += _metric_style_conditions("평균유사도", sim_lo, sim_hi,
                                                      "rgba(39,174,96,0.25)", "rgba(231,76,60,0.15)")
        style_conditions += _metric_style_conditions("오버랩", ov_lo, ov_hi,
                                                      "rgba(39,174,96,0.25)", "rgba(231,76,60,0.15)")
        style_conditions += _metric_style_conditions("Spearman", rho_lo, rho_hi,
                                                      "rgba(39,174,96,0.25)", "rgba(231,76,60,0.15)")

        # 조합 열에 수준별 배경색
        for level, bg in _LEVEL_BG.items():
            style_conditions.append({
                "if": {"column_id": "조합", "filter_query": f'{{조합}} contains "{level}"'},
                "backgroundColor": bg,
            })

        table_columns = [
            {"name": "조합", "id": "조합", "type": "text"},
            {"name": "쿼리", "id": "쿼리", "type": "text"},
            {"name": "평균유사도", "id": "평균유사도", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "오버랩", "id": "오버랩", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Spearman", "id": "Spearman", "type": "numeric", "format": {"specifier": ".4f"}},
        ]

        results_children = [
            html.H4("파라미터 기여도 순위", style={"marginBottom": "10px"}),
            dcc.Graph(figure=imp_fig),
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "12px", "marginBottom": "20px",
            }, children=imp_cards),
            dcc.Graph(figure=fig),
            html.H4("상세 결과 (상위 50건, 유사도 내림차순)", style={"marginTop": "20px"}),
            dash_table.DataTable(
                columns=table_columns,
                data=table_data,
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
                style_header={"backgroundColor": "#2c3e50", "color": "#fff"},
                style_data_conditional=style_conditions,
                page_size=20,
                sort_action="native",
            ),
        ]

        return (
            bar_style,
            f"분석 완료: {len(analysis)}개 조합-쿼리 결과",
            results_children,
            "",
            True,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)
