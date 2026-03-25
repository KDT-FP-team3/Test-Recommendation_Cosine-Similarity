"""
sensitivity.py -- 파라미터 민감도 분석 모듈 (27조합)
====================================================================
장르·키워드·수치 가중치를 상(1.5)/중(1.0)/하(0.5) 3단계로 조합하여
3^3 = 27가지 경우에 대해 Top-20 추천 목록이 어떻게 달라지는지 분석.

추가로 자유 텍스트 검색 5개 쿼리에 대해서도 동일 비교 수행.

기준선: (중,중,중) = (1.0, 1.0, 1.0)

사용:
    python main.py --sensitivity
"""

import itertools
import numpy as np
import config


# ═══════════════════════════════════════════════════════════════════
# 유틸리티: Spearman 순위 상관계수 (scipy 의존 없이)
# ═══════════════════════════════════════════════════════════════════

def _spearman_rho(ranks_a: list, ranks_b: list) -> float:
    n = len(ranks_a)
    if n < 2:
        return 0.0
    d = np.array(ranks_a, dtype=np.float64) - np.array(ranks_b, dtype=np.float64)
    return float(1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))


def _level_label(val: float) -> str:
    """가중치 값 → 상/중/하 라벨"""
    for name, v in config.SENSITIVITY_LEVELS.items():
        if abs(v - val) < 0.01:
            return name
    return f"{val:.1f}"


def _combo_label(wg: float, wk: float, wn: float) -> str:
    """조합 라벨: (중,상,하) 형태"""
    return f"({_level_label(wg)},{_level_label(wk)},{_level_label(wn)})"


# ═══════════════════════════════════════════════════════════════════
# 27조합 민감도 분석기
# ═══════════════════════════════════════════════════════════════════

class SensitivityAnalyzer:
    """
    장르·키워드·수치 가중치를 상/중/하로 조합한 27가지 경우에 대해
    기준선(중,중,중) 대비 추천 목록 변화를 분석.
    """

    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.SENSITIVITY_TOP_K
        self.levels = list(config.SENSITIVITY_LEVELS.values())  # [0.5, 1.0, 1.5]
        self.text_queries = getattr(config, "SENSITIVITY_TEXT_QUERIES", [])

    # ------------------------------------------------------------------
    # 27조합 생성
    # ------------------------------------------------------------------
    def _generate_combos(self) -> list[dict]:
        """3^3 = 27가지 (장르,키워드,수치) 조합 생성."""
        combos = []
        for wg, wk, wn in itertools.product(self.levels, repeat=3):
            combos.append({
                "weight_genre": wg,
                "weight_keyword": wk,
                "weight_numeric": wn,
                "label": _combo_label(wg, wk, wn),
            })
        return combos

    # ------------------------------------------------------------------
    # 메인 분석 실행
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        전체 민감도 분석 실행.

        Returns:
            {
                "combos": [{label, wg, wk, wn, recs, quant, text_recs}],
                "baseline_idx": int,
                "analysis": [{label, overlap, spearman, displacement, accuracy, reliability, ...}],
                "text_analysis": [{label, per_query: {query: {overlap, spearman, ...}}}],
                "rank_tables": {query_title: {baseline: [...], combos: {label: [...]}}}
            }
        """
        from main import run_pipeline

        combos = self._generate_combos()

        print("\n" + "=" * 72)
        print("파라미터 민감도 분석 (27조합: 상/중/하)")
        print(f"  레벨: 상={self.levels[2]}, 중={self.levels[1]}, 하={self.levels[0]}")
        print(f"  조합 수: {len(combos)}")
        print(f"  Top-K: {self.top_k}")
        if self.text_queries:
            print(f"  자유 검색 쿼리: {len(self.text_queries)}개")
        print("=" * 72)

        # ── 1) 모든 27조합 실행 ──
        results = []
        baseline_idx = None

        for i, combo in enumerate(combos):
            label = combo["label"]
            is_baseline = (combo["weight_genre"] == 1.0 and
                           combo["weight_keyword"] == 1.0 and
                           combo["weight_numeric"] == 1.0)
            if is_baseline:
                baseline_idx = i

            marker = " ◀ baseline" if is_baseline else ""
            print(f"\n  [{i+1:2d}/27] {label}{marker}")

            result = run_pipeline({
                "weight_genre": combo["weight_genre"],
                "weight_keyword": combo["weight_keyword"],
                "weight_numeric": combo["weight_numeric"],
                "top_k": self.top_k,
            })

            # 영화 제목 → 클러스터 ID 매핑
            cluster_map = {}
            if result.get("clusters") is not None:
                for idx, m in enumerate(result["train_movies"]):
                    cluster_map[m["title"]] = int(result["clusters"][idx])

            entry = {
                "label": label,
                "weight_genre": combo["weight_genre"],
                "weight_keyword": combo["weight_keyword"],
                "weight_numeric": combo["weight_numeric"],
                "recs": result["recommendations"],
                "quant": result["quant"],
                "embedding": result["embedding"],
                "train_movies": result["train_movies"],
                "cluster_map": cluster_map,
                "cluster_info": result.get("cluster_info", {}),
            }

            # 자유 텍스트 검색 (있으면)
            if self.text_queries:
                from search import MovieSearchEngine
                engine = MovieSearchEngine(result["embedding"], result["train_movies"])
                text_recs = {}
                for query_text in self.text_queries:
                    sr = engine.search_by_text(query_text, top_k=self.top_k)
                    text_recs[query_text] = sr.get("recommendations", [])
                entry["text_recs"] = text_recs

            results.append(entry)

        # ── 2) 기준선 대비 비교 분석 ──
        baseline = results[baseline_idx]
        analysis = []
        text_analysis = []

        for i, entry in enumerate(results):
            # 테스트 영화 비교
            cmp = self._compare_all_queries(baseline["recs"], entry["recs"])
            rel = self._compute_reliability(entry["quant"])
            overall = entry["quant"].get("overall", {})

            row = {
                "label": entry["label"],
                "weight_genre": entry["weight_genre"],
                "weight_keyword": entry["weight_keyword"],
                "weight_numeric": entry["weight_numeric"],
                "overlap_ratio": cmp["avg_overlap"],
                "spearman_rho": cmp["avg_spearman"],
                "avg_rank_displacement": cmp["avg_displacement"],
                "avg_similarity": overall.get("avg_similarity", 0),
                "genre_precision": overall.get("avg_genre_precision", 0),
                "keyword_precision": overall.get("avg_keyword_precision", 0),
                "reliability_sim": rel["std_similarity"],
                "is_baseline": (i == baseline_idx),
            }
            analysis.append(row)

            # 자유 텍스트 비교
            if self.text_queries and "text_recs" in entry:
                text_cmp = {}
                for query_text in self.text_queries:
                    base_list = baseline.get("text_recs", {}).get(query_text, [])
                    var_list = entry.get("text_recs", {}).get(query_text, [])
                    text_cmp[query_text] = self._compare_rankings(base_list, var_list)
                text_analysis.append({
                    "label": entry["label"],
                    "per_query": text_cmp,
                })

        # ── 3) 순위 비교 테이블 ──
        rank_tables = self._build_rank_tables(baseline, results)

        # ── 4) 요약 출력 ──
        self._print_summary(analysis)

        return {
            "combos": [{k: v for k, v in e.items()
                        if k not in ("embedding", "train_movies")}
                       for e in results],
            "baseline_idx": baseline_idx,
            "analysis": analysis,
            "text_analysis": text_analysis,
            "rank_tables": rank_tables,
            "cluster_info": baseline.get("cluster_info", {}),
        }

    # ------------------------------------------------------------------
    # 비교 함수들
    # ------------------------------------------------------------------
    def _compare_all_queries(self, baseline_recs: dict,
                             variant_recs: dict) -> dict:
        overlaps, spearmans, displacements = [], [], []
        for query_title in baseline_recs:
            if query_title not in variant_recs:
                continue
            cmp = self._compare_rankings(
                baseline_recs[query_title], variant_recs[query_title])
            overlaps.append(cmp["overlap_ratio"])
            spearmans.append(cmp["spearman_rho"])
            displacements.append(cmp["avg_rank_displacement"])

        return {
            "avg_overlap": round(float(np.mean(overlaps)), 4) if overlaps else 0.0,
            "avg_spearman": round(float(np.mean(spearmans)), 4) if spearmans else 0.0,
            "avg_displacement": round(float(np.mean(displacements)), 4) if displacements else 0.0,
        }

    def _compare_rankings(self, base_recs: list, var_recs: list) -> dict:
        base_titles = [r["title"] for r in base_recs]
        var_titles = [r["title"] for r in var_recs]
        base_set, var_set = set(base_titles), set(var_titles)

        common = base_set & var_set
        overlap_ratio = len(common) / max(len(base_titles), 1)

        if len(common) >= 2:
            # 공통 영화를 기준선 순서로 정렬 후 1~n 재순위 부여
            base_order = [t for t in base_titles if t in common]
            var_order = [t for t in var_titles if t in common]
            # 재순위: base_order 기준 1~n
            base_rerank = {t: i + 1 for i, t in enumerate(base_order)}
            var_rerank = {t: i + 1 for i, t in enumerate(var_order)}
            common_list = base_order  # 기준선 순서대로
            r_base = [base_rerank[t] for t in common_list]
            r_var = [var_rerank[t] for t in common_list]
            spearman = _spearman_rho(r_base, r_var)
            # 순위 변동은 원래 순위 기준
            orig_base = {t: i + 1 for i, t in enumerate(base_titles)}
            orig_var = {t: i + 1 for i, t in enumerate(var_titles)}
            displacement = float(np.mean([abs(orig_base[t] - orig_var[t])
                                          for t in common_list]))
        else:
            spearman, displacement = 0.0, 0.0

        return {
            "overlap_ratio": round(overlap_ratio, 4),
            "spearman_rho": round(spearman, 4),
            "avg_rank_displacement": round(displacement, 2),
            "new_entries": list(var_set - base_set),
            "dropped_entries": list(base_set - var_set),
        }

    def _compute_reliability(self, quant: dict) -> dict:
        per_query = quant.get("per_query", {})
        if not per_query:
            return {"std_similarity": 0.0, "std_genre_precision": 0.0}
        sims = [v["avg_similarity"] for v in per_query.values()]
        return {
            "std_similarity": round(float(np.std(sims)), 4) if sims else 0.0,
            "std_genre_precision": 0.0,
        }

    # ------------------------------------------------------------------
    # 순위 비교 테이블
    # ------------------------------------------------------------------
    def _build_rank_tables(self, baseline: dict, all_results: list) -> dict:
        tables = {}
        base_recs = baseline["recs"]
        base_cmap = baseline.get("cluster_map", {})

        for query_title in base_recs:
            base_list = base_recs[query_title]
            base_rank_map = {r["title"]: r["rank"] for r in base_list}

            table = {
                "baseline": [
                    (r["rank"], r["title"], r["similarity"],
                     base_cmap.get(r["title"], -1))
                    for r in base_list
                ],
                "combos": {},
            }

            # 가장 유사도가 높은 조합과 가장 낮은 조합
            best_sim, worst_sim = 0, 999
            best_entry, worst_entry = None, None

            for entry in all_results:
                if entry["label"] == baseline["label"]:
                    continue
                if query_title not in entry["recs"]:
                    continue
                avg_s = np.mean([r["similarity"] for r in entry["recs"][query_title]])
                if avg_s > best_sim:
                    best_sim = avg_s
                    best_entry = entry
                if avg_s < worst_sim:
                    worst_sim = avg_s
                    worst_entry = entry

            for tag, entry in [("best", best_entry), ("worst", worst_entry)]:
                if entry is None:
                    continue
                cmap = entry.get("cluster_map", {})
                rows = []
                for r in entry["recs"][query_title][:20]:
                    old_rank = base_rank_map.get(r["title"])
                    change = (old_rank - r["rank"]) if old_rank else "NEW"
                    cluster_id = cmap.get(r["title"], -1)
                    rows.append((r["rank"], r["title"], r["similarity"],
                                 change, entry["label"], cluster_id))
                table["combos"][tag] = rows

            tables[query_title] = table

        return tables

    # ------------------------------------------------------------------
    # 요약 출력
    # ------------------------------------------------------------------
    def _print_summary(self, analysis: list):
        print("\n" + "=" * 72)
        print("민감도 분석 결과 요약 (27조합)")
        print("=" * 72)
        print(f"  {'조합':>14}  {'겹침률':>7}  {'순위상관':>8}  {'순위변동':>8}  "
              f"{'유사도':>7}  {'장르정밀':>8}  {'신뢰도σ':>8}")
        print(f"  {'─' * 70}")

        for d in sorted(analysis, key=lambda x: -x["avg_similarity"]):
            marker = " ◀" if d["is_baseline"] else ""
            print(f"  {d['label']:>14}  "
                  f"{d['overlap_ratio']:7.1%}  "
                  f"{d['spearman_rho']:8.4f}  "
                  f"{d['avg_rank_displacement']:8.2f}  "
                  f"{d['avg_similarity']:7.4f}  "
                  f"{d['genre_precision']:8.4f}  "
                  f"{d['reliability_sim']:8.4f}"
                  f"{marker}")

        print("=" * 72)
