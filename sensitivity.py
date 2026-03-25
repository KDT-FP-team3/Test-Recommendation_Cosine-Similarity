"""
sensitivity.py -- 파라미터 민감도 분석 모듈 (27조합)
====================================================================
장르/키워드/텍스트 가중치를 상(1.5)/중(1.0)/하(0.5) 3단계로 조합하여
3^3 = 27가지 경우에 대해 Top-20 추천 목록 변화를 분석.
수치 가중치는 0.5로 고정.

기준선: (중,중,중) = (1.0, 1.0, 1.0)
"""

import itertools
import numpy as np
import config

from embedding import HybridMovieEmbedding
from search import MovieSearchEngine


# ===================================================================
# 유틸리티
# ===================================================================

def _spearman_rho(ranks_a, ranks_b):
    n = len(ranks_a)
    if n < 2:
        return 0.0
    d = np.array(ranks_a, dtype=np.float64) - np.array(ranks_b, dtype=np.float64)
    return float(1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))


def _level_label(val):
    for name, v in config.SENSITIVITY_LEVELS.items():
        if abs(v - val) < 0.01:
            return name
    return f"{val:.1f}"


def _combo_label(wg, wk, wt):
    return f"({_level_label(wg)},{_level_label(wk)},{_level_label(wt)})"


# ===================================================================
# 민감도 분석기
# ===================================================================

class SensitivityAnalyzer:
    """
    장르/키워드/텍스트 가중치를 상/중/하로 조합한 27가지 경우에 대해
    기준선(중,중,중) 대비 추천 목록 변화를 분석.
    """

    def __init__(self, top_k=None):
        self.top_k = top_k or config.SENSITIVITY_TOP_K
        self.levels = list(config.SENSITIVITY_LEVELS.values())
        self.text_queries = getattr(config, "SENSITIVITY_TEXT_QUERIES", [])

    def analyze_movies(self, embedding, train_movies, test_movies,
                        progress_callback=None):
        """
        테스트 영화에 대한 27조합 민감도 분석.

        Parameters
        ----------
        progress_callback : callable, optional
            (current, total) 형태로 진행 상황을 전달하는 콜백.

        Returns
        -------
        all_results : dict
            {combo_label: {query_title: [(movie_id, similarity), ...]}}
        analysis : list[dict]
            각 항목에 combo, wg, wk, wt, query, overlap, spearman_rho,
            avg_displacement, avg_similarity, genre_precision 포함
        """
        combos = list(itertools.product(self.levels, repeat=3))
        total = len(combos)
        baseline_wg, baseline_wk, baseline_wt = 1.0, 1.0, 1.0

        # 테스트 영화의 장르 조회용 매핑
        test_genres = {m["title"]: set(m.get("genres", [])) for m in test_movies}
        train_data = {m["id"]: m for m in train_movies}

        # 기준선 결과  -- (id, sim) 튜플 리스트
        baseline_results = self._run_combo(
            embedding, train_movies, test_movies,
            baseline_wg, baseline_wk, baseline_wt
        )

        all_results = {}
        analysis = []

        for idx, (wg, wk, wt) in enumerate(combos):
            if progress_callback:
                progress_callback(idx + 1, total)
            label = _combo_label(wg, wk, wt)
            combo_results = self._run_combo(
                embedding, train_movies, test_movies, wg, wk, wt
            )
            all_results[label] = combo_results

            # 비교 지표 계산
            for query_title in baseline_results:
                base_items = baseline_results[query_title]
                combo_items = combo_results.get(query_title, [])

                base_ids = [mid for mid, _ in base_items]
                combo_ids = [mid for mid, _ in combo_items]
                combo_sims = [sim for _, sim in combo_items]

                overlap = len(set(base_ids) & set(combo_ids)) / self.top_k if base_ids else 0

                # Spearman 순위 상관
                common = set(base_ids) & set(combo_ids)
                if len(common) >= 2:
                    base_ranks = [base_ids.index(x) for x in common]
                    combo_ranks = [combo_ids.index(x) for x in common]
                    rho = _spearman_rho(base_ranks, combo_ranks)
                else:
                    rho = 0.0

                # 순위 변위
                displacement = 0
                for mid in base_ids:
                    if mid in combo_ids:
                        displacement += abs(base_ids.index(mid) - combo_ids.index(mid))
                    else:
                        displacement += self.top_k
                avg_displacement = displacement / self.top_k if self.top_k > 0 else 0

                # 평균 유사도
                avg_similarity = float(np.mean(combo_sims)) if combo_sims else 0.0

                # 장르 정밀도: Top-20 중 쿼리와 장르 1개 이상 공유하는 비율
                query_genres = test_genres.get(query_title, set())
                if combo_ids and query_genres:
                    genre_hits = 0
                    for mid in combo_ids:
                        rec_movie = train_data.get(mid)
                        if rec_movie:
                            rec_genres = set(rec_movie.get("genres", []))
                            if query_genres & rec_genres:
                                genre_hits += 1
                    genre_precision = genre_hits / len(combo_ids)
                else:
                    genre_precision = 0.0

                analysis.append({
                    "combo": label,
                    "wg": wg, "wk": wk, "wt": wt,
                    "query": query_title,
                    "overlap": round(overlap, 4),
                    "spearman_rho": round(rho, 4),
                    "avg_displacement": round(avg_displacement, 2),
                    "avg_similarity": round(avg_similarity, 4),
                    "genre_precision": round(genre_precision, 4),
                })

        return all_results, analysis

    def analyze_text_queries(self, embedding, train_movies):
        """
        자유 텍스트 쿼리에 대한 27조합 민감도 분석.
        """
        if not self.text_queries:
            return {}, []

        combos = list(itertools.product(self.levels, repeat=3))
        baseline_wg, baseline_wk, baseline_wt = 1.0, 1.0, 1.0

        # 기준선
        baseline_results = self._run_text_combo(
            embedding, train_movies, baseline_wg, baseline_wk, baseline_wt
        )

        all_results = {}
        analysis = []

        for wg, wk, wt in combos:
            label = _combo_label(wg, wk, wt)
            combo_results = self._run_text_combo(
                embedding, train_movies, wg, wk, wt
            )
            all_results[label] = combo_results

            for query_text in self.text_queries:
                base_items = baseline_results.get(query_text, [])
                combo_items = combo_results.get(query_text, [])

                base_ids = [mid for mid, _ in base_items]
                combo_ids = [mid for mid, _ in combo_items]
                combo_sims = [sim for _, sim in combo_items]

                overlap = len(set(base_ids) & set(combo_ids)) / self.top_k if base_ids else 0

                common = set(base_ids) & set(combo_ids)
                if len(common) >= 2:
                    base_ranks = [base_ids.index(x) for x in common]
                    combo_ranks = [combo_ids.index(x) for x in common]
                    rho = _spearman_rho(base_ranks, combo_ranks)
                else:
                    rho = 0.0

                avg_similarity = float(np.mean(combo_sims)) if combo_sims else 0.0

                analysis.append({
                    "combo": label,
                    "wg": wg, "wk": wk, "wt": wt,
                    "query": query_text[:30],
                    "overlap": round(overlap, 4),
                    "spearman_rho": round(rho, 4),
                    "avg_similarity": round(avg_similarity, 4),
                })

        return all_results, analysis

    def _run_combo(self, embedding, train_movies, test_movies, wg, wk, wt):
        """
        특정 가중치 조합으로 추천 실행.

        Returns
        -------
        dict : {query_title: [(movie_id, similarity), ...]}
        """
        new_raw = embedding.rebuild_with_weights(
            w_genre=wg, w_keyword=wk, w_numeric=0.5, w_text=wt
        )

        # 임시 raw_vectors 교체
        old_raw = embedding.raw_vectors
        embedding.raw_vectors = new_raw

        train_ids = [m["id"] for m in train_movies]
        results = {}

        for test_movie in test_movies:
            test_id = test_movie["id"]
            if test_id not in new_raw:
                continue
            query_vec = new_raw[test_id]
            ranked = embedding.compute_similarity_to_train(query_vec, train_ids)
            top_items = [(mid, sim) for mid, sim in ranked if mid != test_id][:self.top_k]
            results[test_movie["title"]] = top_items

        # 복원
        embedding.raw_vectors = old_raw
        return results

    def _run_text_combo(self, embedding, train_movies, wg, wk, wt):
        """텍스트 쿼리에 대한 특정 가중치 조합 실행."""
        new_raw = embedding.rebuild_with_weights(
            w_genre=wg, w_keyword=wk, w_numeric=0.5, w_text=wt
        )

        old_raw = embedding.raw_vectors
        embedding.raw_vectors = new_raw

        train_ids = [m["id"] for m in train_movies]
        results = {}

        for query_text in self.text_queries:
            query_vec = embedding.build_query_vector(text=query_text)
            ranked = embedding.compute_similarity_to_train(query_vec, train_ids)
            top_items = [(mid, sim) for mid, sim in ranked][:self.top_k]
            results[query_text] = top_items

        embedding.raw_vectors = old_raw
        return results

    def print_analysis(self, analysis):
        """분석 결과를 콘솔에 출력."""
        print("\n" + "=" * 72)
        print("민감도 분석 결과 (27조합) -- 조합 순서: (장르, 키워드, 텍스트)")
        print("=" * 72)

        by_query = {}
        for item in analysis:
            q = item["query"]
            if q not in by_query:
                by_query[q] = []
            by_query[q].append(item)

        for query, items in by_query.items():
            print(f"\n  쿼리: {query}")
            print(f"  {'조합':<40} {'오버랩':>8} {'Spearman':>10}")
            print(f"  {'-'*58}")
            for item in items:
                print(f"  {item['combo']:<40} {item['overlap']:>8.2%} "
                      f"{item['spearman_rho']:>10.4f}")
