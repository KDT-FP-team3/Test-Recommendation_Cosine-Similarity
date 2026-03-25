"""
evaluator.py -- Stage 6: 정량적/정성적 평가 + 비교 분석
====================================================================
추천 결과를 다각도로 평가한다.

[정량적 평가]
  - 코사인 유사도 통계, 장르/키워드 정밀도, 다양성
  - 텍스트 일관성: Top-K 간 텍스트 임베딩 평균 유사도
  - 그룹별 기여도 분석

[정성적 평가]
  - 추천 장르 분포, 연도 범위, 국가 다양성
  - Top1 직관성, 공간 거리

[비교 분석]
  - 적합/부적합 판정 (임계값 기반)
"""

import numpy as np
from collections import Counter

from embedding import HybridMovieEmbedding
import config


class RecommendationEvaluator:
    """추천 결과를 정량/정성적으로 평가하고 비교 분석하는 클래스"""

    def __init__(self, embedding, train_movies, test_movies, coords=None):
        self.embedding = embedding
        self.train_data = {m["id"]: m for m in train_movies}
        self.test_data = {m["title"]: m for m in test_movies}
        self.test_id_map = {m["title"]: m["id"] for m in test_movies}
        self.coords = coords or {}

    # ==================================================================
    # 정량적 평가
    # ==================================================================

    def evaluate_quantitative(self, recommendations):
        """정량적 평가 수행."""
        metrics = {"per_query": {}, "overall": {}}

        all_sims = []
        all_genre_prec = []
        all_kw_prec = []
        all_diversity = []
        all_text_coherence = []

        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie or not recs:
                continue

            # 유사도 통계
            sims = [r["similarity"] for r in recs]
            all_sims.extend(sims)

            # 장르 Precision@K
            genre_hits = sum(
                1 for r in recs
                if set(query_movie.get("genres", [])) & set(r.get("genres", []))
            )
            genre_prec = genre_hits / len(recs) if recs else 0
            all_genre_prec.append(genre_prec)

            # 키워드 Precision@K
            kw_hits = sum(
                1 for r in recs if r.get("shared_keywords")
            )
            kw_prec = kw_hits / len(recs) if recs else 0
            all_kw_prec.append(kw_prec)

            # 다양성: 추천 영화 쌍 간 평균 비유사도
            diversity = self._compute_diversity(recs)
            all_diversity.append(diversity)

            # 텍스트 일관성: 추천 영화 간 텍스트 임베딩 유사도
            text_coh = self._compute_text_coherence(recs)
            all_text_coherence.append(text_coh)

            # 그룹별 평균 기여도
            group_avg = self._compute_group_contribution(recs)

            metrics["per_query"][query_title] = {
                "avg_similarity": round(np.mean(sims), 4),
                "max_similarity": round(np.max(sims), 4),
                "min_similarity": round(np.min(sims), 4),
                "std_similarity": round(float(np.std(sims)), 4),
                "genre_precision": round(genre_prec, 4),
                "keyword_precision": round(kw_prec, 4),
                "diversity": round(diversity, 4),
                "text_coherence": round(text_coh, 4),
                "group_contribution": group_avg,
            }

        if all_sims:
            metrics["overall"] = {
                "avg_similarity": round(float(np.mean(all_sims)), 4),
                "std_similarity": round(float(np.std(all_sims)), 4),
                "avg_genre_precision": round(float(np.mean(all_genre_prec)), 4),
                "avg_keyword_precision": round(float(np.mean(all_kw_prec)), 4),
                "avg_diversity": round(float(np.mean(all_diversity)), 4),
                "avg_text_coherence": round(float(np.mean(all_text_coherence)), 4),
            }

        return metrics

    def _compute_diversity(self, recs):
        """추천 결과 내 다양성 계산"""
        if len(recs) < 2:
            return 0.0
        ids = [r["id"] for r in recs if r.get("id") in self.embedding.raw_vectors]
        if len(ids) < 2:
            return 0.0
        vecs = np.array([self.embedding.raw_vectors[i] for i in ids])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        vecs_norm = vecs / norms
        sim_mat = vecs_norm @ vecs_norm.T
        n = len(ids)
        pair_count = 0
        total_dissim = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total_dissim += 1.0 - sim_mat[i, j]
                pair_count += 1
        return float(total_dissim / pair_count) if pair_count > 0 else 0.0

    def _compute_text_coherence(self, recs):
        """추천 영화 간 텍스트 임베딩 일관성"""
        ids = [r["id"] for r in recs if r.get("id") in self.embedding.group_vectors]
        if len(ids) < 2:
            return 0.0
        text_vecs = np.array([
            self.embedding.group_vectors[i]["text"] for i in ids
        ])
        norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        text_norm = text_vecs / norms
        sim_mat = text_norm @ text_norm.T
        n = len(ids)
        total_sim = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += sim_mat[i, j]
                pair_count += 1
        return float(total_sim / pair_count) if pair_count > 0 else 0.0

    def _compute_group_contribution(self, recs):
        """추천 결과의 그룹별 평균 유사도 기여도"""
        groups = {"genre": [], "keyword": [], "numeric": [], "text": []}
        for r in recs:
            gs = r.get("group_similarity", {})
            for g in groups:
                if g in gs:
                    groups[g].append(gs[g])
        return {g: round(float(np.mean(v)), 4) if v else 0.0 for g, v in groups.items()}

    # ==================================================================
    # 정성적 평가
    # ==================================================================

    def evaluate_qualitative(self, recommendations):
        """정성적 평가 수행."""
        qualitative = {}

        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie or not recs:
                continue

            analysis = {}

            # 추천 장르 분포
            rec_genres = []
            for r in recs:
                rec_genres.extend(r.get("genres", []))
            analysis["추천_장르_분포"] = dict(Counter(rec_genres).most_common(5))

            # 연도 범위
            years = [r.get("year", 0) for r in recs]
            analysis["추천_연도_범위"] = f"{min(years)}~{max(years)}"
            analysis["추천_평균_연도"] = round(float(np.mean(years)))

            # 국가 다양성
            nations = set()
            for r in recs:
                n = r.get("nation", "")
                if n:
                    nations.add(n)
            analysis["제작국가_다양성"] = list(nations)

            # 좌표 거리 분석
            query_id = self.test_id_map.get(query_title)
            q_coord = self.coords.get(query_id)
            distances = []
            if q_coord is not None:
                for r in recs:
                    r_coord = self.coords.get(r.get("id"))
                    if r_coord is not None:
                        dist = float(np.linalg.norm(q_coord - r_coord))
                        distances.append((r["title"], round(dist, 3)))
                distances.sort(key=lambda x: x[1])
            analysis["공간_거리"] = distances

            # Top1 직관성
            top_rec = recs[0]
            shared_g = set(query_movie.get("genres", [])) & set(top_rec.get("genres", []))
            shared_k = top_rec.get("shared_keywords", [])
            if shared_g and shared_k:
                intuitive = "높음 (장르+키워드 공통)"
            elif shared_g or shared_k:
                intuitive = "보통 (일부 특징만 공통)"
            else:
                intuitive = "낮음 (표면적 공통점 적음 - 잠재 패턴 유사)"
            analysis["Top1_직관성"] = intuitive

            qualitative[query_title] = analysis

        return qualitative

    # ==================================================================
    # 비교 분석
    # ==================================================================

    def classify_adequacy(self, quant, qual):
        """적합/부적합 판정."""
        adequacy = {}

        for query_title in quant.get("per_query", {}):
            q_metrics = quant["per_query"][query_title]
            q_qual = qual.get(query_title, {})
            reasons = []

            if q_metrics["genre_precision"] < config.THRESHOLD_GENRE_PRECISION:
                reasons.append(
                    f"장르 정밀도 낮음 ({q_metrics['genre_precision']:.2f} < "
                    f"{config.THRESHOLD_GENRE_PRECISION})"
                )
            if q_metrics["avg_similarity"] < config.THRESHOLD_AVG_SIMILARITY:
                reasons.append(
                    f"평균 유사도 낮음 ({q_metrics['avg_similarity']:.3f} < "
                    f"{config.THRESHOLD_AVG_SIMILARITY})"
                )
            distances = q_qual.get("공간_거리", [])
            if distances:
                avg_dist = np.mean([d[1] for d in distances])
                if avg_dist > config.THRESHOLD_3D_DISTANCE:
                    reasons.append(
                        f"공간 거리 큼 ({avg_dist:.2f} > "
                        f"{config.THRESHOLD_3D_DISTANCE})"
                    )
            intuitive = q_qual.get("Top1_직관성", "")
            if intuitive.startswith("낮음"):
                reasons.append("Top1 직관성 낮음")

            adequacy[query_title] = {
                "adequate": len(reasons) == 0,
                "reasons": reasons,
            }

        return adequacy

    def compare_metrics(self, quant, qual, recommendations):
        """정량/정성 비교 분석 결과 반환."""
        adequacy = self.classify_adequacy(quant, qual)

        genre_stats = {}
        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie:
                continue
            is_adequate = adequacy.get(query_title, {}).get("adequate", True)
            for genre in query_movie.get("genres", []):
                if genre not in genre_stats:
                    genre_stats[genre] = {"total": 0, "inadequate": 0}
                genre_stats[genre]["total"] += 1
                if not is_adequate:
                    genre_stats[genre]["inadequate"] += 1

        genre_rates = {}
        for genre, stats in genre_stats.items():
            rate = stats["inadequate"] / stats["total"] if stats["total"] else 0
            genre_rates[genre] = round(rate, 3)

        n_adequate = sum(1 for a in adequacy.values() if a["adequate"])
        n_total = len(adequacy)
        adequate_metrics = []
        inadequate_metrics = []
        for qt, a in adequacy.items():
            pq = quant["per_query"].get(qt, {})
            if a["adequate"]:
                adequate_metrics.append(pq)
            else:
                inadequate_metrics.append(pq)

        summary = {
            "total": n_total,
            "adequate": n_adequate,
            "inadequate": n_total - n_adequate,
            "adequate_avg_sim": round(
                float(np.mean([m.get("avg_similarity", 0) for m in adequate_metrics])), 4
            ) if adequate_metrics else 0,
            "inadequate_avg_sim": round(
                float(np.mean([m.get("avg_similarity", 0) for m in inadequate_metrics])), 4
            ) if inadequate_metrics else 0,
        }

        return {
            "adequacy": adequacy,
            "genre_rates": genre_rates,
            "summary": summary,
        }

    # ==================================================================
    # 콘솔 출력
    # ==================================================================

    def print_report(self, recommendations):
        """평가 전체 보고서를 콘솔에 출력하고 결과를 반환"""
        quant = self.evaluate_quantitative(recommendations)
        qual = self.evaluate_qualitative(recommendations)
        comparison = self.compare_metrics(quant, qual, recommendations)

        print("\n" + "=" * 72)
        print("정량적 평가 결과")
        print("=" * 72)

        if quant.get("overall"):
            print("\n> 전체 평균")
            for k, v in quant["overall"].items():
                print(f"    {k}: {v}")

        print("\n> 영화별 상세")
        for title, m in quant["per_query"].items():
            print(f"\n  {title}")
            for k, v in m.items():
                if k != "group_contribution":
                    print(f"      {k}: {v}")
                else:
                    gc = v
                    print(f"      그룹 기여도: 장르={gc.get('genre',0):.3f} "
                          f"키워드={gc.get('keyword',0):.3f} "
                          f"수치={gc.get('numeric',0):.3f} "
                          f"텍스트={gc.get('text',0):.3f}")

        print("\n" + "=" * 72)
        print("정성적 평가 결과")
        print("=" * 72)

        for title, analysis in qual.items():
            print(f"\n  {title}")
            for k, v in analysis.items():
                print(f"      {k}: {v}")

        print("\n" + "=" * 72)
        print("비교 분석 (적합/부적합 판정)")
        print("=" * 72)

        adequacy = comparison["adequacy"]
        summary = comparison["summary"]

        print(f"\n  전체: {summary['total']}편 중 "
              f"적합 {summary['adequate']}편 / "
              f"부적합 {summary['inadequate']}편")

        for title, result in adequacy.items():
            status = "적합" if result["adequate"] else "부적합"
            print(f"\n  [{status}] {title}")
            if result["reasons"]:
                for reason in result["reasons"]:
                    print(f"      - {reason}")

        print("\n" + "=" * 72)

        return quant, qual, comparison
