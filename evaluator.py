"""
evaluator.py -- Stage 6: 정량적/정성적 평가 + 비교 분석
====================================================================
추천 결과를 다각도로 평가한다.

[정량적 평가]
  - 코사인 유사도 통계 (평균, 표준편차, 최대/최소)
  - Genre Precision@K: Top-K 중 장르 일치 비율
  - Keyword Precision@K: Top-K 중 키워드 일치 비율
  - Diversity: 추천 결과 내 영화들 간 평균 비유사도

[정성적 평가]
  - 추천 장르 분포
  - 연도 범위
  - 국가 다양성
  - 3D 공간 거리
  - Top1 직관성

[비교 분석]
  - 적합/부적합 판정 (임계값 기반)
  - 장르별 부적합 비율
"""

import numpy as np
from collections import Counter

from embedding import MovieEmbedding
import config


class RecommendationEvaluator:
    """추천 결과를 정량/정성적으로 평가하고 비교 분석하는 클래스"""

    def __init__(self,
                 embedding: MovieEmbedding,
                 train_movies: list[dict],
                 test_movies: list[dict],
                 coords: dict = None):
        """
        Args:
            embedding: 학습 완료된 MovieEmbedding
            train_movies: 학습 영화 리스트
            test_movies: 테스트 영화 리스트
            coords: {title: np.ndarray} 2D/3D 좌표 (정성적 평가에 사용)
        """
        self.embedding = embedding
        self.train_data = {m["title"]: m for m in train_movies}
        self.test_data = {m["title"]: m for m in test_movies}
        self.coords = coords or {}

    # ══════════════════════════════════════════════════════════════
    # 정량적 평가
    # ══════════════════════════════════════════════════════════════

    def evaluate_quantitative(self, recommendations: dict) -> dict:
        """
        정량적 평가 수행.

        Args:
            recommendations: {query_title: [rec_dicts]}

        Returns:
            {"per_query": {...}, "overall": {...}}
        """
        metrics = {"per_query": {}, "overall": {}}

        all_sims = []
        all_genre_prec = []
        all_kw_prec = []
        all_diversity = []

        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie:
                continue

            # 유사도 통계
            sims = [r["similarity"] for r in recs]
            all_sims.extend(sims)

            # 장르 Precision@K
            genre_hits = sum(
                1 for r in recs
                if set(query_movie["genres"]) & set(r["genres"])
            )
            genre_prec = genre_hits / len(recs) if recs else 0
            all_genre_prec.append(genre_prec)

            # 키워드 Precision@K
            kw_hits = 0
            for r in recs:
                train_m = self.train_data.get(r["title"])
                if train_m:
                    shared_kw = set(query_movie["keywords"]) & set(train_m["keywords"])
                    if shared_kw:
                        kw_hits += 1
            kw_prec = kw_hits / len(recs) if recs else 0
            all_kw_prec.append(kw_prec)

            # 다양성
            if len(recs) > 1:
                pair_dissims = []
                for i in range(len(recs)):
                    for j in range(i + 1, len(recs)):
                        sim_ij = self.embedding.compute_similarity(
                            recs[i]["title"], recs[j]["title"]
                        )
                        pair_dissims.append(1 - sim_ij)
                diversity = float(np.mean(pair_dissims))
            else:
                diversity = 0.0
            all_diversity.append(diversity)

            metrics["per_query"][query_title] = {
                "avg_similarity": round(np.mean(sims), 4),
                "max_similarity": round(np.max(sims), 4),
                "min_similarity": round(np.min(sims), 4),
                "genre_precision": round(genre_prec, 4),
                "keyword_precision": round(kw_prec, 4),
                "diversity": round(diversity, 4),
            }

        if all_sims:
            metrics["overall"] = {
                "avg_similarity": round(float(np.mean(all_sims)), 4),
                "std_similarity": round(float(np.std(all_sims)), 4),
                "avg_genre_precision": round(float(np.mean(all_genre_prec)), 4),
                "avg_keyword_precision": round(float(np.mean(all_kw_prec)), 4),
                "avg_diversity": round(float(np.mean(all_diversity)), 4),
            }

        return metrics

    # ══════════════════════════════════════════════════════════════
    # 정성적 평가
    # ══════════════════════════════════════════════════════════════

    def evaluate_qualitative(self, recommendations: dict) -> dict:
        """
        정성적 평가 수행.

        Returns:
            {query_title: {분석 결과}}
        """
        qualitative = {}

        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie:
                continue

            analysis = {}

            # 추천 장르 분포
            rec_genres = []
            for r in recs:
                rec_genres.extend(r["genres"])
            analysis["추천_장르_분포"] = dict(Counter(rec_genres).most_common(5))

            # 연도 범위
            years = [r["year"] for r in recs]
            analysis["추천_연도_범위"] = f"{min(years)}~{max(years)}"
            analysis["추천_평균_연도"] = round(float(np.mean(years)))

            # 국가 다양성
            origins = set()
            for r in recs:
                train_m = self.train_data.get(r["title"])
                if train_m:
                    origins.add(train_m["origin"])
            analysis["제작국가_다양성"] = list(origins)

            # 좌표 거리 분석
            q_coord = self.coords.get(query_title)
            distances = []
            if q_coord is not None:
                for r in recs:
                    r_coord = self.coords.get(r["title"])
                    if r_coord is not None:
                        dist = float(np.linalg.norm(q_coord - r_coord))
                        distances.append((r["title"], round(dist, 3)))
                distances.sort(key=lambda x: x[1])
            analysis["공간_거리"] = distances

            # Top1 직관성
            top_rec = recs[0]
            shared_g = set(query_movie["genres"]) & set(top_rec["genres"])
            train_top = self.train_data.get(top_rec["title"], {})
            shared_k = set(query_movie["keywords"]) & set(
                train_top.get("keywords", [])
            )
            if shared_g and shared_k:
                intuitive = "높음 (장르+키워드 공통)"
            elif shared_g or shared_k:
                intuitive = "보통 (일부 특징만 공통)"
            else:
                intuitive = "낮음 (표면적 공통점 적음 - 잠재 패턴 유사)"
            analysis["Top1_직관성"] = intuitive

            qualitative[query_title] = analysis

        return qualitative

    # ══════════════════════════════════════════════════════════════
    # 비교 분석 (적합/부적합 판정)
    # ══════════════════════════════════════════════════════════════

    def classify_adequacy(self, quant: dict, qual: dict) -> dict:
        """
        각 테스트 영화 추천의 적합/부적합 판정.

        판정 기준 (하나라도 해당하면 "부적합"):
          - genre_precision < THRESHOLD_GENRE_PRECISION
          - avg_similarity < THRESHOLD_AVG_SIMILARITY
          - 평균 공간 거리 > THRESHOLD_3D_DISTANCE
          - Top1_직관성이 "낮음"으로 시작

        Returns:
            {query_title: {"adequate": bool, "reasons": [str]}}
        """
        adequacy = {}

        for query_title in quant.get("per_query", {}):
            q_metrics = quant["per_query"][query_title]
            q_qual = qual.get(query_title, {})

            reasons = []

            # 장르 정밀도 확인
            if q_metrics["genre_precision"] < config.THRESHOLD_GENRE_PRECISION:
                reasons.append(
                    f"장르 정밀도 낮음 ({q_metrics['genre_precision']:.2f} < "
                    f"{config.THRESHOLD_GENRE_PRECISION})"
                )

            # 평균 유사도 확인
            if q_metrics["avg_similarity"] < config.THRESHOLD_AVG_SIMILARITY:
                reasons.append(
                    f"평균 유사도 낮음 ({q_metrics['avg_similarity']:.3f} < "
                    f"{config.THRESHOLD_AVG_SIMILARITY})"
                )

            # 공간 거리 확인
            distances = q_qual.get("공간_거리", [])
            if distances:
                avg_dist = np.mean([d[1] for d in distances])
                if avg_dist > config.THRESHOLD_3D_DISTANCE:
                    reasons.append(
                        f"공간 거리 큼 ({avg_dist:.2f} > "
                        f"{config.THRESHOLD_3D_DISTANCE})"
                    )

            # 직관성 확인
            intuitive = q_qual.get("Top1_직관성", "")
            if intuitive.startswith("낮음"):
                reasons.append("Top1 직관성 낮음")

            adequacy[query_title] = {
                "adequate": len(reasons) == 0,
                "reasons": reasons,
            }

        return adequacy

    def compare_metrics(self, quant: dict, qual: dict,
                        recommendations: dict) -> dict:
        """
        정량/정성 비교 분석 결과 반환.

        Returns:
            {"adequacy": {...}, "genre_rates": {...}, "summary": {...}}
        """
        adequacy = self.classify_adequacy(quant, qual)

        # 장르별 부적합 비율
        genre_stats = {}
        for query_title, recs in recommendations.items():
            query_movie = self.test_data.get(query_title)
            if not query_movie:
                continue
            is_adequate = adequacy.get(query_title, {}).get("adequate", True)
            for genre in query_movie["genres"]:
                if genre not in genre_stats:
                    genre_stats[genre] = {"total": 0, "inadequate": 0}
                genre_stats[genre]["total"] += 1
                if not is_adequate:
                    genre_stats[genre]["inadequate"] += 1

        genre_rates = {}
        for genre, stats in genre_stats.items():
            rate = stats["inadequate"] / stats["total"] if stats["total"] else 0
            genre_rates[genre] = round(rate, 3)

        # 전체 요약
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
                float(np.mean([m["avg_similarity"] for m in adequate_metrics])), 4
            ) if adequate_metrics else 0,
            "inadequate_avg_sim": round(
                float(np.mean([m["avg_similarity"] for m in inadequate_metrics])), 4
            ) if inadequate_metrics else 0,
        }

        return {
            "adequacy": adequacy,
            "genre_rates": genre_rates,
            "summary": summary,
        }

    # ══════════════════════════════════════════════════════════════
    # 콘솔 출력
    # ══════════════════════════════════════════════════════════════

    def print_report(self, recommendations: dict):
        """평가 전체 보고서를 콘솔에 출력하고 결과를 반환"""

        quant = self.evaluate_quantitative(recommendations)
        qual = self.evaluate_qualitative(recommendations)
        comparison = self.compare_metrics(quant, qual, recommendations)

        # 정량적 결과
        print("\n" + "=" * 72)
        print("정량적 평가 결과 (Quantitative)")
        print("=" * 72)

        if quant.get("overall"):
            print("\n> 전체 평균")
            for k, v in quant["overall"].items():
                print(f"    {k}: {v}")

        print("\n> 영화별 상세")
        for title, m in quant["per_query"].items():
            print(f"\n  {title}")
            for k, v in m.items():
                print(f"      {k}: {v}")

        # 정성적 결과
        print("\n" + "=" * 72)
        print("정성적 평가 결과 (Qualitative)")
        print("=" * 72)

        for title, analysis in qual.items():
            print(f"\n  {title}")
            for k, v in analysis.items():
                print(f"      {k}: {v}")

        # 비교 분석
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
