"""
recommender.py -- Stage 5: 코사인 유사도 기반 영화 추천
====================================================================
하이브리드 임베딩 벡터(499D)를 활용하여 입력 영화와 가장 유사한
기존 영화를 Top-K 추천한다.
각 추천에 대해 그룹별 유사도 기여도를 분해하여 설명을 생성한다.
"""

import numpy as np

from embedding import HybridMovieEmbedding
import config


class MovieRecommender:
    """코사인 유사도 기반 영화 추천기"""

    def __init__(self, embedding, train_movies, top_k=None):
        """
        Parameters
        ----------
        embedding : HybridMovieEmbedding
        train_movies : list[dict]
        top_k : int, optional
        """
        self.embedding = embedding
        self.train_movies = train_movies
        self.train_ids = [m["id"] for m in train_movies]
        self.train_data = {m["id"]: m for m in train_movies}
        self.top_k = top_k if top_k is not None else config.TOP_K

    def recommend(self, query_movie, top_k=None):
        """
        영화에 대해 가장 유사한 기존 영화 Top-K 추천.

        Parameters
        ----------
        query_movie : dict
        top_k : int, optional

        Returns
        -------
        list[dict]
        """
        k = top_k or self.top_k
        query_id = query_movie["id"]

        query_vec = self.embedding.get_raw_vector(query_id)
        if query_vec is None:
            return []

        ranked = self.embedding.compute_similarity_to_train(
            query_vec, self.train_ids
        )

        results = []
        for mid, sim in ranked:
            if mid == query_id:
                continue
            if mid not in self.train_data:
                continue

            train_movie = self.train_data[mid]

            shared_genres = list(
                set(query_movie.get("genres", [])) & set(train_movie.get("genres", []))
            )
            shared_kw = list(
                set(query_movie.get("keywords_matched", []))
                & set(train_movie.get("keywords_matched", []))
            )

            # 그룹별 유사도 분해
            group_sim = self.embedding.compute_group_similarity(query_id, mid)

            explanation = self._generate_explanation(
                query_movie, train_movie, shared_genres, shared_kw, group_sim
            )

            results.append({
                "rank": len(results) + 1,
                "id": mid,
                "title": train_movie["title"],
                "title_eng": train_movie.get("title_eng", ""),
                "year": train_movie["year"],
                "genres": train_movie.get("genres", []),
                "similarity": round(sim, 4),
                "shared_genres": shared_genres,
                "shared_keywords": shared_kw,
                "group_similarity": group_sim,
                "nation": train_movie.get("nation", ""),
                "directors": train_movie.get("directors", []),
                "poster_path": train_movie.get("poster_path"),
                "explanation": explanation,
            })
            if len(results) >= k:
                break

        return results

    def recommend_batch(self, test_movies, top_k=None):
        """여러 테스트 영화에 대한 일괄 추천."""
        all_results = {}
        for movie in test_movies:
            all_results[movie["title"]] = self.recommend(movie, top_k)
        return all_results

    def _generate_explanation(self, query, train, shared_genres, shared_kw, group_sim):
        """추천 이유 텍스트 생성"""
        reasons = []
        if shared_genres:
            reasons.append(f"공통 장르: {', '.join(shared_genres[:3])}")
        if shared_kw:
            reasons.append(f"공통 키워드: {', '.join(shared_kw[:3])}")

        # 그룹별 기여도 표시
        top_group = max(
            [(g, s) for g, s in group_sim.items() if g != "total"],
            key=lambda x: x[1],
            default=("", 0)
        )
        group_names = {
            "genre": "장르", "keyword": "키워드",
            "numeric": "수치", "text": "줄거리"
        }
        if top_group[1] > 0.3:
            gname = group_names.get(top_group[0], top_group[0])
            reasons.append(f"주요 유사 요인: {gname} ({top_group[1]:.2f})")

        if not reasons:
            reasons.append("전체적인 특징 패턴이 유사")

        return " | ".join(reasons)

    def print_recommendations(self, query_movie, top_k=None):
        """추천 결과를 콘솔에 출력"""
        results = self.recommend(query_movie, top_k)
        title = query_movie["title"]

        print("=" * 72)
        print(f"입력 영화: {title} ({query_movie['year']})")
        print(f"  장르: {', '.join(query_movie.get('genres', []))}")
        kw = query_movie.get("keywords_matched", [])
        print(f"  키워드: {', '.join(kw[:5])}")
        print("-" * 72)
        print(f"추천 결과 (Top-{len(results)})")
        print("-" * 72)

        for r in results:
            print(f"  #{r['rank']:2d}  {r['title']} ({r['year']})")
            print(f"       유사도: {r['similarity']:.4f}")
            print(f"       장르: {', '.join(r['genres'])}")
            gs = r.get("group_similarity", {})
            print(f"       [장르:{gs.get('genre',0):.2f} 키워드:{gs.get('keyword',0):.2f} "
                  f"수치:{gs.get('numeric',0):.2f} 텍스트:{gs.get('text',0):.2f}]")
            print(f"       이유: {r['explanation']}")
            print()

        print("=" * 72)
        return results
