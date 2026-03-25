"""
recommender.py -- Stage 5: 코사인 유사도 기반 영화 추천
====================================================================
임베딩 벡터(54D)를 활용하여 입력 영화와 가장 유사한 기존 영화를 추천한다.
추천은 고차원 원본 벡터 간 코사인 유사도 기준으로 수행된다.
"""

import numpy as np

from embedding import MovieEmbedding
import config


class MovieRecommender:
    """코사인 유사도 기반 영화 추천기"""

    def __init__(self, embedding: MovieEmbedding,
                 train_movies: list[dict],
                 top_k: int = None):
        self.embedding = embedding
        self.train_movies = train_movies
        self.train_titles = [m["title"] for m in train_movies]
        self.train_data = {m["title"]: m for m in train_movies}
        self.top_k = top_k if top_k is not None else config.TOP_K

    def recommend(self, query_movie: dict,
                  top_k: int = None) -> list[dict]:
        """
        신작 영화에 대해 가장 유사한 기존 영화 Top-K 추천.

        Args:
            query_movie: 쿼리 영화 dict
            top_k: 추천 수 (None이면 self.top_k)

        Returns:
            [{"title", "similarity", "shared_genres",
              "shared_keywords", "year", "explanation"}, ...]
        """
        k = top_k or self.top_k
        query_title = query_movie["title"]

        # 배치 유사도 계산
        sim_matrix = self.embedding.compute_similarity_matrix(
            [query_title], self.train_titles
        )
        sim_row = sim_matrix[0]

        # 정렬
        sorted_indices = np.argsort(sim_row)[::-1][:k]

        results = []
        for idx in sorted_indices:
            title = self.train_titles[idx]
            sim = float(sim_row[idx])
            train_movie = self.train_data[title]

            shared_genres = list(
                set(query_movie["genres"]) & set(train_movie["genres"])
            )
            shared_kw = list(
                set(query_movie["keywords"]) & set(train_movie["keywords"])
            )
            explanation = self._generate_explanation(
                query_movie, train_movie, shared_genres, shared_kw
            )

            results.append({
                "rank": len(results) + 1,
                "title": title,
                "year": train_movie["year"],
                "genres": train_movie["genres"],
                "similarity": round(sim, 4),
                "shared_genres": shared_genres,
                "shared_keywords": shared_kw,
                "explanation": explanation,
            })

        return results

    def recommend_batch(self, test_movies: list[dict],
                        top_k: int = None) -> dict:
        """
        여러 테스트 영화에 대한 일괄 추천.

        Returns:
            {query_title: [recommendations]}
        """
        all_results = {}
        for movie in test_movies:
            all_results[movie["title"]] = self.recommend(movie, top_k)
        return all_results

    def _generate_explanation(self, query: dict, train: dict,
                              shared_genres: list, shared_kw: list) -> str:
        """추천 이유 텍스트 생성"""
        reasons = []
        if shared_genres:
            reasons.append(f"공통 장르: {', '.join(shared_genres)}")
        if shared_kw:
            reasons.append(f"공통 키워드: {', '.join(shared_kw)}")

        mood_diff = abs(query["mood"] - train["mood"])
        if mood_diff < 0.15:
            reasons.append("유사한 분위기/톤")

        tempo_diff = abs(query["tempo"] - train["tempo"])
        if tempo_diff < 0.15:
            reasons.append("유사한 전개 속도")

        if not reasons:
            reasons.append("전체적인 특징 패턴이 유사")

        return " | ".join(reasons)

    def print_recommendations(self, query_movie: dict,
                              top_k: int = None) -> list[dict]:
        """추천 결과를 콘솔에 출력"""
        results = self.recommend(query_movie, top_k)
        title = query_movie["title"]

        print("=" * 72)
        print(f"입력 영화: {title} ({query_movie['year']})")
        print(f"   장르: {', '.join(query_movie['genres'])}")
        print(f"   키워드: {', '.join(query_movie['keywords'])}")
        print(f"   분위기: {query_movie['mood']:.2f} | "
              f"속도: {query_movie['tempo']:.2f} | "
              f"시각 스타일: {query_movie['visual_style']:.2f}")
        print("-" * 72)
        print(f"추천 결과 (Top-{len(results)})")
        print("-" * 72)

        for r in results:
            print(f"  #{r['rank']}  {r['title']} ({r['year']})")
            print(f"      유사도: {r['similarity']:.4f}")
            print(f"      장르: {', '.join(r['genres'])}")
            print(f"      이유: {r['explanation']}")
            print()

        print("=" * 72)
        return results
