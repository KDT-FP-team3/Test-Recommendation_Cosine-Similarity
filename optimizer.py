"""
optimizer.py -- 시뮬레이티드 어닐링 기반 가중치 최적화
====================================================================
테스트 영화 데이터셋에 대한 평가 메트릭을 최적화하는 가중치 조합을 탐색.

목적함수:
  score = 0.35*avg_similarity + 0.30*genre_precision
        + 0.25*text_coherence - 0.10*|diversity - 0.3|

탐색 전략:
  1. 초기값: 현재 가중치
  2. 이웃 생성: 각 가중치를 ±step 범위에서 무작위 perturbation
  3. 시뮬레이티드 어닐링: 온도 감소에 따라 악화 해 수용 확률 감소
  4. 조기 종료: 연속 patience회 개선 없으면 종료
"""

import numpy as np

from embedding import HybridMovieEmbedding
from recommender import MovieRecommender
from evaluator import RecommendationEvaluator
import config


class WeightOptimizer:
    """시뮬레이티드 어닐링 기반 가중치 최적화기"""

    def __init__(self, embedding, train_movies, test_movies, coords=None):
        self.embedding = embedding
        self.train_movies = train_movies
        self.test_movies = test_movies
        self.coords = coords or {}

    def _evaluate_weights(self, wg, wk, wn, wt):
        """주어진 가중치로 추천 후 평가 메트릭을 계산한다."""
        new_raw = self.embedding.rebuild_with_weights(
            w_genre=wg, w_keyword=wk, w_numeric=wn, w_text=wt
        )

        old_raw = self.embedding.raw_vectors
        self.embedding.raw_vectors = new_raw

        recommender = MovieRecommender(
            self.embedding, self.train_movies, top_k=config.TOP_K
        )
        recommendations = recommender.recommend_batch(self.test_movies)

        evaluator = RecommendationEvaluator(
            self.embedding, self.train_movies, self.test_movies, self.coords
        )
        quant = evaluator.evaluate_quantitative(recommendations)

        self.embedding.raw_vectors = old_raw

        overall = quant.get("overall", {})
        avg_sim = overall.get("avg_similarity", 0)
        genre_prec = overall.get("avg_genre_precision", 0)
        text_coh = overall.get("avg_text_coherence", 0)
        diversity = overall.get("avg_diversity", 0)

        diversity_penalty = abs(diversity - 0.3)
        score = (0.35 * avg_sim
                 + 0.30 * genre_prec
                 + 0.25 * text_coh
                 - 0.10 * diversity_penalty)

        return {
            "avg_similarity": round(avg_sim, 4),
            "genre_precision": round(genre_prec, 4),
            "text_coherence": round(text_coh, 4),
            "diversity": round(diversity, 4),
            "composite_score": round(score, 4),
        }

    def optimize(self,
                 initial_weights=None,
                 max_iterations=50,
                 initial_temp=1.0,
                 cooling_rate=0.92,
                 step_size=0.2,
                 patience=10,
                 progress_callback=None):
        """
        시뮬레이티드 어닐링으로 최적 가중치를 탐색한다.

        Parameters
        ----------
        initial_weights : dict, optional
            {genre, keyword, numeric, text}
        max_iterations : int
        progress_callback : callable
            (current_iter, total_iters, best_score) 형태

        Returns
        -------
        dict : best_weights, best_score, best_metrics, confidence, accuracy, history
        """
        if initial_weights is None:
            initial_weights = {
                "genre": config.WEIGHT_GENRE,
                "keyword": config.WEIGHT_KEYWORD,
                "numeric": config.WEIGHT_NUMERIC,
                "text": config.WEIGHT_TEXT,
            }

        rng = np.random.RandomState(42)

        current = initial_weights.copy()
        current_metrics = self._evaluate_weights(
            current["genre"], current["keyword"],
            current["numeric"], current["text"]
        )
        current_score = current_metrics["composite_score"]

        best = current.copy()
        best_score = current_score
        best_metrics = current_metrics.copy()

        history = [{
            "iteration": 0,
            "weights": current.copy(),
            "score": current_score,
        }]

        temp = initial_temp
        no_improve_count = 0

        for iteration in range(1, max_iterations + 1):
            if progress_callback:
                progress_callback(iteration, max_iterations, best_score)

            # 이웃 생성
            neighbor = {}
            for key in ["genre", "keyword", "numeric", "text"]:
                delta = rng.uniform(-step_size, step_size)
                val = current[key] + delta
                neighbor[key] = round(np.clip(val, 0.1, 3.0), 1)

            neighbor_metrics = self._evaluate_weights(
                neighbor["genre"], neighbor["keyword"],
                neighbor["numeric"], neighbor["text"]
            )
            neighbor_score = neighbor_metrics["composite_score"]

            # 수용 판정
            delta_score = neighbor_score - current_score
            if delta_score > 0:
                current = neighbor
                current_score = neighbor_score
            else:
                acceptance = np.exp(delta_score / max(temp, 1e-10))
                if rng.random() < acceptance:
                    current = neighbor
                    current_score = neighbor_score

            if current_score > best_score:
                best = current.copy()
                best_score = current_score
                best_metrics = neighbor_metrics.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            history.append({
                "iteration": iteration,
                "weights": current.copy(),
                "score": current_score,
            })

            temp *= cooling_rate

            if no_improve_count >= patience:
                break

        confidence = self._compute_confidence(history, best_score)
        accuracy = self._compute_accuracy(best_metrics)

        return {
            "best_weights": best,
            "best_score": round(best_score, 4),
            "best_metrics": best_metrics,
            "confidence": round(confidence, 4),
            "accuracy": round(accuracy, 4),
            "history": history,
            "iterations": len(history) - 1,
        }

    def _compute_confidence(self, history, best_score):
        """탐색 안정성 기반 신뢰도 (0~1)."""
        if len(history) < 5:
            return 0.3

        scores = [h["score"] for h in history]
        n = len(scores)

        # 후반부 안정성
        tail = scores[int(n * 0.75):]
        if len(tail) > 1:
            tail_std = float(np.std(tail))
            stability = max(0, 1.0 - tail_std * 10)
        else:
            stability = 0.5

        # best_score 근처 해 비율 (재현성)
        close_count = sum(
            1 for s in scores
            if abs(s - best_score) / max(abs(best_score), 1e-10) < 0.05
        )
        reproducibility = min(1.0, close_count / max(n * 0.3, 1))

        # 충분한 탐색 여부
        convergence = min(1.0, n / 20)

        return 0.4 * stability + 0.4 * reproducibility + 0.2 * convergence

    def _compute_accuracy(self, metrics):
        """목적함수 달성률 (0~1)."""
        checks = 0
        total = 4

        if metrics.get("avg_similarity", 0) >= config.THRESHOLD_AVG_SIMILARITY:
            checks += 1
        if metrics.get("genre_precision", 0) >= config.THRESHOLD_GENRE_PRECISION:
            checks += 1
        if metrics.get("text_coherence", 0) >= 0.3:
            checks += 1
        div = metrics.get("diversity", 0)
        if 0.15 <= div <= 0.6:
            checks += 1

        return checks / total
