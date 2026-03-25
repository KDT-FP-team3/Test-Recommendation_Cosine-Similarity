"""
clustering.py -- Stage 3: 군집화
====================================================================
스케일링된 벡터에 KMeans 또는 DBSCAN을 적용하여 영화를 군집화한다.
군집 정보는 시각화에서 색상 구분 및 평가에서 활용된다.

[GPU 가속]
  KMeans: CUDA 사용 가능 시 PyTorch 기반 구현
  DBSCAN: sklearn 사용 (GPU 구현 복잡도 대비 실익이 적음)
"""

import numpy as np
from collections import Counter

import config

# ------------------------------------------------------------------
# PyTorch CUDA 감지
# ------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False


class MovieClusterer:
    """영화 벡터를 군집화하는 클래스"""

    def __init__(self,
                 method: str = None,
                 n_clusters: int = None,
                 eps: float = None,
                 min_samples: int = None):
        self.method = method or config.CLUSTER_METHOD
        self.n_clusters = n_clusters if n_clusters is not None else config.KMEANS_N_CLUSTERS
        self.eps = eps if eps is not None else config.DBSCAN_EPS
        self.min_samples = min_samples if min_samples is not None else config.DBSCAN_MIN_SAMPLES

        self.use_cuda = HAS_TORCH and CUDA_AVAILABLE
        self.device = torch.device("cuda") if self.use_cuda else None

        self.labels_ = None
        self.n_clusters_found = 0

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        군집화 수행.

        Args:
            X: (N, D) 스케일링된 벡터 행렬

        Returns:
            (N,) 클러스터 라벨 배열
        """
        if self.method == "kmeans":
            self.labels_ = self._kmeans(X)
        elif self.method == "dbscan":
            self.labels_ = self._dbscan(X)
        else:
            raise ValueError(f"지원하지 않는 군집화 방법: {self.method}")

        unique = set(self.labels_)
        unique.discard(-1)
        self.n_clusters_found = len(unique)

        print(f"[Stage 3] 군집화 완료: {self.method.upper()} -> "
              f"{self.n_clusters_found}개 클러스터")
        if self.method == "dbscan":
            n_noise = int(np.sum(self.labels_ == -1))
            print(f"   노이즈 포인트: {n_noise}개")

        return self.labels_

    def _kmeans(self, X: np.ndarray) -> np.ndarray:
        """KMeans 군집화"""
        if self.use_cuda:
            return self._kmeans_cuda(X)
        else:
            return self._kmeans_sklearn(X)

    def _kmeans_sklearn(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return km.fit_predict(X)

    def _kmeans_cuda(self, X: np.ndarray) -> np.ndarray:
        """PyTorch GPU 기반 KMeans"""
        # CUDA 결정성 보장: 동일 데이터 → 동일 클러스터
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n, d = X_t.shape
        k = self.n_clusters

        # 초기 중심: 시드 고정으로 재현성 보장
        g = torch.Generator(device=self.device)
        g.manual_seed(42)
        indices = torch.randperm(n, device=self.device, generator=g)[:k]
        centroids = X_t[indices].clone()

        max_iter = 300
        for _ in range(max_iter):
            # 거리 계산: (N, K)
            dists = torch.cdist(X_t, centroids)
            labels = dists.argmin(dim=1)

            # 중심 업데이트
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = X_t[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels.cpu().numpy()

    def _dbscan(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN 군집화 (sklearn)"""
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return db.fit_predict(X)

    # ------------------------------------------------------------------
    # 클러스터 분석
    # ------------------------------------------------------------------
    def get_cluster_info(self, labels: np.ndarray,
                         movies: list[dict]) -> dict:
        """
        클러스터별 장르/키워드 분포 분석.

        Returns:
            {cluster_id: {"count": int, "top_genres": [...], "top_keywords": [...]}}
        """
        info = {}
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_movies = [m for m, in_cluster in zip(movies, mask) if in_cluster]

            # 장르 분포
            all_genres = []
            for m in cluster_movies:
                all_genres.extend(m["genres"])
            top_genres = Counter(all_genres).most_common(5)

            # 키워드 분포
            all_kw = []
            for m in cluster_movies:
                all_kw.extend(m["keywords"])
            top_keywords = Counter(all_kw).most_common(5)

            label_name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
            info[label_name] = {
                "count": len(cluster_movies),
                "top_genres": top_genres,
                "top_keywords": top_keywords,
            }

        return info
