"""
reduction.py -- Stage 4: 차원 축소
====================================================================
스케일링된 고차원 벡터(499D)를 2D 또는 3D로 축소하여 시각화 좌표를 생성한다.

[지원 방법]
  PCA  : 선형 축소 -- 빠르고 해석 가능 (축 의미 분석)
  t-SNE: 비선형 축소 -- 군집 구조 보존에 유리

[GPU 가속]
  PCA: CUDA SVD 또는 sklearn PCA
  t-SNE: sklearn (GPU 구현 미지원)
"""

import numpy as np

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


class DimensionReducer:
    """차원 축소 클래스 (PCA / t-SNE)"""

    def __init__(self,
                 method=None,
                 n_components=None,
                 tsne_perplexity=None,
                 tsne_learning_rate=None,
                 tsne_n_iter=None):
        if method:
            self.method = method
        elif config.USE_TSNE:
            self.method = "tsne"
        else:
            self.method = "pca"
        self.n_components = n_components if n_components is not None else config.PCA_COMPONENTS
        self.tsne_perplexity = tsne_perplexity if tsne_perplexity is not None else config.TSNE_PERPLEXITY
        self.tsne_learning_rate = tsne_learning_rate if tsne_learning_rate is not None else config.TSNE_LEARNING_RATE
        self.tsne_n_iter = tsne_n_iter if tsne_n_iter is not None else config.TSNE_N_ITER

        self.use_cuda = HAS_TORCH and CUDA_AVAILABLE
        self.device = torch.device("cuda") if self.use_cuda else None

        self._components = None
        self._pca_mean = None
        self._explained_var_ratio = None
        self._cuda_components = None
        self._cuda_mean = None

        self.is_fitted = False

    def fit_transform(self, X):
        """학습 데이터에 대해 차원 축소 수행."""
        if self.method == "pca":
            result = self._fit_transform_pca(X)
        else:
            result = self._fit_transform_tsne(X)

        self.is_fitted = True
        print(f"[reduction] 차원 축소 완료: {self.method.upper()} "
              f"({X.shape[1]}D -> {self.n_components}D)")
        if self.method == "pca" and self._explained_var_ratio is not None:
            var_str = ", ".join(f"{v:.3f}" for v in self._explained_var_ratio)
            total = sum(self._explained_var_ratio)
            print(f"  PCA 분산 비율: [{var_str}] (총 {total:.1%})")

        return result

    def _fit_transform_pca(self, X):
        if self.use_cuda:
            return self._pca_cuda(X)
        else:
            return self._pca_sklearn(X)

    def _pca_sklearn(self, X):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components)
        result = pca.fit_transform(X)
        self._components = pca.components_
        self._pca_mean = pca.mean_
        self._explained_var_ratio = pca.explained_variance_ratio_
        return result

    def _pca_cuda(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self._cuda_mean = X_t.mean(dim=0)
        X_centered = X_t - self._cuda_mean
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        self._cuda_components = Vh[:self.n_components, :]
        total_var = (S ** 2).sum()
        self._explained_var_ratio = (
            (S[:self.n_components] ** 2) / total_var
        ).cpu().numpy()
        X_reduced = X_centered @ self._cuda_components.T
        self._components = self._cuda_components.cpu().numpy()
        self._pca_mean = self._cuda_mean.cpu().numpy()
        return X_reduced.cpu().numpy()

    def _fit_transform_tsne(self, X):
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.tsne_perplexity,
            learning_rate=self.tsne_learning_rate,
            n_iter=self.tsne_n_iter,
            random_state=42,
        )
        return tsne.fit_transform(X)

    def transform(self, X):
        """새 데이터를 학습된 PCA로 변환."""
        if not self.is_fitted:
            raise RuntimeError("먼저 fit_transform()을 호출하세요.")
        if self.method == "tsne":
            raise RuntimeError("t-SNE는 새 데이터 변환을 지원하지 않습니다.")

        if self.use_cuda and self._cuda_components is not None:
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_centered = X_t - self._cuda_mean
            result = X_centered @ self._cuda_components.T
            return result.cpu().numpy()
        else:
            X_centered = X - self._pca_mean
            return X_centered @ self._components.T

    def get_explained_variance(self):
        if self._explained_var_ratio is None:
            return np.array([])
        return self._explained_var_ratio

    def get_components(self):
        return self._components

    def interpret_axes(self, feature_names, top_n=5):
        """PCA 축별 주요 특징 해석."""
        if self._components is None:
            return {}
        result = {}
        if self.n_components == 2:
            axis_labels = ["PC1 (X축)", "PC2 (Y축)"]
        else:
            axis_labels = ["PC1 (X축)", "PC2 (Y축)", "PC3 (Z축)"]

        for i, label in enumerate(axis_labels):
            if i >= len(self._components):
                break
            weights = self._components[i]
            indices = np.argsort(np.abs(weights))[::-1][:top_n]
            result[label] = [
                (feature_names[idx], round(float(weights[idx]), 3))
                for idx in indices
            ]
        return result
