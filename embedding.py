"""
embedding.py -- Stage 2: 특징 벡터 생성
====================================================================
영화 데이터를 고차원 특징 벡터(54D)로 변환하고 정규화한다.
PCA/t-SNE 차원 축소는 별도 모듈(reduction.py)에서 처리한다.

[벡터 구조]
  장르 원-핫 (17D) + 키워드 바이너리 (30D) + 수치 특징 (7D) = 54D
  각 그룹에 가중치(WEIGHT_GENRE, WEIGHT_KEYWORD, WEIGHT_NUMERIC) 적용

[GPU 가속]
  CUDA 사용 가능 시 StandardScaler, 코사인 유사도를 GPU에서 처리
"""

import numpy as np

import config

# ------------------------------------------------------------------
# PyTorch CUDA 감지
# ------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

if not (HAS_TORCH and CUDA_AVAILABLE):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity


class MovieEmbedding:
    """영화 데이터를 가중 특징 벡터로 변환하는 엔진"""

    def __init__(self,
                 genres: list[str] = None,
                 keywords: list[str] = None,
                 numeric_features: list[str] = None,
                 weight_genre: float = None,
                 weight_keyword: float = None,
                 weight_numeric: float = None):
        self.genres = genres or config.ALL_GENRES
        self.keywords = keywords or config.ALL_KEYWORDS
        self.numeric_features = numeric_features or config.NUMERIC_FEATURES
        self.w_genre = weight_genre if weight_genre is not None else config.WEIGHT_GENRE
        self.w_keyword = weight_keyword if weight_keyword is not None else config.WEIGHT_KEYWORD
        self.w_numeric = weight_numeric if weight_numeric is not None else config.WEIGHT_NUMERIC

        self.use_cuda = HAS_TORCH and CUDA_AVAILABLE
        self.device = torch.device("cuda") if self.use_cuda else None

        # 스케일러 파라미터
        if self.use_cuda:
            self._mean = None
            self._std = None
        else:
            self._scaler = StandardScaler()

        self.is_fitted = False
        self.feature_names = self._build_feature_names()
        self.dim = len(self.feature_names)

        # 벡터 저장소
        self.raw_vectors = {}      # title -> np.ndarray (가중 원본)
        self.scaled_vectors = {}   # title -> np.ndarray (스케일링 후)

    # ------------------------------------------------------------------
    # 특징 이름 목록
    # ------------------------------------------------------------------
    def _build_feature_names(self) -> list[str]:
        return (
            [f"genre:{g}" for g in self.genres]
            + [f"kw:{k}" for k in self.keywords]
            + self.numeric_features
        )

    def get_feature_names(self) -> list[str]:
        return self.feature_names

    # ------------------------------------------------------------------
    # 영화 -> 벡터 변환
    # ------------------------------------------------------------------
    def movie_to_vector(self, movie: dict) -> np.ndarray:
        """단일 영화를 가중 특징 벡터로 변환"""
        genre_vec = np.array(
            [1.0 if g in movie["genres"] else 0.0 for g in self.genres],
            dtype=np.float64,
        )
        keyword_vec = np.array(
            [1.0 if k in movie["keywords"] else 0.0 for k in self.keywords],
            dtype=np.float64,
        )
        numeric_vec = np.array(
            [movie.get(f) if movie.get(f) is not None else 0.5
             for f in self.numeric_features],
            dtype=np.float64,
        )

        # 가중치 적용
        full_vector = np.concatenate([
            genre_vec * self.w_genre,
            keyword_vec * self.w_keyword,
            numeric_vec * self.w_numeric,
        ])
        return full_vector

    # ------------------------------------------------------------------
    # fit: 스케일러 학습
    # ------------------------------------------------------------------
    def fit(self, movies: list[dict]):
        """학습 데이터로 벡터 생성 + 스케일러 학습"""
        vectors = []
        titles = []
        for m in movies:
            vec = self.movie_to_vector(m)
            self.raw_vectors[m["title"]] = vec
            vectors.append(vec)
            titles.append(m["title"])

        X = np.array(vectors)

        if self.use_cuda:
            X_scaled = self._fit_cuda(X)
        else:
            X_scaled = self._fit_sklearn(X)

        for i, title in enumerate(titles):
            self.scaled_vectors[title] = X_scaled[i]

        self.is_fitted = True
        print(f"[Stage 2] 임베딩 완료: {len(movies)}편 -> {self.dim}D 벡터")
        print(f"   가중치: 장르={self.w_genre}, 키워드={self.w_keyword}, 수치={self.w_numeric}")
        if self.use_cuda:
            print(f"   연산 장치: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"   연산 장치: CPU (sklearn)")

    def _fit_sklearn(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(X)

    def _fit_cuda(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self._mean = X_t.mean(dim=0)
        self._std = X_t.std(dim=0)
        self._std[self._std < 1e-10] = 1.0
        X_scaled = (X_t - self._mean) / self._std
        return X_scaled.cpu().numpy()

    # ------------------------------------------------------------------
    # transform: 새 데이터 변환
    # ------------------------------------------------------------------
    def transform(self, movies: list[dict]) -> np.ndarray:
        """새 영화를 학습된 스케일러로 변환"""
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요.")

        vectors = []
        titles = []
        for m in movies:
            vec = self.movie_to_vector(m)
            self.raw_vectors[m["title"]] = vec
            vectors.append(vec)
            titles.append(m["title"])

        X = np.array(vectors)

        if self.use_cuda:
            X_scaled = self._transform_cuda(X)
        else:
            X_scaled = self._transform_sklearn(X)

        for i, title in enumerate(titles):
            self.scaled_vectors[title] = X_scaled[i]

        return X_scaled

    def _transform_sklearn(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(X)

    def _transform_cuda(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_scaled = (X_t - self._mean) / self._std
        return X_scaled.cpu().numpy()

    # ------------------------------------------------------------------
    # 유사도 계산
    # ------------------------------------------------------------------
    def compute_similarity(self, title_a: str, title_b: str) -> float:
        """두 영화 간 코사인 유사도 (원본 가중 벡터 기준)"""
        vec_a = self.raw_vectors.get(title_a)
        vec_b = self.raw_vectors.get(title_b)
        if vec_a is None or vec_b is None:
            return 0.0

        if self.use_cuda:
            a = torch.tensor(vec_a, dtype=torch.float32, device=self.device)
            b = torch.tensor(vec_b, dtype=torch.float32, device=self.device)
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        else:
            return cosine_similarity(
                vec_a.reshape(1, -1), vec_b.reshape(1, -1)
            )[0, 0]

    def compute_similarity_matrix(
        self, query_titles: list[str], train_titles: list[str]
    ) -> np.ndarray:
        """배치 코사인 유사도 행렬 (M, N)"""
        q_vecs = np.array([self.raw_vectors[t] for t in query_titles])
        t_vecs = np.array([self.raw_vectors[t] for t in train_titles])

        if self.use_cuda:
            q = torch.tensor(q_vecs, dtype=torch.float32, device=self.device)
            t = torch.tensor(t_vecs, dtype=torch.float32, device=self.device)
            q_norm = F.normalize(q, dim=1)
            t_norm = F.normalize(t, dim=1)
            sim = q_norm @ t_norm.T
            return sim.cpu().numpy()
        else:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            return cos_sim(q_vecs, t_vecs)

    def get_raw_vector(self, title: str) -> np.ndarray:
        """특정 영화의 원본 가중 벡터 반환"""
        return self.raw_vectors.get(title)

    def get_scaled_vector(self, title: str) -> np.ndarray:
        """특정 영화의 스케일링된 벡터 반환"""
        return self.scaled_vectors.get(title)

    def get_all_scaled_matrix(self, titles: list[str]) -> np.ndarray:
        """여러 영화의 스케일링된 벡터를 행렬로 반환"""
        return np.array([self.scaled_vectors[t] for t in titles])
