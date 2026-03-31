"""
embedding.py -- 하이브리드 특징 벡터 생성
====================================================================
영화 데이터를 4개 그룹의 하이브리드 벡터(499D)로 변환한다.

[벡터 구조]
  장르 원-핫 (30D) + 키워드 바이너리 (80D) + 수치 (5D) + 텍스트 임베딩 (384D)
  = 499D

[정규화 전략]
  각 그룹을 L2-정규화한 후 가중치를 곱하고 연결(concatenate)한다.
  이를 통해 차원 수 차이에 의한 편향을 방지한다.

[GPU 가속]
  sentence-transformers, PyTorch CUDA를 활용한다.
"""

import os
import numpy as np
from tqdm import tqdm

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


def _l2_normalize(vectors):
    """행별 L2 정규화. 영벡터는 그대로 유지한다."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return vectors / norms


class HybridMovieEmbedding:
    """
    영화 데이터를 4개 그룹 하이브리드 벡터로 변환하는 엔진.

    그룹 1: 장르 원-핫 (30D)
    그룹 2: 키워드 바이너리 (80D)
    그룹 3: 파생 수치 (5D)
    그룹 4: 줄거리 텍스트 임베딩 (384D)
    """

    def __init__(self,
                 weight_genre=None,
                 weight_keyword=None,
                 weight_numeric=None,
                 weight_text=None):
        self.genres = config.ALL_GENRES
        self.keywords = config.ALL_KEYWORDS
        self.numeric_features = config.NUMERIC_FEATURES
        self.genre_set = set(self.genres)
        self.keyword_set = set(self.keywords)

        self.w_genre = weight_genre if weight_genre is not None else config.WEIGHT_GENRE
        self.w_keyword = weight_keyword if weight_keyword is not None else config.WEIGHT_KEYWORD
        self.w_numeric = weight_numeric if weight_numeric is not None else config.WEIGHT_NUMERIC
        self.w_text = weight_text if weight_text is not None else config.WEIGHT_TEXT

        self.use_cuda = HAS_TORCH and CUDA_AVAILABLE
        self.device = torch.device("cuda") if self.use_cuda else None

        # 표준화: CUDA -> 수동 mean/std, CPU -> StandardScaler
        if self.use_cuda:
            self._mean = None
            self._std = None
        else:
            self._scaler = StandardScaler()

        self.is_fitted = False
        self._text_model = None
        self._train_ids = []

        # 벡터 저장소
        self.raw_vectors = {}       # id -> np.ndarray (가중 원본)
        self.scaled_vectors = {}    # id -> np.ndarray (스케일링 후)
        self.group_vectors = {}     # id -> dict{genre, keyword, numeric, text}

        # 차원 정보
        self.genre_dim = len(self.genres)
        self.keyword_dim = len(self.keywords)
        self.numeric_dim = len(self.numeric_features)
        self.text_dim = config.TEXT_EMBED_DIM
        self.dim = self.genre_dim + self.keyword_dim + self.numeric_dim + self.text_dim

        self.feature_names = self._build_feature_names()

    def _build_feature_names(self):
        return (
            [f"genre:{g}" for g in self.genres]
            + [f"kw:{k}" for k in self.keywords]
            + [f"num:{f}" for f in self.numeric_features]
            + [f"text:{i}" for i in range(self.text_dim)]
        )

    def get_feature_names(self):
        return self.feature_names

    # ------------------------------------------------------------------
    # 텍스트 모델 로딩
    # ------------------------------------------------------------------
    def _load_text_model(self):
        """sentence-transformers 모델을 지연 로드한다."""
        if self._text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            device_str = "cuda" if self.use_cuda else "cpu"
            print(f"[embedding] 텍스트 모델 로드 중: {config.TEXT_MODEL_NAME}")
            self._text_model = SentenceTransformer(
                config.TEXT_MODEL_NAME, device=device_str
            )
            print(f"[embedding] 텍스트 모델 로드 완료 (장치: {device_str})")
        except ImportError:
            print("[embedding] sentence-transformers 미설치. pip install sentence-transformers")
            self._text_model = None

    # ------------------------------------------------------------------
    # 메타데이터 벡터 생성
    # ------------------------------------------------------------------
    def _movie_to_metadata_vector(self, movie):
        """단일 영화 -> 메타데이터 벡터 (genre + keyword + numeric)"""
        genre_vec = np.array(
            [1.0 if g in movie.get("genres", []) else 0.0 for g in self.genres],
            dtype=np.float32,
        )
        kw_matched = movie.get("keywords_matched", [])
        keyword_vec = np.array(
            [1.0 if k in kw_matched else 0.0 for k in self.keywords],
            dtype=np.float32,
        )
        numeric_vec = np.array(
            [movie.get(f, 0.5) for f in self.numeric_features],
            dtype=np.float32,
        )
        return genre_vec, keyword_vec, numeric_vec

    # ------------------------------------------------------------------
    # 텍스트 임베딩
    # ------------------------------------------------------------------
    def _encode_texts(self, texts, show_progress=True):
        """텍스트 목록을 sentence-transformers로 인코딩한다."""
        self._load_text_model()
        if self._text_model is None:
            return np.zeros((len(texts), self.text_dim), dtype=np.float32)

        # 빈 텍스트 처리
        processed = []
        for t in texts:
            if t and t.strip():
                processed.append(t.strip()[:512])  # 최대 512자
            else:
                processed.append("영화")  # 빈 텍스트 대체

        embeddings = self._text_model.encode(
            processed,
            batch_size=config.TEXT_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    def _encode_single_text(self, text):
        """단일 텍스트를 인코딩한다."""
        self._load_text_model()
        if self._text_model is None:
            return np.zeros(self.text_dim, dtype=np.float32)
        if not text or not text.strip():
            text = "영화"
        embedding = self._text_model.encode(
            text.strip()[:512],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # 가중 벡터 생성 (그룹별 L2 정규화 + 가중치)
    # ------------------------------------------------------------------
    def _build_weighted_vectors(self, genre_mat, keyword_mat, numeric_mat, text_mat):
        """
        4개 그룹 행렬을 L2 정규화하고 가중치를 곱한 후 연결한다.

        Returns
        -------
        weighted : np.ndarray (N, dim)
        """
        g_norm = _l2_normalize(genre_mat) * self.w_genre
        k_norm = _l2_normalize(keyword_mat) * self.w_keyword
        n_norm = _l2_normalize(numeric_mat) * self.w_numeric
        t_norm = _l2_normalize(text_mat) * self.w_text

        return np.hstack([g_norm, k_norm, n_norm, t_norm])

    # ------------------------------------------------------------------
    # fit: 학습 데이터 처리
    # ------------------------------------------------------------------
    def fit(self, movies, show_progress=True):
        """
        학습 데이터로 벡터를 생성하고 스케일러를 학습한다.

        Parameters
        ----------
        movies : list[dict]
        show_progress : bool
        """
        n = len(movies)
        genre_mat = np.zeros((n, self.genre_dim), dtype=np.float32)
        keyword_mat = np.zeros((n, self.keyword_dim), dtype=np.float32)
        numeric_mat = np.zeros((n, self.numeric_dim), dtype=np.float32)
        ids = []

        for i, m in enumerate(movies):
            g, k, num = self._movie_to_metadata_vector(m)
            genre_mat[i] = g
            keyword_mat[i] = k
            numeric_mat[i] = num
            ids.append(m["id"])

        # 텍스트 임베딩 (캐시 확인)
        text_mat = self._load_or_compute_text_embeddings(
            movies, ids, show_progress
        )

        # 가중 벡터 생성
        weighted = self._build_weighted_vectors(
            genre_mat, keyword_mat, numeric_mat, text_mat
        )

        # StandardScaler 학습
        if self.use_cuda:
            X_scaled = self._fit_cuda(weighted)
        else:
            X_scaled = self._fit_sklearn(weighted)

        # 저장
        for i, mid in enumerate(ids):
            self.raw_vectors[mid] = weighted[i]
            self.scaled_vectors[mid] = X_scaled[i]
            self.group_vectors[mid] = {
                "genre": genre_mat[i],
                "keyword": keyword_mat[i],
                "numeric": numeric_mat[i],
                "text": text_mat[i],
            }

        self.is_fitted = True
        self._train_ids = ids

        print(f"[embedding] 임베딩 완료: {n:,}편 -> {self.dim}D 벡터")
        print(f"  가중치: 장르={self.w_genre}, 키워드={self.w_keyword}, "
              f"수치={self.w_numeric}, 텍스트={self.w_text}")
        if self.use_cuda:
            print(f"  연산 장치: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  연산 장치: CPU")

    def _load_or_compute_text_embeddings(self, movies, ids, show_progress):
        """텍스트 임베딩을 캐시에서 로드하거나 새로 계산한다."""
        cache_path = config.TEXT_EMBED_CACHE

        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            if cached.shape[0] == len(movies):
                print(f"[embedding] 텍스트 임베딩 캐시 로드: {cache_path}")
                return cached

        print(f"[embedding] 텍스트 임베딩 생성 중 ({len(movies):,}편)...")
        texts = []
        for m in movies:
            text = m.get("plot_ko", "") or m.get("plot_en", "") or ""
            texts.append(text)

        text_mat = self._encode_texts(texts, show_progress)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, text_mat)
        print(f"[embedding] 텍스트 임베딩 캐시 저장: {cache_path}")

        return text_mat

    def _fit_sklearn(self, X):
        return self._scaler.fit_transform(X)

    def _fit_cuda(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self._mean = X_t.mean(dim=0)
        self._std = X_t.std(dim=0)
        self._std[self._std < 1e-10] = 1.0
        X_scaled = (X_t - self._mean) / self._std
        return X_scaled.cpu().numpy()

    # ------------------------------------------------------------------
    # transform: 새 데이터 변환
    # ------------------------------------------------------------------
    def transform(self, movies, show_progress=False):
        """새 영화를 학습된 스케일러로 변환한다."""
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요.")

        n = len(movies)
        genre_mat = np.zeros((n, self.genre_dim), dtype=np.float32)
        keyword_mat = np.zeros((n, self.keyword_dim), dtype=np.float32)
        numeric_mat = np.zeros((n, self.numeric_dim), dtype=np.float32)
        ids = []

        for i, m in enumerate(movies):
            g, k, num = self._movie_to_metadata_vector(m)
            genre_mat[i] = g
            keyword_mat[i] = k
            numeric_mat[i] = num
            ids.append(m["id"])

        # 텍스트 임베딩
        texts = [m.get("plot_ko", "") or m.get("plot_en", "") or "" for m in movies]
        text_mat = self._encode_texts(texts, show_progress)

        # 가중 벡터
        weighted = self._build_weighted_vectors(
            genre_mat, keyword_mat, numeric_mat, text_mat
        )

        # 스케일링
        if self.use_cuda:
            X_scaled = self._transform_cuda(weighted)
        else:
            X_scaled = self._transform_sklearn(weighted)

        for i, mid in enumerate(ids):
            self.raw_vectors[mid] = weighted[i]
            self.scaled_vectors[mid] = X_scaled[i]
            self.group_vectors[mid] = {
                "genre": genre_mat[i],
                "keyword": keyword_mat[i],
                "numeric": numeric_mat[i],
                "text": text_mat[i],
            }

        return X_scaled

    def _transform_sklearn(self, X):
        return self._scaler.transform(X)

    def _transform_cuda(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_scaled = (X_t - self._mean) / self._std
        return X_scaled.cpu().numpy()

    # ------------------------------------------------------------------
    # 쿼리 벡터 생성 (검색용)
    # ------------------------------------------------------------------
    def build_query_vector(self, genres=None, keywords=None,
                           numeric_values=None, text=None):
        """
        자유 검색용 쿼리 벡터를 생성한다.

        Parameters
        ----------
        genres : list[str], optional
        keywords : list[str], optional
        numeric_values : dict, optional
        text : str, optional

        Returns
        -------
        np.ndarray (dim,)
        """
        genre_vec = np.array(
            [1.0 if g in (genres or []) else 0.0 for g in self.genres],
            dtype=np.float32,
        )
        keyword_vec = np.array(
            [1.0 if k in (keywords or []) else 0.0 for k in self.keywords],
            dtype=np.float32,
        )
        nv = numeric_values or {}
        numeric_vec = np.array(
            [nv.get(f, 0.5) for f in self.numeric_features],
            dtype=np.float32,
        )

        # 텍스트 임베딩
        if text:
            text_vec = self._encode_single_text(text)
        else:
            text_vec = np.zeros(self.text_dim, dtype=np.float32)

        # 그룹별 L2 정규화 + 가중치
        genre_mat = genre_vec.reshape(1, -1)
        keyword_mat = keyword_vec.reshape(1, -1)
        numeric_mat = numeric_vec.reshape(1, -1)
        text_mat = text_vec.reshape(1, -1)

        weighted = self._build_weighted_vectors(
            genre_mat, keyword_mat, numeric_mat, text_mat
        )
        return weighted[0]

    # ------------------------------------------------------------------
    # 유사도 계산
    # ------------------------------------------------------------------
    def compute_similarity_to_train(self, query_vec, train_ids=None):
        """
        쿼리 벡터와 학습 데이터 간 코사인 유사도를 계산한다.

        Returns
        -------
        list[(id, similarity)]  유사도 내림차순 정렬
        """
        if train_ids is None:
            train_ids = self._train_ids

        train_vecs = np.array([self.raw_vectors[tid] for tid in train_ids])
        query = query_vec.reshape(1, -1)

        if self.use_cuda:
            q = torch.tensor(query, dtype=torch.float32, device=self.device)
            t = torch.tensor(train_vecs, dtype=torch.float32, device=self.device)
            q_n = F.normalize(q, dim=1)
            t_n = F.normalize(t, dim=1)
            sims = (q_n @ t_n.T).squeeze(0).cpu().numpy()
        else:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sims = cos_sim(query, train_vecs)[0]

        results = [(train_ids[i], float(sims[i])) for i in range(len(train_ids))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def compute_group_similarity(self, id_a, id_b):
        """
        두 영화 간 그룹별 코사인 유사도를 분해한다.

        Returns
        -------
        dict : {genre, keyword, numeric, text, total}
        """
        ga = self.group_vectors.get(id_a)
        gb = self.group_vectors.get(id_b)
        if ga is None or gb is None:
            return {"genre": 0, "keyword": 0, "numeric": 0, "text": 0, "total": 0}

        def _cos(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-10 or nb < 1e-10:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        result = {}
        for group in ["genre", "keyword", "numeric", "text"]:
            result[group] = _cos(ga[group], gb[group])

        # 전체 유사도
        va = self.raw_vectors.get(id_a)
        vb = self.raw_vectors.get(id_b)
        if va is not None and vb is not None:
            result["total"] = _cos(va, vb)
        else:
            result["total"] = 0.0

        return result

    def compute_similarity_matrix(self, query_ids, train_ids):
        """배치 코사인 유사도 행렬 (M, N)"""
        valid_q = [t for t in query_ids if t in self.raw_vectors]
        valid_t = [t for t in train_ids if t in self.raw_vectors]
        if not valid_q or not valid_t:
            return np.zeros((len(query_ids), len(train_ids)))
        q_vecs = np.array([self.raw_vectors[t] for t in valid_q])
        t_vecs = np.array([self.raw_vectors[t] for t in valid_t])

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

    # ------------------------------------------------------------------
    # 벡터 접근
    # ------------------------------------------------------------------
    def get_raw_vector(self, movie_id):
        return self.raw_vectors.get(movie_id)

    def get_scaled_vector(self, movie_id):
        return self.scaled_vectors.get(movie_id)

    def get_all_scaled_matrix(self, ids):
        valid = [i for i in ids if i in self.scaled_vectors]
        if not valid:
            return np.zeros((0, self.dim))
        return np.array([self.scaled_vectors[i] for i in valid])

    def rebuild_with_weights(self, w_genre=None, w_keyword=None,
                             w_numeric=None, w_text=None):
        """
        가중치만 변경하여 벡터를 재생성한다 (텍스트 임베딩 재계산 불필요).
        대시보드 실시간 업데이트용.

        Returns
        -------
        dict : {id: raw_vector}
        """
        wg = w_genre if w_genre is not None else self.w_genre
        wk = w_keyword if w_keyword is not None else self.w_keyword
        wn = w_numeric if w_numeric is not None else self.w_numeric
        wt = w_text if w_text is not None else self.w_text

        new_raw = {}
        for mid, groups in self.group_vectors.items():
            g = groups["genre"].reshape(1, -1)
            k = groups["keyword"].reshape(1, -1)
            n = groups["numeric"].reshape(1, -1)
            t = groups["text"].reshape(1, -1)

            g_n = _l2_normalize(g) * wg
            k_n = _l2_normalize(k) * wk
            n_n = _l2_normalize(n) * wn
            t_n = _l2_normalize(t) * wt

            vec = np.hstack([g_n, k_n, n_n, t_n])[0]
            new_raw[mid] = vec

        return new_raw
