"""
search.py -- 영화 검색 엔진
====================================================================
두 가지 검색 방식을 지원한다:

[1] 제목 검색 (Title Search)
    영화 제목을 입력하면 학습 데이터에서 가장 유사한 영화를 추천.
    - 학습 데이터에 있는 제목: 해당 영화 제외 후 유사 영화 추천
    - 테스트 데이터에 있는 제목: 기존 추천 로직 사용
    - 데이터에 없는 제목: 부분 일치로 검색 시도

[2] 자유 텍스트 검색 (Free Text Search)
    장르, 키워드, 느낌, 분위기 등을 자유롭게 입력하면
    텍스트를 파싱하여 54D 쿼리 벡터를 생성하고 유사 영화를 추천.
    한국어/영어 모두 지원.
"""

import numpy as np

from embedding import MovieEmbedding
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
    from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════════════════════════
# 한국어/영어 별칭 사전
# ═══════════════════════════════════════════════════════════════════

# 장르 별칭 -> config.ALL_GENRES 이름
GENRE_ALIASES = {
    # Action
    "action": "Action", "액션": "Action",
    # Adventure
    "adventure": "Adventure", "모험": "Adventure", "어드벤처": "Adventure",
    # Animation
    "animation": "Animation", "애니메이션": "Animation", "애니": "Animation",
    # Comedy
    "comedy": "Comedy", "코미디": "Comedy", "코메디": "Comedy",
    # Crime
    "crime": "Crime", "범죄": "Crime",
    # Drama
    "drama": "Drama", "드라마": "Drama",
    # Fantasy
    "fantasy": "Fantasy", "판타지": "Fantasy",
    # Horror
    "horror": "Horror", "호러": "Horror", "공포": "Horror",
    # Mystery
    "mystery": "Mystery", "미스터리": "Mystery",
    # Romance
    "romance": "Romance", "로맨스": "Romance", "로맨틱": "Romance",
    # Sci-Fi
    "sci-fi": "Sci-Fi", "sf": "Sci-Fi", "공상과학": "Sci-Fi",
    "사이파이": "Sci-Fi", "과학": "Sci-Fi",
    # Thriller
    "thriller": "Thriller", "스릴러": "Thriller",
    # War
    "war": "War", "전쟁": "War",
    # Musical
    "musical": "Musical", "뮤지컬": "Musical",
    # Documentary
    "documentary": "Documentary", "다큐멘터리": "Documentary", "다큐": "Documentary",
    # Family
    "family": "Family", "가족": "Family",
    # History
    "history": "History", "역사": "History", "사극": "History",
    "역사적": "History",
}

# 키워드 별칭 -> config.ALL_KEYWORDS 이름
KEYWORD_ALIASES = {
    # superhero
    "superhero": "superhero", "슈퍼히어로": "superhero", "히어로": "superhero",
    "영웅": "superhero",
    # space
    "space": "space", "우주": "space",
    # time-travel
    "time-travel": "time-travel", "시간여행": "time-travel",
    "타임트래블": "time-travel",
    # dystopia
    "dystopia": "dystopia", "디스토피아": "dystopia",
    # revenge
    "revenge": "revenge", "복수": "revenge", "복수극": "revenge",
    # love
    "love": "love", "사랑": "love", "연애": "love",
    # friendship
    "friendship": "friendship", "우정": "friendship",
    # survival
    "survival": "survival", "생존": "survival", "서바이벌": "survival",
    # war (keyword)
    "전투": "war",
    # heist
    "heist": "heist", "강도": "heist", "도둑": "heist",
    # coming-of-age
    "coming-of-age": "coming-of-age", "성장": "coming-of-age",
    "성장기": "coming-of-age",
    # technology
    "technology": "technology", "기술": "technology", "테크놀로지": "technology",
    "테크": "technology",
    # monster
    "monster": "monster", "몬스터": "monster", "괴물": "monster",
    # magic
    "magic": "magic", "마법": "magic", "마술": "magic",
    # detective
    "detective": "detective", "탐정": "detective", "수사": "detective",
    "형사": "detective",
    # family (keyword)
    "가정": "family",
    # conspiracy
    "conspiracy": "conspiracy", "음모": "conspiracy",
    # sports
    "sports": "sports", "스포츠": "sports",
    # music
    "music": "music", "음악": "music",
    # nature
    "nature": "nature", "자연": "nature",
    # robot
    "robot": "robot", "로봇": "robot",
    # politics
    "politics": "politics", "정치": "politics",
    # psychological
    "psychological": "psychological", "심리": "psychological",
    "심리적": "psychological", "심리극": "psychological",
    # historical
    "historical": "historical", "역사물": "historical",
    # comedy (keyword)
    # (handled by genre alias)
    # apocalypse
    "apocalypse": "apocalypse", "종말": "apocalypse", "아포칼립스": "apocalypse",
    # alien
    "alien": "alien", "외계인": "alien", "에일리언": "alien",
    # spy
    "spy": "spy", "스파이": "spy", "첩보": "spy",
    # martial-arts
    "martial-arts": "martial-arts", "무술": "martial-arts",
    "격투": "martial-arts",
    # animation (keyword)
    # (handled by genre alias)
}

# 분위기 단어 -> mood 값 (0.0=밝음, 1.0=어두움)
MOOD_WORDS = {
    # 밝은 (low mood)
    "밝은": 0.15, "유쾌한": 0.15, "즐거운": 0.15, "따뜻한": 0.2,
    "희망적": 0.2, "가벼운": 0.15, "경쾌한": 0.15,
    "bright": 0.15, "cheerful": 0.15, "warm": 0.2,
    "light": 0.15, "uplifting": 0.2, "feel-good": 0.15,
    # 중간 (mid mood)
    "감동적": 0.45, "진지한": 0.55, "사실적": 0.5,
    "serious": 0.55, "emotional": 0.45, "realistic": 0.5,
    # 어두운 (high mood)
    "어두운": 0.85, "무거운": 0.8, "암울한": 0.9, "우울한": 0.85,
    "잔혹한": 0.95, "공포스러운": 0.9, "섬뜩한": 0.9,
    "긴장감": 0.75, "긴장": 0.75, "불안한": 0.8,
    "dark": 0.85, "grim": 0.9, "intense": 0.75,
    "heavy": 0.8, "bleak": 0.9, "tense": 0.75,
    "scary": 0.9, "creepy": 0.9,
}

# 전개 속도 단어 -> tempo 값 (0.0=느림, 1.0=빠름)
TEMPO_WORDS = {
    # 느림 (low tempo)
    "느린": 0.2, "느긋한": 0.2, "잔잔한": 0.15,
    "차분한": 0.2, "서정적": 0.2,
    "slow": 0.2, "calm": 0.2, "meditative": 0.15,
    # 중간 (mid tempo)
    "보통": 0.5, "적절한": 0.5,
    "moderate": 0.5,
    # 빠름 (high tempo)
    "빠른": 0.85, "속도감": 0.8, "질주": 0.9,
    "논스톱": 0.9, "숨막히는": 0.85,
    "fast": 0.85, "rapid": 0.85, "nonstop": 0.9,
    "fast-paced": 0.85, "action-packed": 0.85,
}

# 시각 스타일 단어 -> visual_style 값 (0.0=사실적, 1.0=환상적)
VISUAL_WORDS = {
    # 사실적 (low visual)
    "사실적": 0.15, "리얼": 0.15, "다큐": 0.1,
    "realistic": 0.15, "gritty": 0.2, "raw": 0.15,
    # 중간
    "세련된": 0.5, "스타일리시": 0.55,
    "stylish": 0.55,
    # 환상적 (high visual)
    "화려한": 0.85, "환상적": 0.9, "시각적": 0.8,
    "비주얼": 0.8, "스펙터클": 0.9, "CG": 0.85,
    "flashy": 0.85, "visual": 0.8, "spectacular": 0.9,
    "stunning": 0.85, "cgi": 0.85,
}

# 예산 단어 -> budget_scale 값 (0.0=저예산, 1.0=블록버스터)
BUDGET_WORDS = {
    # 저예산 (low budget)
    "저예산": 0.1, "인디": 0.15, "독립영화": 0.1, "소규모": 0.15,
    "low-budget": 0.1, "indie": 0.15, "independent": 0.15,
    # 중간
    "중간예산": 0.5, "보통규모": 0.5,
    "mid-budget": 0.5,
    # 대작 (high budget)
    "블록버스터": 0.9, "대작": 0.85, "대규모": 0.8, "초대형": 0.95,
    "blockbuster": 0.9, "big-budget": 0.85, "epic": 0.8,
}

# 스타파워 단어 -> star_power 값
STAR_WORDS = {
    "무명": 0.15, "신인": 0.2,
    "스타": 0.8, "대스타": 0.9, "유명배우": 0.8,
    "unknown": 0.15, "star": 0.8, "celebrity": 0.85,
    "a-list": 0.9,
}

# ═══════════════════════════════════════════════════════════════════
# 예시 검색어
# ═══════════════════════════════════════════════════════════════════

# 제목 검색 예시 (2026년 개봉 예정작)
TITLE_EXAMPLES = [m["title"] for m in config.TEST_MOVIES]

# 자유 텍스트 검색 예시 (7개)
TEXT_EXAMPLES = [
    "어두운 분위기의 SF 우주 생존 영화",
    "밝은 로맨틱 코미디 사랑 이야기",
    "빠른 전개의 첩보 액션 스파이 스릴러",
    "역사적 전쟁 드라마 심리극",
    "화려한 블록버스터 판타지 모험 마법",
    "가족 애니메이션 따뜻한 코미디",
    "범죄 스릴러 복수 미스터리 어두운",
]


# ═══════════════════════════════════════════════════════════════════
# 검색 엔진
# ═══════════════════════════════════════════════════════════════════

class MovieSearchEngine:
    """제목 검색 + 자유 텍스트 검색을 통합 제공하는 검색 엔진"""

    def __init__(self, embedding: MovieEmbedding, train_movies: list[dict]):
        """
        Args:
            embedding: 학습 완료된 MovieEmbedding (fit + transform 완료)
            train_movies: 학습 영화 리스트
        """
        self.embedding = embedding
        self.train_movies = train_movies
        self.train_data = {m["title"]: m for m in train_movies}
        self.train_titles = [m["title"] for m in train_movies]

        self.use_cuda = HAS_TORCH and CUDA_AVAILABLE
        self.device = torch.device("cuda") if self.use_cuda else None

    # ══════════════════════════════════════════════════════════════
    # [1] 제목 검색
    # ══════════════════════════════════════════════════════════════

    def search_by_title(self, title: str, top_k: int = None) -> dict:
        """
        영화 제목으로 유사한 영화를 검색한다.

        Args:
            title: 검색할 영화 제목
            top_k: 추천 수 (None이면 config.TOP_K)

        Returns:
            {"query_title": str, "query_movie": dict or None,
             "match_type": str, "results": [rec_dicts]}
        """
        k = top_k or config.TOP_K

        # 1) 학습 데이터에서 정확히 일치
        if title in self.train_data:
            movie = self.train_data[title]
            results = self._recommend_excluding(movie, title, k)
            return {
                "query_title": title,
                "query_movie": movie,
                "match_type": "exact_train",
                "results": results,
            }

        # 2) 테스트 데이터에서 정확히 일치
        test_data = {m["title"]: m for m in config.TEST_MOVIES}
        if title in test_data:
            movie = dict(test_data[title])
            if movie.get("critic_score") is None:
                movie["critic_score"] = 0.5
            if movie.get("audience_score") is None:
                movie["audience_score"] = 0.5
            # 임베딩에 벡터가 없으면 생성
            if title not in self.embedding.raw_vectors:
                self.embedding.transform([movie])
            results = self._recommend_from_vector(
                self.embedding.raw_vectors[title], title, k
            )
            return {
                "query_title": title,
                "query_movie": movie,
                "match_type": "exact_test",
                "results": results,
            }

        # 3) 부분 일치 (대소문자 무시)
        title_lower = title.lower()
        candidates = []
        for t in self.train_titles:
            if title_lower in t.lower():
                candidates.append(t)

        if len(candidates) == 1:
            movie = self.train_data[candidates[0]]
            results = self._recommend_excluding(movie, candidates[0], k)
            return {
                "query_title": candidates[0],
                "query_movie": movie,
                "match_type": "partial_train",
                "results": results,
            }
        elif len(candidates) > 1:
            return {
                "query_title": title,
                "query_movie": None,
                "match_type": "multiple_matches",
                "candidates": candidates[:10],
                "results": [],
            }

        # 4) 일치하는 영화 없음
        return {
            "query_title": title,
            "query_movie": None,
            "match_type": "not_found",
            "results": [],
        }

    def _recommend_excluding(self, movie: dict, exclude_title: str,
                             top_k: int) -> list[dict]:
        """학습 데이터 중 자기 자신을 제외하고 추천"""
        query_vec = self.embedding.raw_vectors[exclude_title]
        other_titles = [t for t in self.train_titles if t != exclude_title]

        sims = self._cosine_similarity_batch(query_vec, other_titles)
        sorted_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            t = other_titles[idx]
            train_movie = self.train_data[t]
            sim = float(sims[idx])
            shared_genres = list(set(movie["genres"]) & set(train_movie["genres"]))
            shared_kw = list(set(movie["keywords"]) & set(train_movie["keywords"]))
            explanation = self._generate_explanation(
                movie, train_movie, shared_genres, shared_kw
            )
            results.append({
                "rank": len(results) + 1,
                "title": t,
                "year": train_movie["year"],
                "genres": train_movie["genres"],
                "similarity": round(sim, 4),
                "shared_genres": shared_genres,
                "shared_keywords": shared_kw,
                "explanation": explanation,
            })
        return results

    def _recommend_from_vector(self, query_vec: np.ndarray,
                               query_title: str,
                               top_k: int) -> list[dict]:
        """임의의 벡터로부터 유사 영화 추천"""
        sims = self._cosine_similarity_batch(query_vec, self.train_titles)
        sorted_indices = np.argsort(sims)[::-1][:top_k]

        # 쿼리 정보 추출 (벡터에서 장르/키워드 역파싱)
        query_genres = self._vector_to_genres(query_vec)
        query_keywords = self._vector_to_keywords(query_vec)

        results = []
        for idx in sorted_indices:
            t = self.train_titles[idx]
            train_movie = self.train_data[t]
            sim = float(sims[idx])
            shared_genres = list(set(query_genres) & set(train_movie["genres"]))
            shared_kw = list(set(query_keywords) & set(train_movie["keywords"]))

            reasons = []
            if shared_genres:
                reasons.append(f"공통 장르: {', '.join(shared_genres)}")
            if shared_kw:
                reasons.append(f"공통 키워드: {', '.join(shared_kw)}")
            if not reasons:
                reasons.append("전체적인 특징 패턴이 유사")

            results.append({
                "rank": len(results) + 1,
                "title": t,
                "year": train_movie["year"],
                "genres": train_movie["genres"],
                "similarity": round(sim, 4),
                "shared_genres": shared_genres,
                "shared_keywords": shared_kw,
                "explanation": " | ".join(reasons),
            })
        return results

    # ══════════════════════════════════════════════════════════════
    # [2] 자유 텍스트 검색
    # ══════════════════════════════════════════════════════════════

    def search_by_text(self, text: str, top_k: int = None) -> dict:
        """
        자유 텍스트를 파싱하여 유사 영화를 검색한다.

        Args:
            text: 자유 입력 텍스트 (한국어/영어 혼용 가능)
            top_k: 추천 수 (None이면 config.TOP_K)

        Returns:
            {"query_text": str, "parsed": dict, "results": [rec_dicts]}
        """
        k = top_k or config.TOP_K

        # 텍스트 파싱
        parsed = self._parse_text(text)

        # 파싱 결과로 가상 영화 dict 생성
        pseudo_movie = self._build_pseudo_movie(parsed)

        # 54D 벡터 생성
        query_vec = self.embedding.movie_to_vector(pseudo_movie)

        # 유사도 검색
        results = self._recommend_from_vector(
            query_vec, f"[검색] {text}", k
        )

        return {
            "query_text": text,
            "parsed": parsed,
            "pseudo_movie": pseudo_movie,
            "results": results,
        }

    def _parse_text(self, text: str) -> dict:
        """
        자유 텍스트에서 장르, 키워드, 수치 특성을 추출한다.

        Returns:
            {"genres": [...], "keywords": [...],
             "mood": float, "tempo": float, "visual_style": float,
             "budget_scale": float, "star_power": float}
        """
        tokens = text.lower().replace(",", " ").replace(".", " ").split()

        genres = set()
        keywords = set()
        mood_vals = []
        tempo_vals = []
        visual_vals = []
        budget_vals = []
        star_vals = []

        # 단어 단위 매칭
        for token in tokens:
            # 장르 매칭
            if token in GENRE_ALIASES:
                genres.add(GENRE_ALIASES[token])
            # 키워드 매칭
            if token in KEYWORD_ALIASES:
                keywords.add(KEYWORD_ALIASES[token])
            # 분위기
            if token in MOOD_WORDS:
                mood_vals.append(MOOD_WORDS[token])
            # 전개 속도
            if token in TEMPO_WORDS:
                tempo_vals.append(TEMPO_WORDS[token])
            # 시각 스타일
            if token in VISUAL_WORDS:
                visual_vals.append(VISUAL_WORDS[token])
            # 예산 규모
            if token in BUDGET_WORDS:
                budget_vals.append(BUDGET_WORDS[token])
            # 스타파워
            if token in STAR_WORDS:
                star_vals.append(STAR_WORDS[token])

        # 2-gram 매칭 (하이픈 포함 키워드 처리)
        text_lower = text.lower()
        for alias, target in GENRE_ALIASES.items():
            if len(alias) > 1 and alias in text_lower:
                genres.add(target)
        for alias, target in KEYWORD_ALIASES.items():
            if len(alias) > 1 and alias in text_lower:
                keywords.add(target)
        for alias, val in MOOD_WORDS.items():
            if len(alias) > 1 and alias in text_lower:
                mood_vals.append(val)
        for alias, val in TEMPO_WORDS.items():
            if len(alias) > 1 and alias in text_lower:
                tempo_vals.append(val)
        for alias, val in VISUAL_WORDS.items():
            if len(alias) > 1 and alias in text_lower:
                visual_vals.append(val)
        for alias, val in BUDGET_WORDS.items():
            if len(alias) > 1 and alias in text_lower:
                budget_vals.append(val)
        for alias, val in STAR_WORDS.items():
            if len(alias) > 1 and alias in text_lower:
                star_vals.append(val)

        # 장르에서 키워드 자동 추가 (comedy -> comedy 키워드 등)
        genre_to_kw = {
            "Animation": "animation", "Comedy": "comedy",
            "War": "war", "Family": "family",
            "History": "historical",
        }
        for g in genres:
            if g in genre_to_kw:
                keywords.add(genre_to_kw[g])

        return {
            "genres": list(genres),
            "keywords": list(keywords),
            "mood": float(np.mean(mood_vals)) if mood_vals else 0.5,
            "tempo": float(np.mean(tempo_vals)) if tempo_vals else 0.5,
            "visual_style": float(np.mean(visual_vals)) if visual_vals else 0.5,
            "budget_scale": float(np.mean(budget_vals)) if budget_vals else 0.5,
            "star_power": float(np.mean(star_vals)) if star_vals else 0.5,
        }

    def _build_pseudo_movie(self, parsed: dict) -> dict:
        """파싱 결과로 가상 영화 dict 생성"""
        return {
            "title": "[검색 쿼리]",
            "year": 2026,
            "genres": parsed["genres"],
            "keywords": parsed["keywords"],
            "mood": parsed["mood"],
            "tempo": parsed["tempo"],
            "visual_style": parsed["visual_style"],
            "star_power": parsed["star_power"],
            "critic_score": 0.5,
            "audience_score": 0.5,
            "budget_scale": parsed["budget_scale"],
            "origin": "N/A",
        }

    # ══════════════════════════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════════════════════════

    def _cosine_similarity_batch(self, query_vec: np.ndarray,
                                  titles: list[str]) -> np.ndarray:
        """쿼리 벡터와 여러 영화 간 코사인 유사도 배열 반환"""
        train_vecs = np.array([self.embedding.raw_vectors[t] for t in titles])

        if self.use_cuda:
            q = torch.tensor(query_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            t = torch.tensor(train_vecs, dtype=torch.float32,
                             device=self.device)
            q_norm = F.normalize(q, dim=1)
            t_norm = F.normalize(t, dim=1)
            sims = (q_norm @ t_norm.T).squeeze(0)
            return sims.cpu().numpy()
        else:
            return cosine_similarity(
                query_vec.reshape(1, -1), train_vecs
            )[0]

    def _vector_to_genres(self, vec: np.ndarray) -> list[str]:
        """벡터에서 활성화된 장르 추출"""
        genres = []
        for i, g in enumerate(config.ALL_GENRES):
            if vec[i] > 0.5:
                genres.append(g)
        return genres

    def _vector_to_keywords(self, vec: np.ndarray) -> list[str]:
        """벡터에서 활성화된 키워드 추출"""
        offset = len(config.ALL_GENRES)
        keywords = []
        for i, kw in enumerate(config.ALL_KEYWORDS):
            if vec[offset + i] > 0.5:
                keywords.append(kw)
        return keywords

    def _generate_explanation(self, query: dict, train: dict,
                              shared_genres: list, shared_kw: list) -> str:
        """추천 이유 텍스트 생성"""
        reasons = []
        if shared_genres:
            reasons.append(f"공통 장르: {', '.join(shared_genres)}")
        if shared_kw:
            reasons.append(f"공통 키워드: {', '.join(shared_kw)}")

        mood_diff = abs(query.get("mood", 0.5) - train["mood"])
        if mood_diff < 0.15:
            reasons.append("유사한 분위기/톤")

        tempo_diff = abs(query.get("tempo", 0.5) - train["tempo"])
        if tempo_diff < 0.15:
            reasons.append("유사한 전개 속도")

        if not reasons:
            reasons.append("전체적인 특징 패턴이 유사")

        return " | ".join(reasons)

    # ══════════════════════════════════════════════════════════════
    # 콘솔 출력
    # ══════════════════════════════════════════════════════════════

    def print_title_search(self, title: str, top_k: int = None):
        """제목 검색 결과를 콘솔에 출력"""
        result = self.search_by_title(title, top_k)

        print("\n" + "=" * 72)
        print(f"[제목 검색] \"{result['query_title']}\"")

        if result["match_type"] == "not_found":
            print(f"   해당 영화를 찾을 수 없습니다: \"{title}\"")
            print("=" * 72)
            return result

        if result["match_type"] == "multiple_matches":
            print(f"   여러 영화가 일치합니다:")
            for c in result["candidates"]:
                print(f"      - {c}")
            print("=" * 72)
            return result

        match_labels = {
            "exact_train": "학습 데이터 (정확 일치)",
            "exact_test": "테스트 데이터 (2026년 개봉 예정)",
            "partial_train": "학습 데이터 (부분 일치)",
        }
        print(f"   매칭: {match_labels.get(result['match_type'], result['match_type'])}")

        movie = result["query_movie"]
        if movie:
            print(f"   장르: {', '.join(movie['genres'])}")
            print(f"   키워드: {', '.join(movie.get('keywords', []))}")
        print("-" * 72)

        self._print_results(result["results"])
        print("=" * 72)
        return result

    def print_text_search(self, text: str, top_k: int = None):
        """자유 텍스트 검색 결과를 콘솔에 출력"""
        result = self.search_by_text(text, top_k)

        print("\n" + "=" * 72)
        print(f"[자유 검색] \"{result['query_text']}\"")
        print("-" * 72)

        parsed = result["parsed"]
        if parsed["genres"]:
            print(f"   인식된 장르: {', '.join(parsed['genres'])}")
        if parsed["keywords"]:
            print(f"   인식된 키워드: {', '.join(parsed['keywords'])}")
        print(f"   분위기: {parsed['mood']:.2f}  |  "
              f"속도: {parsed['tempo']:.2f}  |  "
              f"시각: {parsed['visual_style']:.2f}")
        print(f"   예산규모: {parsed['budget_scale']:.2f}  |  "
              f"스타파워: {parsed['star_power']:.2f}")
        print("-" * 72)

        self._print_results(result["results"])
        print("=" * 72)
        return result

    def _print_results(self, results: list[dict]):
        """추천 결과 리스트 출력"""
        if not results:
            print("   추천 결과가 없습니다.")
            return

        print(f"추천 결과 (Top-{len(results)})")
        print("-" * 72)
        for r in results:
            print(f"  #{r['rank']}  {r['title']} ({r['year']})")
            print(f"      유사도: {r['similarity']:.4f}")
            print(f"      장르: {', '.join(r['genres'])}")
            print(f"      이유: {r['explanation']}")
            print()

    # ══════════════════════════════════════════════════════════════
    # 인터랙티브 모드
    # ══════════════════════════════════════════════════════════════

    def interactive(self):
        """인터랙티브 검색 모드 실행"""
        print("\n" + "=" * 72)
        print("영화 검색 엔진 - 인터랙티브 모드")
        print("=" * 72)
        print("\n검색 방법:")
        print("  1) 영화 제목을 입력하면 유사한 영화를 추천합니다.")
        print("  2) 장르, 키워드, 느낌 등을 자유롭게 입력해도 됩니다.")
        print("  종료: 'q' 또는 'quit' 입력\n")

        print("--- 제목 검색 예시 ---")
        for i, title in enumerate(TITLE_EXAMPLES, 1):
            print(f"  {i}. {title}")

        print("\n--- 자유 텍스트 검색 예시 ---")
        for i, text in enumerate(TEXT_EXAMPLES, 1):
            print(f"  {i}. {text}")

        print()

        while True:
            try:
                query = input("[검색] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n검색을 종료합니다.")
                break

            if not query:
                continue
            if query.lower() in ("q", "quit", "exit"):
                print("검색을 종료합니다.")
                break

            # 학습/테스트 데이터에 제목이 있는지 확인
            is_title = (
                query in self.train_data
                or query in {m["title"] for m in config.TEST_MOVIES}
            )

            # 부분 일치 확인
            if not is_title:
                q_lower = query.lower()
                partial = [t for t in self.train_titles if q_lower in t.lower()]
                if partial:
                    is_title = True

            if is_title:
                self.print_title_search(query)
            else:
                self.print_text_search(query)


def run_examples(embedding: MovieEmbedding, train_movies: list[dict]):
    """예시 검색을 실행하고 결과를 콘솔에 출력"""
    engine = MovieSearchEngine(embedding, train_movies)

    print("\n" + "#" * 72)
    print("# 제목 검색 예시 (2026년 개봉 예정작)")
    print("#" * 72)
    for title in TITLE_EXAMPLES:
        engine.print_title_search(title)

    print("\n" + "#" * 72)
    print("# 자유 텍스트 검색 예시")
    print("#" * 72)
    for text in TEXT_EXAMPLES:
        engine.print_text_search(text)
