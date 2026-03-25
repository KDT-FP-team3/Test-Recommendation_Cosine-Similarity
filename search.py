"""
search.py -- 영화 검색 엔진
====================================================================
두 가지 검색 방식을 지원한다:

[1] 제목 검색: 제목으로 기존 데이터 매칭 후 유사 영화 추천
[2] 자유 텍스트 검색: 한국어/영어 텍스트 -> 하이브리드 쿼리 벡터 -> 코사인 유사도

sentence-transformers로 텍스트를 직접 임베딩하며,
장르/키워드 별칭 사전으로 메타데이터 파싱도 병행한다.
"""

import numpy as np

from embedding import HybridMovieEmbedding
import config


# ===================================================================
# 한국어 장르 별칭 -> config.ALL_GENRES 이름
# ===================================================================
GENRE_ALIASES = {
    # 드라마
    "드라마": "드라마", "drama": "드라마",
    # 액션
    "액션": "액션", "action": "액션",
    # 코미디
    "코미디": "코미디", "코메디": "코미디", "comedy": "코미디",
    # 스릴러
    "스릴러": "스릴러", "thriller": "스릴러", "서스펜스": "스릴러",
    # 범죄
    "범죄": "범죄", "crime": "범죄",
    # 모험
    "모험": "모험", "어드벤처": "모험", "adventure": "모험",
    # 로맨스
    "로맨스": "로맨스", "멜로": "로맨스", "romance": "로맨스", "사랑": "로맨스",
    # 판타지
    "판타지": "판타지", "fantasy": "판타지",
    # SF
    "sf": "SF", "SF": "SF", "공상과학": "SF", "sci-fi": "SF", "사이파이": "SF",
    # 미스터리
    "미스터리": "미스터리", "mystery": "미스터리", "추리": "미스터리",
    # 가족
    "가족": "가족", "family": "가족",
    # 공포
    "공포": "공포", "호러": "공포", "horror": "공포",
    # 전쟁
    "전쟁": "전쟁", "war": "전쟁",
    # 뮤지컬
    "뮤지컬": "뮤지컬", "뮤직": "뮤지컬", "musical": "뮤지컬",
    # 사극
    "사극": "사극", "시대극": "사극",
    # 무협
    "무협": "무협", "martial-arts": "무협",
    # 애니메이션
    "애니메이션": "애니메이션", "애니": "애니메이션", "animation": "애니메이션",
    # 스포츠
    "스포츠": "스포츠", "sports": "스포츠",
    # 다큐멘터리
    "다큐멘터리": "다큐멘터리", "다큐": "다큐멘터리", "documentary": "다큐멘터리",
    # 인물
    "인물": "인물", "전기": "인물", "biography": "인물",
    # 역사
    "역사": "역사", "history": "역사",
    # 느와르
    "느와르": "느와르", "noir": "느와르",
    # 첩보
    "첩보": "첩보", "spy": "첩보", "스파이": "첩보",
    # 서부
    "서부": "서부", "western": "서부",
    # 재난
    "재난": "재난", "disaster": "재난",
    # 청춘
    "청춘": "청춘", "성장": "청춘",
    # 사회
    "사회": "사회", "social": "사회",
    # 에로
    "에로": "에로", "성인": "에로",
    # 종교
    "종교": "종교", "religion": "종교",
}

# ===================================================================
# 한국어 키워드 별칭 -> config.ALL_KEYWORDS 이름
# ===================================================================
KEYWORD_ALIASES = {
    "복수": "복수", "revenge": "복수",
    "살인": "살인", "murder": "살인",
    "경찰": "경찰", "police": "경찰",
    "납치": "납치", "kidnapping": "납치",
    "마약": "마약", "drugs": "마약",
    "불륜": "불륜", "affair": "불륜",
    "결혼": "결혼", "marriage": "결혼",
    "음모": "음모", "conspiracy": "음모",
    "사랑": "사랑", "love": "사랑",
    "우정": "우정", "friendship": "우정",
    "추적": "추적", "chase": "추적",
    "연쇄살인": "연쇄살인", "serial killer": "연쇄살인",
    "여행": "여행", "travel": "여행",
    "탈출": "탈출", "escape": "탈출",
    "킬러": "킬러", "killer": "킬러",
    "외계인": "외계인", "alien": "외계인",
    "실종": "실종", "missing": "실종",
    "테러": "테러", "terrorism": "테러",
    "전쟁": "전쟁", "war": "전쟁",
    "유령": "유령", "ghost": "유령",
    "마법": "마법", "magic": "마법",
    "모험": "모험", "adventure": "모험",
    "우주": "우주", "space": "우주",
    "로봇": "로봇", "robot": "로봇",
    "시간여행": "시간여행", "time travel": "시간여행",
    "좀비": "죽음", "zombie": "죽음",
    "생존": "탈출", "survival": "탈출",
    "해킹": "음모", "hacking": "음모",
    "가족": "가족", "family": "가족",
    "학교": "학교", "school": "학교",
    "친구": "친구", "friend": "친구",
    "형사": "형사", "detective": "형사",
    "마피아": "마피아", "mafia": "마피아",
    "암살": "암살", "assassination": "암살",
    "스파이": "음모",
    "심리": "기억상실",
    "생존자": "탈출",
    "블록버스터": "속편",
}


class MovieSearchEngine:
    """영화 검색 엔진"""

    def __init__(self, embedding, train_movies, test_movies=None, recommender=None):
        """
        Parameters
        ----------
        embedding : HybridMovieEmbedding
        train_movies : list[dict]
        test_movies : list[dict], optional
        recommender : MovieRecommender, optional
        """
        self.embedding = embedding
        self.train_movies = train_movies
        self.test_movies = test_movies or []
        self.recommender = recommender

        # 제목 -> ID 매핑
        self.title_to_id = {}
        self.id_to_movie = {}
        for m in train_movies:
            self.title_to_id[m["title"]] = m["id"]
            self.id_to_movie[m["id"]] = m
        for m in self.test_movies:
            self.title_to_id[m["title"]] = m["id"]
            self.id_to_movie[m["id"]] = m

    def search(self, query, top_k=None):
        """
        통합 검색: 제목 매칭 시 제목 검색, 아니면 자유 텍스트 검색.

        Returns
        -------
        results : list[dict]
        search_type : str ("title" or "text")
        parsed_info : dict (파싱 정보)
        """
        k = top_k or config.TOP_K
        query = query.strip()

        # 제목 검색 시도
        title_result = self._search_by_title(query, k)
        if title_result:
            return title_result, "title", {"matched_title": query}

        # 자유 텍스트 검색
        results, parsed = self._search_by_text(query, k)
        return results, "text", parsed

    def _search_by_title(self, title, top_k):
        """제목으로 검색"""
        # 정확 매칭
        if title in self.title_to_id:
            mid = self.title_to_id[title]
            vec = self.embedding.get_raw_vector(mid)
            if vec is not None:
                ranked = self.embedding.compute_similarity_to_train(vec)
                return self._format_results(ranked, top_k, exclude_id=mid)

        # 부분 매칭
        matches = []
        query_lower = title.lower()
        for t, mid in self.title_to_id.items():
            if query_lower in t.lower():
                matches.append((t, mid))

        if len(matches) == 1:
            mid = matches[0][1]
            vec = self.embedding.get_raw_vector(mid)
            if vec is not None:
                ranked = self.embedding.compute_similarity_to_train(vec)
                return self._format_results(ranked, top_k, exclude_id=mid)

        return None

    def _search_by_text(self, text, top_k):
        """자유 텍스트 검색"""
        parsed = self._parse_query(text)

        # 하이브리드 쿼리 벡터 생성
        query_vec = self.embedding.build_query_vector(
            genres=parsed.get("genres", []),
            keywords=parsed.get("keywords", []),
            numeric_values=parsed.get("numeric_values", {}),
            text=text,  # 전체 텍스트를 sentence-transformers로 임베딩
        )

        ranked = self.embedding.compute_similarity_to_train(query_vec)
        results = self._format_results(ranked, top_k)

        return results, parsed

    def _parse_query(self, text):
        """텍스트에서 장르/키워드/수치 정보를 파싱한다."""
        text_lower = text.lower()
        words = text_lower.split()

        genres = set()
        keywords = set()

        # 단어별 매칭
        for word in words:
            word_clean = word.strip(",.!?()[]\"'")
            if word_clean in GENRE_ALIASES:
                genres.add(GENRE_ALIASES[word_clean])
            if word_clean in KEYWORD_ALIASES:
                keywords.add(KEYWORD_ALIASES[word_clean])

        # 복합어 매칭 (2-3단어 조합)
        for alias, genre in GENRE_ALIASES.items():
            if alias in text_lower:
                genres.add(genre)
        for alias, kw in KEYWORD_ALIASES.items():
            if alias in text_lower:
                keywords.add(kw)

        return {
            "genres": list(genres),
            "keywords": list(keywords),
            "numeric_values": {},
            "original_text": text,
        }

    def _format_results(self, ranked, top_k, exclude_id=None):
        """순위 결과를 포맷팅한다."""
        results = []
        for mid, sim in ranked:
            if exclude_id and mid == exclude_id:
                continue
            if mid not in self.id_to_movie:
                continue
            movie = self.id_to_movie[mid]

            results.append({
                "rank": len(results) + 1,
                "id": mid,
                "title": movie["title"],
                "title_eng": movie.get("title_eng", ""),
                "year": movie["year"],
                "genres": movie["genres"],
                "similarity": round(sim, 4),
                "nation": movie.get("nation", ""),
                "directors": movie.get("directors", []),
                "poster_path": movie.get("poster_path"),
                "plot_ko": (movie.get("plot_ko", "") or "")[:200],
                "explanation": self._generate_explanation(movie),
            })
            if len(results) >= top_k:
                break

        return results

    def _generate_explanation(self, movie):
        """간단한 추천 이유 생성"""
        parts = []
        if movie.get("genres"):
            parts.append(f"장르: {', '.join(movie['genres'][:3])}")
        if movie.get("keywords_matched"):
            parts.append(f"키워드: {', '.join(movie['keywords_matched'][:3])}")
        if movie.get("nation"):
            parts.append(f"국가: {movie['nation']}")
        return " | ".join(parts) if parts else "전체적인 특징 패턴이 유사"


def run_interactive_search(embedding, train_movies, test_movies=None):
    """대화형 검색 콘솔"""
    engine = MovieSearchEngine(embedding, train_movies, test_movies)

    print("\n" + "=" * 60)
    print("  영화 검색 시스템 (종료: q, quit, exit)")
    print("  - 제목 검색: 영화 제목 입력")
    print("  - 자유 검색: 장르, 키워드, 설명 등 자유롭게 입력")
    print("=" * 60)

    while True:
        try:
            query = input("\n검색> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        results, search_type, parsed = engine.search(query)

        if search_type == "text":
            info = parsed
            if info.get("genres"):
                print(f"  [파싱] 장르: {', '.join(info['genres'])}")
            if info.get("keywords"):
                print(f"  [파싱] 키워드: {', '.join(info['keywords'])}")

        print(f"\n검색 결과 (Top-{len(results)}, 방식: {search_type})")
        print("-" * 60)

        for r in results:
            print(f"  #{r['rank']:2d}  {r['title']} ({r['year']})  "
                  f"유사도: {r['similarity']:.4f}")
            print(f"       {r['explanation']}")
