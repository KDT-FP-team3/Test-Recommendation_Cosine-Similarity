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
    "블랙코미디": "블랙코미디",
    "블랙 코미디": "블랙코미디",
    "실존인물": "실존인물",
    "실화": "실화바탕",
    "실화바탕": "실화바탕",
    "연쇄 살인": "연쇄살인",
    "삼각관계": "삼각관계",
    "삼각 관계": "삼각관계",
    "첫사랑": "첫사랑",
    "기억 상실": "기억상실",
    "범죄조직": "범죄조직",
    "범죄 조직": "범죄조직",
    "독립영화": "독립영화",
    "인디영화": "인디영화",
    "인디": "인디영화",
}


# ===================================================================
# 퍼지 매칭 헬퍼 (외부 의존성 없음)
# ===================================================================

# 흔한 한국어 오타 치환 규칙 (정규화 시 적용)
_TYPO_SUBS = [
    ("숀", "션"),       # 액숀 -> 액션
    ("코메디", "코미디"),
    ("호로", "호러"),
    ("에니메", "애니메"),
    ("에니", "애니"),
    ("쓰릴러", "스릴러"),
    ("멜로드라마", "멜로"),
    ("어드벤쳐", "어드벤처"),
    ("판타지아", "판타지"),
    ("다큐맨터리", "다큐멘터리"),
    ("다큐맨", "다큐멘"),
    ("뮤지컬영화", "뮤지컬"),
    ("로맨틱", "로맨스"),
]


def _normalize_korean(text):
    """공백 제거 + 흔한 오타 치환으로 정규화."""
    text = text.replace(" ", "").lower()
    for old, new in _TYPO_SUBS:
        text = text.replace(old, new)
    return text


def _decompose_hangul(ch):
    """한글 음절을 초성/중성/종성 인덱스로 분해. 비한글은 ord 값 그대로."""
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        offset = code - 0xAC00
        jong = offset % 28
        jung = (offset // 28) % 21
        cho = offset // (28 * 21)
        return (cho, jung, jong)
    return (code,)


def _to_jamo_seq(text):
    """문자열을 자모 시퀀스(int 리스트)로 변환."""
    seq = []
    for ch in text:
        seq.extend(_decompose_hangul(ch))
    return seq


def _jamo_edit_distance(a, b, max_dist=4):
    """자모 시퀀스 기반 Levenshtein 편집 거리. max_dist 초과 시 조기 종료."""
    sa, sb = _to_jamo_seq(a), _to_jamo_seq(b)
    la, lb = len(sa), len(sb)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    # DP (메모리 절약: 2행)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if sa[i - 1] == sb[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[lb]


def _get_fuzzy_threshold(token):
    """토큰 길이에 따른 퍼지 매칭 임계값."""
    syllables = sum(1 for c in token if 0xAC00 <= ord(c) <= 0xD7A3)
    if syllables <= 2 or len(token) <= 3:
        return 1   # 짧은 단어: 1자모만 허용 (공포 != 공부)
    elif syllables <= 3 or len(token) <= 6:
        return 2
    else:
        return 3


def _fuzzy_match_aliases(token, alias_dict):
    """자모 편집 거리 기반 퍼지 매칭. 가장 가까운 별칭 값 반환."""
    if len(token) < 2:
        return None
    threshold = _get_fuzzy_threshold(token)
    best_val = None
    best_dist = threshold + 1
    for alias_key, alias_val in alias_dict.items():
        d = _jamo_edit_distance(token, alias_key, max_dist=threshold)
        if d < best_dist:
            best_dist = d
            best_val = alias_val
    return best_val if best_dist <= threshold else None


# 정규화된 별칭 인덱스 (모듈 로드 시 1회 구축)
_NORM_GENRE_ALIASES = {_normalize_korean(k): v for k, v in GENRE_ALIASES.items()}
_NORM_KEYWORD_ALIASES = {_normalize_korean(k): v for k, v in KEYWORD_ALIASES.items()}


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
        """텍스트에서 장르/키워드/수치 정보를 파싱한다.

        4단계 매칭:
          1) 단어별 정확 매칭 (GENRE_ALIASES, KEYWORD_ALIASES)
          2) 부분 문자열 매칭 (복합어 대응)
          3) 정규화 매칭 (띄어쓰기 제거 + 오타 치환)
          4) 자모 퍼지 매칭 (편집 거리 기반, 미매칭 단어만)
        """
        text_lower = text.lower()
        words = text_lower.split()

        genres = set()
        keywords = set()
        fuzzy_corrections = []  # (원본, 보정값) 기록

        # --- Pass 1: 단어별 정확 매칭 ---
        matched_words = set()
        for word in words:
            word_clean = word.strip(",.!?()[]\"'")
            if word_clean in GENRE_ALIASES:
                genres.add(GENRE_ALIASES[word_clean])
                matched_words.add(word_clean)
            if word_clean in KEYWORD_ALIASES:
                keywords.add(KEYWORD_ALIASES[word_clean])
                matched_words.add(word_clean)

        # --- Pass 2: 부분 문자열 매칭 (복합어) ---
        for alias, genre in GENRE_ALIASES.items():
            if alias in text_lower:
                genres.add(genre)
        for alias, kw in KEYWORD_ALIASES.items():
            if alias in text_lower:
                keywords.add(kw)

        # --- Pass 3: 정규화 매칭 (띄어쓰기 + 오타 치환) ---
        text_norm = _normalize_korean(text_lower)
        for alias_norm, genre in _NORM_GENRE_ALIASES.items():
            if alias_norm in text_norm:
                if genre not in genres:
                    genres.add(genre)
                    fuzzy_corrections.append((alias_norm, f"장르:{genre}"))
        for alias_norm, kw in _NORM_KEYWORD_ALIASES.items():
            if alias_norm in text_norm:
                if kw not in keywords:
                    keywords.add(kw)
                    fuzzy_corrections.append((alias_norm, f"키워드:{kw}"))

        # --- Pass 4: 자모 퍼지 매칭 (미매칭 단어만) ---
        for word in words:
            word_clean = word.strip(",.!?()[]\"'")
            if word_clean in matched_words or len(word_clean) < 2:
                continue
            # 장르 퍼지
            fuzzy_genre = _fuzzy_match_aliases(word_clean, GENRE_ALIASES)
            if fuzzy_genre and fuzzy_genre not in genres:
                genres.add(fuzzy_genre)
                fuzzy_corrections.append((word_clean, f"장르:{fuzzy_genre}"))
            # 키워드 퍼지
            fuzzy_kw = _fuzzy_match_aliases(word_clean, KEYWORD_ALIASES)
            if fuzzy_kw and fuzzy_kw not in keywords:
                keywords.add(fuzzy_kw)
                fuzzy_corrections.append((word_clean, f"키워드:{fuzzy_kw}"))

        return {
            "genres": list(genres),
            "keywords": list(keywords),
            "numeric_values": {},
            "original_text": text,
            "fuzzy_corrections": fuzzy_corrections,
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
            if info.get("fuzzy_corrections"):
                corrections = [f'"{orig}" -> {corrected}'
                               for orig, corrected in info["fuzzy_corrections"]]
                print(f"  [보정] {', '.join(corrections)}")

        print(f"\n검색 결과 (Top-{len(results)}, 방식: {search_type})")
        print("-" * 60)

        for r in results:
            print(f"  #{r['rank']:2d}  {r['title']} ({r['year']})  "
                  f"유사도: {r['similarity']:.4f}")
            print(f"       {r['explanation']}")
