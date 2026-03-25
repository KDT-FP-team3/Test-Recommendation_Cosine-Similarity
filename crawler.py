#!/usr/bin/env python3
"""
crawler.py — 한국 개봉 영화 데이터 수집기
======================================================================
1980년 1월 ~ 2026년 3월 사이 한국에서 개봉한 영화를 수집하여
현재 임베딩 포맷(movie_data.py 호환)으로 변환 저장한다.

[데이터 소스]
  1차: TMDB (The Movie Database)    — 장르/출연/평점/예산/키워드
  2차: KOBIS (영화진흥위원회 오픈API)  — 한국 개봉 공식 DB, 누적 관객수

[API 키 발급]
  TMDB : https://www.themoviedb.org/settings/api  (무료 계정 가입 후 발급)
  KOBIS: https://www.kobis.or.kr/kobisopenapi/homepg/apiservice/regist/registApiKey.do

[설치]
    pip install requests python-dotenv tqdm

[환경 변수 설정]
    .env 파일에 API 키를 저장하면 CLI 인수 없이 실행 가능:
        TMDB_API_KEY=your_key_here

[실행]
    python crawler.py                                       # .env에서 키 로드
    python crawler.py --tmdb-key YOUR_TMDB_KEY              # CLI 인수 우선
    python crawler.py --resume                              # 이어서 수집
    python crawler.py --reset                               # 처음부터 다시
    python crawler.py --only-generate                       # 기존 JSON -> .py 변환만

[출력 파일]
    data/movies.json              — 수집 완료 데이터 (임베딩 포맷 + 추가 필드)
    data/checkpoint.json          — 재시작용 체크포인트
    data/movie_data_crawled.py    — movie_data.py 호환 Python 모듈

[주의]
    - TMDB는 쿼리당 최대 500페이지(10,000건) 제한이 있어 연도별로 분할 수집
    - 전체 수집 예상 시간: 약 2~4시간 (API 속도 제한 준수 기준)
    - Ctrl+C로 중단 시 체크포인트 자동 저장, --resume으로 재개 가능
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ======================================================================
# 설정 상수
# ======================================================================

START_DATE   = "1980-01-01"
END_DATE     = "2026-03-31"
REGION       = "KR"
OUTPUT_DIR   = Path("data")
CHECKPOINT   = OUTPUT_DIR / "checkpoint.json"
FINAL_JSON   = OUTPUT_DIR / "movies.json"
FINAL_PY     = OUTPUT_DIR / "movie_data_crawled.py"

TMDB_BASE    = "https://api.themoviedb.org/3"
KOBIS_BASE   = "http://www.kobis.or.kr/kobisopenapi/webservice/rest"

# TMDB: 40 req/10s 제한 → 0.3초 간격으로 안전하게 사용
TMDB_DELAY   = 0.30
KOBIS_DELAY  = 0.25
SAVE_EVERY   = 20   # 몇 페이지마다 중간 저장할지

# 연도 청크: TMDB 500페이지(10,000건) 제한 우회를 위해 기간을 분할
# 1980~1999 는 2년 단위, 2000~ 는 1년 단위
YEAR_CHUNKS: list[tuple[str, str]] = []
for y in range(1980, 2000, 2):
    YEAR_CHUNKS.append((f"{y}-01-01", f"{y+1}-12-31"))
for y in range(2000, 2027):
    end = "03-31" if y == 2026 else "12-31"
    YEAR_CHUNKS.append((f"{y}-01-01", f"{y}-{end}"))


# ======================================================================
# 장르 매핑
# ======================================================================

TMDB_GENRE_ID_MAP: dict[int, str] = {
    28:    "Action",
    12:    "Adventure",
    16:    "Animation",
    35:    "Comedy",
    80:    "Crime",
    18:    "Drama",
    14:    "Fantasy",
    27:    "Horror",
    9648:  "Mystery",
    10749: "Romance",
    878:   "Sci-Fi",
    53:    "Thriller",
    10752: "War",
    10402: "Musical",
    99:    "Documentary",
    10751: "Family",
    36:    "History",
}

KOBIS_GENRE_MAP: dict[str, str] = {
    "드라마":     "Drama",
    "판타지":     "Fantasy",
    "공포(호러)":  "Horror",
    "공포":      "Horror",
    "로맨스":     "Romance",
    "로멘스":     "Romance",
    "액션":      "Action",
    "코미디":     "Comedy",
    "스릴러":     "Thriller",
    "미스터리":    "Mystery",
    "SF":       "Sci-Fi",
    "SF/판타지":  "Sci-Fi",
    "애니메이션":   "Animation",
    "다큐멘터리":   "Documentary",
    "뮤지컬":     "Musical",
    "뮤지컬/공연":  "Musical",
    "범죄":      "Crime",
    "어드벤처":    "Adventure",
    "모험":      "Adventure",
    "전쟁":      "War",
    "역사":      "History",
    "가족":      "Family",
    "서부":      "Action",
    "성인물(에로)": "Drama",
}


# ======================================================================
# 키워드 매핑 — TMDB 키워드 문자열(소문자 부분 일치) → 프로젝트 키워드
# ======================================================================

ALL_KEYWORDS: list[str] = [
    "superhero", "space", "time-travel", "dystopia", "revenge",
    "love", "friendship", "survival", "war", "heist",
    "coming-of-age", "technology", "monster", "magic", "detective",
    "family", "conspiracy", "sports", "music", "nature",
    "robot", "politics", "psychological", "historical", "comedy",
    "apocalypse", "alien", "spy", "martial-arts", "animation",
]
ALL_KEYWORDS_SET = set(ALL_KEYWORDS)

# (TMDB 키워드 패턴 → 프로젝트 키워드) — 부분 문자열 일치
KW_PATTERNS: list[tuple[str, str]] = [
    # superhero
    ("superhero",           "superhero"),
    ("marvel",              "superhero"),
    ("dc comics",           "superhero"),
    ("based on comic",      "superhero"),
    # space
    ("outer space",         "space"),
    ("space travel",        "space"),
    ("space station",       "space"),
    ("astronaut",           "space"),
    ("nasa",                "space"),
    ("cosmos",              "space"),
    # time-travel
    ("time travel",         "time-travel"),
    ("time loop",           "time-travel"),
    ("time machine",        "time-travel"),
    # dystopia
    ("dystopia",            "dystopia"),
    ("totalitarian",        "dystopia"),
    ("post-apocalyptic",    "dystopia"),
    ("authoritarian",       "dystopia"),
    # revenge
    ("revenge",             "revenge"),
    ("vengeance",           "revenge"),
    # love / romance
    ("falling in love",     "love"),
    ("unrequited love",     "love"),
    ("romantic",            "love"),
    # friendship
    ("friendship",          "friendship"),
    ("buddy",               "friendship"),
    # survival
    ("survival",            "survival"),
    ("post-apocalypse",     "survival"),
    # war
    ("world war",           "war"),
    ("battle",              "war"),
    ("soldier",             "war"),
    ("military",            "war"),
    ("combat",              "war"),
    # heist
    ("heist",               "heist"),
    ("robbery",             "heist"),
    ("bank rob",            "heist"),
    ("con artist",          "heist"),
    # coming-of-age
    ("coming of age",       "coming-of-age"),
    ("growing up",          "coming-of-age"),
    ("teenager",            "coming-of-age"),
    ("adolescen",           "coming-of-age"),
    ("high school",         "coming-of-age"),
    # technology
    ("artificial intelligence", "technology"),
    ("hacker",              "technology"),
    ("cyberpunk",           "technology"),
    ("technolog",           "technology"),
    ("virtual reality",     "technology"),
    # monster
    ("monster",             "monster"),
    ("creature",            "monster"),
    ("zombie",              "monster"),
    ("kaiju",               "monster"),
    ("demon",               "monster"),
    # magic
    ("magic",               "magic"),
    ("witch",               "magic"),
    ("wizard",              "magic"),
    ("sorcerer",            "magic"),
    ("supernatural",        "magic"),
    ("enchant",             "magic"),
    # detective
    ("detective",           "detective"),
    ("murder investigat",   "detective"),
    ("serial killer",       "detective"),
    ("whodunit",            "detective"),
    ("private investigator","detective"),
    # family
    ("parent child",        "family"),
    ("father",              "family"),
    ("mother",              "family"),
    ("single parent",       "family"),
    ("family relationship", "family"),
    # conspiracy
    ("conspiracy",          "conspiracy"),
    ("cover-up",            "conspiracy"),
    ("government secret",   "conspiracy"),
    ("shadow organization", "conspiracy"),
    # sports
    ("sport",               "sports"),
    ("football",            "sports"),
    ("baseball",            "sports"),
    ("basketball",          "sports"),
    ("soccer",              "sports"),
    ("martial art",         "sports"),
    ("boxing",              "sports"),
    # music
    ("musician",            "music"),
    ("singer",              "music"),
    ("concert",             "music"),
    ("band",                "music"),
    ("rock music",          "music"),
    # nature
    ("wildlife",            "nature"),
    ("ocean",               "nature"),
    ("jungle",              "nature"),
    ("wilderness",          "nature"),
    ("environment",         "nature"),
    # robot
    ("robot",               "robot"),
    ("android",             "robot"),
    ("cyborg",              "robot"),
    ("mecha",               "robot"),
    # politics
    ("election",            "politics"),
    ("political",           "politics"),
    ("government",          "politics"),
    ("president",           "politics"),
    ("corruption",          "politics"),
    # psychological
    ("psychological",       "psychological"),
    ("mental illness",      "psychological"),
    ("obsession",           "psychological"),
    ("paranoia",            "psychological"),
    # historical
    ("based on true story", "historical"),
    ("period piece",        "historical"),
    ("19th century",        "historical"),
    ("20th century",        "historical"),
    ("historical event",    "historical"),
    # comedy
    ("slapstick",           "comedy"),
    ("parody",              "comedy"),
    ("black comedy",        "comedy"),
    ("satire",              "comedy"),
    # apocalypse
    ("apocalypse",          "apocalypse"),
    ("end of the world",    "apocalypse"),
    ("nuclear",             "apocalypse"),
    ("extinction",          "apocalypse"),
    # alien
    ("alien",               "alien"),
    ("extraterrestrial",    "alien"),
    ("ufo",                 "alien"),
    ("invasion",            "alien"),
    # spy
    ("espionage",           "spy"),
    ("secret agent",        "spy"),
    ("cia",                 "spy"),
    ("mi6",                 "spy"),
    ("intelligence agenc",  "spy"),
    # martial-arts
    ("kung fu",             "martial-arts"),
    ("karate",              "martial-arts"),
    ("taekwondo",           "martial-arts"),
    ("wushu",               "martial-arts"),
    # animation
    ("anime",               "animation"),
    ("computer animation",  "animation"),
    ("stop motion",         "animation"),
]


# ======================================================================
# 임베딩 수치 특징 자동 추론 (장르/메타데이터 기반)
# ======================================================================

def infer_mood(genres: list[str]) -> float:
    """
    분위기 추론 (0.0=매우 밝음, 1.0=매우 어두움)
    장르 조합으로 결정. 수동 검수 권장.
    """
    dark   = {"Horror", "Thriller", "Crime", "War", "Mystery"}
    bright = {"Comedy", "Animation", "Family", "Musical"}
    neutral_dark = {"Drama", "History"}
    neutral_bright = {"Romance", "Adventure", "Fantasy"}

    score = 0.50
    for g in genres:
        if g in dark:           score += 0.12
        elif g in neutral_dark: score += 0.05
        elif g in bright:       score -= 0.12
        elif g in neutral_bright: score -= 0.05

    return round(max(0.10, min(0.95, score)), 2)


def infer_tempo(genres: list[str], runtime: int = 0) -> float:
    """
    전개 속도 추론 (0.0=느린 전개, 1.0=빠른 전개)
    """
    fast = {"Action", "Thriller", "Horror", "Adventure"}
    slow = {"Drama", "Documentary", "History", "Musical"}

    score = 0.55
    for g in genres:
        if g in fast: score += 0.10
        if g in slow: score -= 0.10

    # 상영 시간 보정
    if runtime > 150:   score -= 0.08
    elif runtime < 90 and runtime > 0: score += 0.05

    return round(max(0.10, min(0.95, score)), 2)


def infer_visual_style(genres: list[str], budget_usd: int = 0) -> float:
    """
    시각 스타일 추론 (0.0=사실적, 1.0=환상적/스타일리시)
    """
    stylish   = {"Animation", "Fantasy", "Sci-Fi", "Musical", "Adventure"}
    realistic = {"Documentary", "Drama", "History", "Crime"}

    score = 0.50
    for g in genres:
        if g in stylish:   score += 0.12
        if g in realistic: score -= 0.08

    # 블록버스터 예산 보정
    if budget_usd > 150_000_000: score += 0.12
    elif budget_usd > 60_000_000: score += 0.06

    return round(max(0.10, min(1.00, score)), 2)


def infer_star_power(cast: list[dict]) -> float:
    """
    TMDB cast popularity 기반 스타파워 추론 (0~1)
    상위 5인의 popularity 평균을 로그 정규화
    """
    if not cast:
        return 0.30
    pops = [c.get("popularity", 0) for c in cast[:5] if c.get("popularity")]
    if not pops:
        return 0.20
    avg = sum(pops) / len(pops)
    # TMDB popularity: 0~100+ (간혹 수백까지)
    # log10(101) ≈ 2.004 → popularity 100이면 1.0
    score = math.log10(avg + 1) / math.log10(101)
    return round(max(0.05, min(1.00, score)), 2)


def infer_budget_scale(budget_usd: int) -> float:
    """예산(USD) → 0~1 버짓 스케일 (구간별 선형 보간)"""
    thresholds = [
        (0,           0.05),
        (500_000,     0.10),
        (3_000_000,   0.20),
        (10_000_000,  0.35),
        (30_000_000,  0.50),
        (70_000_000,  0.65),
        (130_000_000, 0.80),
        (200_000_000, 0.90),
        (300_000_000, 1.00),
    ]
    if budget_usd <= 0:
        return 0.10   # 정보 없음
    for i in range(len(thresholds) - 1):
        lo_usd, lo_val = thresholds[i]
        hi_usd, hi_val = thresholds[i + 1]
        if lo_usd <= budget_usd < hi_usd:
            t = (budget_usd - lo_usd) / (hi_usd - lo_usd)
            return round(lo_val + t * (hi_val - lo_val), 2)
    return 1.00


def infer_keywords(tmdb_kw_list: list[dict], genres: list[str]) -> list[str]:
    """
    TMDB 키워드 목록 + 장르로부터 프로젝트 키워드 추론
    """
    found: set[str] = set()

    # TMDB 키워드 → 패턴 매핑
    for kw in tmdb_kw_list:
        name = kw.get("name", "").lower()
        for pattern, proj_kw in KW_PATTERNS:
            if pattern in name:
                found.add(proj_kw)

    # 장르에서 직접 유추 가능한 키워드
    genre_keyword_map: dict[str, str] = {
        "War":         "war",
        "Animation":   "animation",
        "Musical":     "music",
        "History":     "historical",
        "Documentary": "historical",
        "Romance":     "love",
        "Comedy":      "comedy",
        "Fantasy":     "magic",
        "Sci-Fi":      "space",
    }
    for g in genres:
        if g in genre_keyword_map:
            found.add(genre_keyword_map[g])

    return sorted(found & ALL_KEYWORDS_SET)


def normalize_score(raw: float, scale: float = 10.0) -> float:
    """평점을 0~1로 정규화 (TMDB는 10점 만점)"""
    if not raw or raw <= 0:
        return 0.50
    return round(min(raw / scale, 1.00), 2)


def get_origin(production_countries: list[dict], original_language: str = "") -> str:
    """제작 국가 ISO 코드 반환"""
    if production_countries:
        return production_countries[0].get("iso_3166_1", "US")
    lang_to_country = {
        "ko": "KR", "ja": "JP", "zh": "CN",
        "fr": "FR", "de": "DE", "es": "ES",
        "it": "IT", "hi": "IN", "ru": "RU",
        "en": "US",
    }
    return lang_to_country.get(original_language, "US")


# ======================================================================
# TMDB API 클라이언트
# ======================================================================

class TMDBClient:
    """TMDB REST API 래퍼"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _get(self, endpoint: str, params: dict | None = None,
             retry: int = 3) -> dict:
        url = f"{TMDB_BASE}/{endpoint}"
        p   = {"api_key": self.api_key, "language": "ko-KR"}
        if params:
            p.update(params)
        for attempt in range(retry):
            try:
                resp = self.session.get(url, params=p, timeout=15)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 10))
                    print(f"  [TMDB] Rate-limit, {wait}초 대기...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(TMDB_DELAY)
                return resp.json()
            except requests.RequestException as e:
                print(f"  [TMDB 오류] {endpoint} (시도 {attempt+1}/{retry}): {e}",
                      file=sys.stderr)
                time.sleep(3)
        return {}

    def discover(self, date_start: str, date_end: str, page: int = 1) -> dict:
        """한국 개봉 영화 목록 (날짜 범위 + 페이지)"""
        return self._get("discover/movie", {
            "region":             REGION,
            "release_date.gte":   date_start,
            "release_date.lte":   date_end,
            "with_release_type":  "3|2",      # 극장 + 제한 상영
            "sort_by":            "release_date.asc",
            "vote_count.gte":     3,           # 극희귀 영화 제외
            "page":               page,
        })

    def movie_detail(self, movie_id: int) -> dict:
        """영화 상세 (credits + keywords + release_dates 포함)"""
        return self._get(f"movie/{movie_id}", {
            "append_to_response": "credits,keywords,release_dates",
        })


# ======================================================================
# KOBIS API 클라이언트 (한국 관객수 보완용)
# ======================================================================

class KOBISClient:
    """영화진흥위원회 오픈API 래퍼"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{KOBIS_BASE}/{endpoint}"
        p   = {"key": self.api_key, "itemPerPage": "100"}
        if params:
            p.update(params)
        try:
            resp = self.session.get(url, params=p, timeout=15)
            resp.raise_for_status()
            time.sleep(KOBIS_DELAY)
            return resp.json()
        except requests.RequestException as e:
            print(f"  [KOBIS 오류] {endpoint}: {e}", file=sys.stderr)
            time.sleep(2)
            return {}

    def search_movie_list(self, open_year: str, page: int = 1) -> dict:
        """특정 개봉 연도 영화 목록"""
        return self._get("movie/searchMovieList.json", {
            "curPage":     str(page),
            "openStartDt": open_year,
            "openEndDt":   open_year,
        })

    def get_movie_info(self, movie_cd: str) -> dict:
        """영화 상세 (감독, 배우, 관객수 포함)"""
        return self._get("movie/searchMovieInfo.json", {
            "movieCd": movie_cd,
        })


# ======================================================================
# TMDB 상세 데이터 → 임베딩 포맷 변환
# ======================================================================

def tmdb_detail_to_record(detail: dict) -> dict | None:
    """
    TMDB movie detail -> 프로젝트 임베딩 포맷 dict

    반환 구조:
        movie_data.py 호환 필드 + 추가 메타 필드
        (director, actors, runtime, budget_usd,
         audience_count, release_date, tmdb_id, ...)
    """
    if not detail or not detail.get("id"):
        return None

    title    = detail.get("title") or detail.get("original_title") or "Unknown"
    raw_date = detail.get("release_date") or ""
    year_str = raw_date[:4]
    year     = int(year_str) if year_str.isdigit() else 0
    if year < 1980 or year > 2026:
        return None

    # -- 장르
    genres = [
        TMDB_GENRE_ID_MAP[g["id"]]
        for g in detail.get("genres", [])
        if g["id"] in TMDB_GENRE_ID_MAP
    ]
    if not genres:
        genres = ["Drama"]

    # -- 출연/감독
    credits   = detail.get("credits") or {}
    cast      = credits.get("cast") or []
    crew      = credits.get("crew") or []
    directors = [p["name"] for p in crew if p.get("job") == "Director"]
    actors    = [p["name"] for p in cast[:5]]

    # -- 키워드
    tmdb_kws = (detail.get("keywords") or {}).get("keywords") or []
    keywords = infer_keywords(tmdb_kws, genres)

    # -- 평점 (TMDB vote_average: 0~10)
    vote_avg   = detail.get("vote_average") or 0
    vote_count = detail.get("vote_count")   or 0
    if vote_count >= 10:
        critic_score   = normalize_score(vote_avg)
        audience_score = normalize_score(vote_avg)
    else:
        critic_score   = None   # 데이터 부족 → 0.5 대체
        audience_score = None

    # -- 예산/규모
    budget_usd   = detail.get("budget") or 0
    budget_scale = infer_budget_scale(budget_usd)

    # -- 수치 특징 추론
    runtime      = detail.get("runtime") or 0
    mood         = infer_mood(genres)
    tempo        = infer_tempo(genres, runtime)
    visual_style = infer_visual_style(genres, budget_usd)
    star_power   = infer_star_power(cast)

    # -- 국가
    prod_countries = detail.get("production_countries") or []
    origin = get_origin(prod_countries, detail.get("original_language") or "en")

    return {
        # ---- movie_data.py 호환 핵심 필드 ----
        "title":          title,
        "year":           year,
        "genres":         genres,
        "mood":           mood,
        "tempo":          tempo,
        "visual_style":   visual_style,
        "star_power":     star_power,
        "critic_score":   critic_score,
        "audience_score": audience_score,
        "keywords":       keywords,
        "budget_scale":   budget_scale,
        "origin":         origin,
        # ---- 추가 메타 필드 ----
        "director":        directors[0] if directors else "",
        "actors":          actors,
        "runtime":         runtime,
        "budget_usd":      budget_usd,
        "audience_count":  0,           # KOBIS 보완 예정
        "release_date":    raw_date,
        "original_title":  detail.get("original_title") or "",
        "overview":        (detail.get("overview") or "")[:300],
        "tmdb_id":         detail.get("id"),
        "tmdb_vote_avg":   round(vote_avg, 1),
        "tmdb_vote_count": vote_count,
        "tmdb_popularity": round(detail.get("popularity") or 0, 2),
    }


def enrich_with_kobis(record: dict, kobis_movie_cd: str,
                      kobis: "KOBISClient") -> dict:
    """
    KOBIS movieCd로 관객수 / 감독 / 배우 보완
    한국 국적 영화에만 적용 권장
    """
    info_resp = kobis.get_movie_info(kobis_movie_cd)
    info = (info_resp.get("movieInfoResult") or {}).get("movieInfo") or {}
    if not info:
        return record

    # 누적 관객수 (audits 필드)
    audits = info.get("audits") or []
    for a in audits:
        aud_no = a.get("auditNo") or "0"
        try:
            record["audience_count"] = int(str(aud_no).replace(",", ""))
        except (ValueError, TypeError):
            pass
        break

    # 감독 보완
    directors = info.get("directors") or []
    if directors and not record.get("director"):
        record["director"] = directors[0].get("peopleNm") or ""

    # 배우 보완
    actors = info.get("actors") or []
    if actors and not record.get("actors"):
        record["actors"] = [
            a.get("peopleNm") for a in actors[:5] if a.get("peopleNm")
        ]

    return record


# ======================================================================
# 체크포인트 / 저장 유틸리티
# ======================================================================

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            return json.load(f)
    return {
        "done_chunks":   [],   # 완료된 (start, end) 청크 목록
        "collected_ids": [],   # 수집 완료된 TMDB movie_id 목록
    }


def save_checkpoint(cp: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(cp, f)


def load_movies() -> dict[int, dict]:
    if FINAL_JSON.exists():
        with open(FINAL_JSON, encoding="utf-8") as f:
            lst = json.load(f)
        return {int(m["tmdb_id"]): m for m in lst if m.get("tmdb_id")}
    return {}


def save_movies(movies: dict[int, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lst = sorted(movies.values(), key=lambda x: x.get("release_date") or "")
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


# ======================================================================
# Python 모듈 생성 (movie_data.py 호환)
# ======================================================================

def generate_python_module(movies: list[dict]) -> None:
    """
    수집된 영화 목록을 movie_data.py 호환 Python 모듈로 저장
    기존 MOVIES_TRAIN / MOVIES_TEST_2026 구조와 동일하게 생성
    """
    header = [
        '"""',
        "movie_data_crawled.py — 자동 생성 크롤링 데이터",
        f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"수집 기간: {START_DATE} ~ {END_DATE}  (한국 개봉 기준)",
        f"총 영화 수: {len(movies)}편",
        "",
        "mood         : 0.0(밝음) ~ 1.0(어두움)  — 장르 기반 자동 추론, 수동 검수 권장",
        "tempo        : 0.0(느린) ~ 1.0(빠른)   — 장르+상영시간 기반 자동 추론",
        "visual_style : 0.0(사실) ~ 1.0(환상)   — 장르+예산 기반 자동 추론",
        "star_power   : TMDB cast popularity 기반 로그 정규화",
        "critic_score : TMDB vote_average / 10  (투표수 < 10이면 None)",
        "audience_score: critic_score와 동일 (KOBIS 관객수로 보완 가능)",
        "budget_scale : TMDB budget 구간 선형 보간 (미상이면 0.10)",
        "keywords     : TMDB 키워드 + 장르로부터 자동 유추",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]

    lines = header[:]
    lines += [
        "# fmt: off",
        "MOVIES_CRAWLED: list[dict] = [",
    ]

    for m in movies:
        lines.append("    {")
        lines.append(f'        "title":          {json.dumps(m["title"], ensure_ascii=False)},')
        lines.append(f'        "year":           {m["year"]},')
        lines.append(f'        "genres":         {json.dumps(m["genres"])},')
        lines.append(f'        "mood":           {m["mood"]},')
        lines.append(f'        "tempo":          {m["tempo"]},')
        lines.append(f'        "visual_style":   {m["visual_style"]},')
        lines.append(f'        "star_power":     {m["star_power"]},')
        lines.append(f'        "critic_score":   {repr(m["critic_score"])},')
        lines.append(f'        "audience_score": {repr(m["audience_score"])},')
        lines.append(f'        "keywords":       {json.dumps(m["keywords"])},')
        lines.append(f'        "budget_scale":   {m["budget_scale"]},')
        lines.append(f'        "origin":         {json.dumps(m["origin"])},')
        # 추가 메타 필드
        lines.append(f'        "director":       {json.dumps(m.get("director", ""), ensure_ascii=False)},')
        lines.append(f'        "actors":         {json.dumps(m.get("actors", []), ensure_ascii=False)},')
        lines.append(f'        "audience_count": {m.get("audience_count", 0)},')
        lines.append(f'        "release_date":   {json.dumps(m.get("release_date", ""))},')
        lines.append(f'        "tmdb_id":        {m.get("tmdb_id", 0)},')
        lines.append("    },")

    lines += [
        "]",
        "# fmt: on",
        "",
        "",
        "def get_crawled_movies() -> list[dict]:",
        '    """크롤링된 전체 영화 데이터 반환"""',
        "    return MOVIES_CRAWLED",
        "",
        "",
        "def get_movies_by_year(start: int, end: int) -> list[dict]:",
        '    """특정 연도 범위 영화 필터링"""',
        "    return [m for m in MOVIES_CRAWLED if start <= m['year'] <= end]",
        "",
        "",
        "def get_korean_movies() -> list[dict]:",
        '    """한국 국적 영화만 반환"""',
        "    return [m for m in MOVIES_CRAWLED if m.get('origin') == 'KR']",
        "",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  -> {FINAL_PY}  ({len(movies)}편)")


# ======================================================================
# 메인 크롤링 파이프라인
# ======================================================================

def crawl_chunk(tmdb: TMDBClient, kobis: KOBISClient | None,
                date_start: str, date_end: str,
                movies: dict[int, dict], done_ids: set[int]) -> int:
    """
    단일 날짜 범위(청크) 크롤링
    TMDB discover 500페이지 한계 때문에 기간을 쪼개서 호출
    반환: 이번 청크에서 새로 수집된 영화 수
    """
    first = tmdb.discover(date_start, date_end, page=1)
    total_pages  = min(first.get("total_pages",  1), 500)
    new_count = 0

    for page in range(1, total_pages + 1):
        if page == 1:
            page_data = first
        else:
            page_data = tmdb.discover(date_start, date_end, page=page)

        for item in page_data.get("results") or []:
            mid = item.get("id")
            if not mid or mid in done_ids:
                continue

            detail = tmdb.movie_detail(mid)
            record = tmdb_detail_to_record(detail)
            done_ids.add(mid)

            if record is None:
                continue

            # KOBIS 보완: 한국 영화이고 KOBIS 키가 있을 때
            if kobis and record.get("origin") == "KR":
                year_str = str(record["year"])
                kb_resp  = kobis.search_movie_list(open_year=year_str)
                kb_list  = ((kb_resp.get("movieListResult") or {})
                            .get("movieList") or [])
                for km in kb_list:
                    en_title = (km.get("movieNmEn") or "").lower().strip()
                    orig     = (record.get("original_title") or "").lower().strip()
                    kr_title = (km.get("movieNm") or "").lower().strip()
                    title    = record["title"].lower().strip()
                    if en_title and (en_title == orig or en_title == title):
                        record = enrich_with_kobis(record, km["movieCd"], kobis)
                        break
                    if kr_title and (kr_title == title or kr_title == orig):
                        record = enrich_with_kobis(record, km["movieCd"], kobis)
                        break

            movies[mid] = record
            new_count  += 1


    return new_count


def run_crawl(tmdb: TMDBClient, kobis: KOBISClient | None) -> None:
    """전체 크롤링 오케스트레이션"""
    print("\n[STEP 1] 한국 개봉 영화 수집 시작")
    print(f"         범위: {START_DATE} ~ {END_DATE}  |  연도 청크: {len(YEAR_CHUNKS)}개")
    print("-" * 68)

    cp       = load_checkpoint()
    movies   = load_movies()
    done_ids = set(int(x) for x in cp.get("collected_ids", []))
    done_chunks = set(
        tuple(c) for c in cp.get("done_chunks", [])
    )

    remaining = [(ds, de) for ds, de in YEAR_CHUNKS
                 if (ds, de) not in done_chunks]

    if not remaining:
        print("  모든 청크가 이미 수집 완료되었습니다.")
        save_movies(movies)
        return

    pbar = tqdm(
        remaining,
        desc="수집",
        unit="chunk",
        ncols=72,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for ds, de in pbar:
        period = f"{ds[:4]}-{de[:4]}"
        pbar.set_postfix_str(f"{period}  {len(movies)}편", refresh=True)

        try:
            crawl_chunk(tmdb, kobis, ds, de, movies, done_ids)
        except KeyboardInterrupt:
            raise

        done_chunks.add((ds, de))

        # 체크포인트 저장
        cp["done_chunks"]   = [list(c) for c in done_chunks]
        cp["collected_ids"] = list(done_ids)
        save_checkpoint(cp)
        save_movies(movies)

    pbar.close()

    save_movies(movies)
    print(f"\n[완료] 총 {len(movies)}편 수집")

    # Python 모듈 생성
    print("\n[STEP 2] Python 모듈 생성")
    movie_list = sorted(movies.values(),
                        key=lambda x: x.get("release_date") or "")
    generate_python_module(movie_list)

    print("\n완료!")
    print(f"  데이터: {FINAL_JSON}")
    print(f"  모듈  : {FINAL_PY}")
    print("\n[다음 단계]")
    print("  1. data/movie_data_crawled.py 를 검토하고")
    print("     mood/tempo/visual_style 등 자동 추론 값을 수동 보정하세요.")
    print("  2. movie_data.py의 MOVIES_TRAIN을 대용량 데이터로 교체하려면")
    print("     get_crawled_movies()를 import해서 사용하세요.")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="한국 개봉 영화 데이터 크롤러 (1980-01 ~ 2026-03)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # .env에 TMDB_API_KEY 설정 후 실행 (권장)
  python crawler.py

  # CLI 인수로 키 직접 전달
  python crawler.py --tmdb-key abc123

  # 이전 수집 이어서 재개
  python crawler.py --resume

  # 처음부터 다시 수집
  python crawler.py --reset

  # 이미 수집된 movies.json을 Python 모듈로만 변환
  python crawler.py --only-generate
        """,
    )
    p.add_argument("--tmdb-key",       default="", help="TMDB API Key")
    p.add_argument("--kobis-key",      default="", help="KOBIS API Key (선택)")
    p.add_argument("--resume",         action="store_true",
                   help="체크포인트에서 이어서 수집")
    p.add_argument("--reset",          action="store_true",
                   help="체크포인트 초기화 후 처음부터")
    p.add_argument("--only-generate",  action="store_true",
                   help="크롤링 없이 data/movies.json -> .py 변환만 수행")
    return p.parse_args()


def main() -> None:
    load_dotenv()  # .env 파일에서 환경 변수 로드
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Python 모듈 생성만 수행
    if args.only_generate:
        if not FINAL_JSON.exists():
            print(f"오류: {FINAL_JSON} 가 없습니다. 먼저 크롤링을 실행하세요.")
            sys.exit(1)
        with open(FINAL_JSON, encoding="utf-8") as f:
            movie_list = json.load(f)
        generate_python_module(movie_list)
        return

    # CLI 인수 → 환경 변수 순서로 API 키 결정
    tmdb_key  = args.tmdb_key  or os.getenv("TMDB_API_KEY", "")
    kobis_key = args.kobis_key or os.getenv("KOBIS_API_KEY", "")

    if not tmdb_key:
        print("오류: TMDB API 키가 필요합니다.")
        print("  방법 1: .env 파일에 TMDB_API_KEY=your_key 설정")
        print("  방법 2: --tmdb-key 인수로 전달")
        print("  발급  : https://www.themoviedb.org/settings/api")
        sys.exit(1)

    # ── 체크포인트 초기화
    if args.reset:
        for f in [CHECKPOINT, FINAL_JSON]:
            if f.exists():
                f.unlink()
        print("체크포인트 및 기존 데이터 초기화 완료")

    # ── API 클라이언트 초기화
    tmdb  = TMDBClient(tmdb_key)
    kobis = KOBISClient(kobis_key) if kobis_key else None

    print("=" * 68)
    print(" 한국 개봉 영화 크롤러")
    print("=" * 68)
    print(f" TMDB  API: 활성화")
    print(f" KOBIS API: {'활성화 (관객수 보완)' if kobis else '비활성 (--kobis-key 로 추가 가능)'}")
    print(f" 수집 범위: {START_DATE} ~ {END_DATE}")
    print(f" 출력 경로: {OUTPUT_DIR.resolve()}")
    print("=" * 68)

    try:
        run_crawl(tmdb, kobis)
    except KeyboardInterrupt:
        print("\n\n중단됨. 체크포인트가 저장되었습니다.")
        print("  재개: python crawler.py --resume")
        sys.exit(0)


if __name__ == "__main__":
    main()
