"""
data_loader.py -- KMDB 데이터 로드 및 전처리
====================================================================
Data_new/movies/ 하위 554개 JSON 파일을 순회하여 영화 레코드를 정규화하고,
극영화/애니메이션/다큐멘터리만 필터링한다.
장르 매핑, 키워드 정리, 파생 수치 계산 후 캐시에 저장한다.
"""

import json
import glob
import os
import pickle
import re
from collections import Counter
from tqdm import tqdm

import config


def _safe_int(value, default=None):
    """문자열을 int로 안전하게 변환한다."""
    if value is None:
        return default
    try:
        v = int(str(value).strip())
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def _extract_plot(plots_data):
    """plots 딕셔너리에서 한국어/영어 줄거리를 추출한다."""
    plot_ko = ""
    plot_en = ""
    if not isinstance(plots_data, dict):
        return plot_ko, plot_en
    plot_list = plots_data.get("plot", [])
    if not isinstance(plot_list, list):
        return plot_ko, plot_en
    for p in plot_list:
        if not isinstance(p, dict):
            continue
        text = p.get("plotText", "").strip()
        lang = p.get("plotLang", "")
        if lang == "한국어" and text and not plot_ko:
            plot_ko = text
        elif lang == "영어" and text and not plot_en:
            plot_en = text
    return plot_ko, plot_en


def _extract_directors(directors_data):
    """directors 딕셔너리에서 감독 이름 목록을 추출한다."""
    names = []
    if not isinstance(directors_data, dict):
        return names
    director_list = directors_data.get("director", [])
    if not isinstance(director_list, list):
        return names
    for d in director_list:
        if isinstance(d, dict):
            nm = d.get("directorNm", "").strip()
            if nm:
                names.append(nm)
    return names


def _extract_actors(actors_data, max_count=10):
    """actors 딕셔너리에서 배우 이름 목록을 추출한다."""
    names = []
    if not isinstance(actors_data, dict):
        return names
    actor_list = actors_data.get("actor", [])
    if not isinstance(actor_list, list):
        return names
    for a in actor_list:
        if isinstance(a, dict):
            nm = a.get("actorNm", "").strip()
            if nm:
                names.append(nm)
            if len(names) >= max_count:
                break
    return names


def _normalize_genres(raw_genre_str):
    """원본 장르 문자열을 표준 장르 목록으로 변환한다."""
    if not raw_genre_str:
        return []
    parts = re.split(r"[/,]", raw_genre_str)
    mapped = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        std = config.GENRE_MAP.get(part)
        if std:
            mapped.add(std)
    return sorted(mapped)


def _normalize_keywords(raw_keyword_str):
    """원본 키워드 문자열을 정리된 키워드 목록으로 변환한다."""
    if not raw_keyword_str:
        return []
    parts = re.split(r"[|,]", raw_keyword_str)
    keywords = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 1:
            keywords.append(part)
    return keywords


def _find_poster_path(data_dir, year_str, month_str, movie_id, movie_seq, title):
    """로컬 포스터 이미지 경로를 찾는다."""
    image_dir = os.path.join(data_dir, year_str, month_str, "image")
    if not os.path.isdir(image_dir):
        return None, False

    prefix = f"{movie_id}_{movie_seq}_"
    try:
        for fname in os.listdir(image_dir):
            if fname.startswith(prefix) and fname.lower().endswith(".jpg"):
                full_path = os.path.join(image_dir, fname)
                has_image = "_noimage" not in fname.lower()
                return full_path, has_image
    except OSError:
        pass
    return None, False


def _compute_numeric_features(movie_dict):
    """파생 수치 피처를 계산한다 (0~1 범위)."""
    rt = movie_dict.get("runtime")
    if rt and rt > 0:
        movie_dict["runtime_norm"] = max(0.0, min(1.0, (rt - 30) / (240 - 30)))
    else:
        movie_dict["runtime_norm"] = 0.5

    year = movie_dict.get("year", 2000)
    movie_dict["year_norm"] = max(0.0, min(1.0, (year - 1980) / (2026 - 1980)))

    kw_count = len(movie_dict.get("keywords_matched", []))
    movie_dict["keyword_richness"] = min(kw_count / 10.0, 1.0)

    cast_count = len(movie_dict.get("actors", []))
    movie_dict["cast_size_norm"] = min(cast_count / 20.0, 1.0)

    genre_count = len(movie_dict.get("genres", []))
    movie_dict["genre_count_norm"] = min(genre_count / 5.0, 1.0)

    return movie_dict


def load_movies(data_dir=None, use_cache=True, show_progress=True):
    """
    KMDB 데이터를 로드하고 전처리한다.

    Parameters
    ----------
    data_dir : str, optional
        데이터 디렉토리 경로 (기본: config.DATA_DIR)
    use_cache : bool
        캐시 파일이 있으면 사용할지 여부
    show_progress : bool
        진행 바 표시 여부

    Returns
    -------
    list[dict]
        전처리된 영화 딕셔너리 목록
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    if use_cache and os.path.exists(config.CACHE_PATH):
        print(f"[data_loader] 캐시 로드: {config.CACHE_PATH}")
        with open(config.CACHE_PATH, "rb") as f:
            movies = pickle.load(f)
        print(f"[data_loader] {len(movies):,}편 로드 완료")
        return movies

    json_pattern = os.path.join(data_dir, "**", "json", "*.json")
    json_files = sorted(glob.glob(json_pattern, recursive=True))
    print(f"[data_loader] JSON 파일 {len(json_files)}개 발견")

    if not json_files:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_pattern}")

    keyword_set = set(config.ALL_KEYWORDS)

    movies = []
    skipped_type = 0
    skipped_no_title = 0

    iterator = tqdm(json_files, desc="데이터 로드") if show_progress else json_files

    for jf in iterator:
        parts = jf.replace("\\", "/").split("/")
        month_str = None
        year_str = None
        for p in parts:
            if re.match(r"\d{4}_\d{2}$", p):
                month_str = p
                year_str = p.split("_")[0]
                break

        with open(jf, encoding="utf-8") as f:
            try:
                raw_movies = json.load(f)
            except json.JSONDecodeError:
                continue

        if not isinstance(raw_movies, list):
            continue

        for m in raw_movies:
            if not isinstance(m, dict):
                continue

            movie_type = m.get("type", "").strip()
            if movie_type not in config.ALLOWED_TYPES:
                skipped_type += 1
                continue

            title = m.get("title", "").strip()
            if not title:
                skipped_no_title += 1
                continue

            movie_id = m.get("movieId", "").strip()
            movie_seq = m.get("movieSeq", "").strip()
            doc_id = m.get("DOCID", f"{movie_id}{movie_seq}")

            prod_year = _safe_int(m.get("prodYear"), default=2000)
            genres = _normalize_genres(m.get("genre", ""))
            raw_keywords = _normalize_keywords(m.get("keywords", ""))
            keywords_matched = [k for k in raw_keywords if k in keyword_set]

            plot_ko, plot_en = _extract_plot(m.get("plots"))
            directors = _extract_directors(m.get("directors"))
            actors = _extract_actors(m.get("actors"))
            runtime = _safe_int(m.get("runtime"))
            audi_acc = _safe_int(m.get("audiAcc"))
            nation = m.get("nation", "").strip()
            rating = m.get("rating", "").strip()

            poster_path = None
            has_poster = False
            if month_str and year_str:
                poster_path, has_poster = _find_poster_path(
                    data_dir, year_str, month_str,
                    movie_id, movie_seq, title
                )

            title_eng = m.get("titleEng", "").strip()
            awards = " ".join(filter(None, [
                m.get("Awards1", "").strip(),
                m.get("Awards2", "").strip(),
            ]))
            release_date = m.get("repRlsDate", "").strip()

            movie_dict = {
                "id": doc_id,
                "title": title,
                "title_eng": title_eng,
                "year": prod_year,
                "release_date": release_date,
                "genres": genres,
                "genre_raw": m.get("genre", ""),
                "keywords_all": raw_keywords,
                "keywords_matched": keywords_matched,
                "runtime": runtime,
                "rating": rating,
                "type": movie_type,
                "nation": nation,
                "directors": directors,
                "actors": actors,
                "plot_ko": plot_ko,
                "plot_en": plot_en,
                "poster_path": poster_path,
                "has_poster": has_poster,
                "audience_count": audi_acc,
                "awards": awards,
            }

            movie_dict = _compute_numeric_features(movie_dict)
            movies.append(movie_dict)

    print(f"[data_loader] 로드 완료: {len(movies):,}편")
    print(f"[data_loader] 제외: 타입 필터 {skipped_type:,}, 제목 없음 {skipped_no_title:,}")

    os.makedirs(os.path.dirname(config.CACHE_PATH), exist_ok=True)
    with open(config.CACHE_PATH, "wb") as f:
        pickle.dump(movies, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[data_loader] 캐시 저장: {config.CACHE_PATH}")

    return movies


def select_test_movies(movies, count=None, titles=None):
    """
    테스트 영화를 선정한다.

    Parameters
    ----------
    movies : list[dict]
    count : int, optional
    titles : list[str], optional

    Returns
    -------
    train_movies, test_movies
    """
    if count is None:
        count = config.TEST_MOVIE_COUNT

    if titles:
        title_set = set(titles)
        test = [m for m in movies if m["title"] in title_set]
        train = [m for m in movies if m["title"] not in title_set]
        return train, test

    candidates = [
        m for m in movies
        if m["year"] >= 2025 and m["plot_ko"] and len(m["genres"]) > 0
    ]
    candidates.sort(key=lambda x: x.get("release_date", ""), reverse=True)

    test = candidates[:count]
    test_ids = {m["id"] for m in test}
    train = [m for m in movies if m["id"] not in test_ids]

    return train, test


def get_data_field_info():
    """데이터 필드 정보를 반환한다 (시각화용)."""
    fields = {
        "genre": {
            "category": "embedding",
            "encoding": "one-hot",
            "embedding_dim": config.GENRE_DIM,
            "description": f"장르 원-핫 인코딩 ({config.GENRE_DIM}D)",
        },
        "keywords": {
            "category": "embedding",
            "encoding": "binary",
            "embedding_dim": config.KEYWORD_DIM,
            "description": f"키워드 바이너리 인코딩 ({config.KEYWORD_DIM}D)",
        },
        "runtime": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "런타임 정규화 (1D)",
        },
        "prodYear": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "연도 정규화 (1D)",
        },
        "actors_count": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "출연진 규모 정규화 (1D)",
        },
        "keywords_count": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "키워드 풍부도 (1D)",
        },
        "genre_count": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "장르 수 정규화 (1D)",
        },
        "plots": {
            "category": "embedding",
            "encoding": "text",
            "embedding_dim": config.TEXT_EMBED_DIM,
            "description": f"줄거리 텍스트 임베딩 ({config.TEXT_EMBED_DIM}D)",
        },
        "title": {
            "category": "identifier",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "영화 제목 (식별자)",
        },
        "nation": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "제작 국가",
        },
        "directors": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "감독",
        },
        "rating": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "관람 등급",
        },
        "posters": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "포스터 이미지",
        },
        "audiAcc": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "누적 관객수",
        },
        "Awards": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "수상 정보",
        },
    }
    return fields


if __name__ == "__main__":
    movies = load_movies(use_cache=False)
    print(f"\n총 {len(movies):,}편 로드")
    if movies:
        m = movies[0]
        print(f"예시: {m['title']} ({m['year']}) - {m['genres']}")
        print(f"  줄거리: {m['plot_ko'][:80]}..." if m["plot_ko"] else "  줄거리: 없음")
        print(f"  키워드(매칭): {m['keywords_matched']}")
        print(f"  수치: runtime_norm={m['runtime_norm']:.2f}, year_norm={m['year_norm']:.2f}")

    genre_counter = Counter()
    for m in movies:
        for g in m["genres"]:
            genre_counter[g] += 1
    print("\n장르 분포 (상위 15):")
    for g, c in genre_counter.most_common(15):
        print(f"  {g}: {c:,}")
