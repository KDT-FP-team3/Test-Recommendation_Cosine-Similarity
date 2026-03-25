"""
data_loader.py -- Stage 1: 데이터 로드 및 검증
====================================================================
크롤링된 JSON 데이터를 로드하고, 임베딩에 필요한 필수 필드를 검증한다.
누락된 수치 특징은 중립값(0.5)으로 대체한다.
"""

import json
from pathlib import Path

import config


# ═══════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════

# 임베딩에 필요한 필수 필드
_REQUIRED_FIELDS = [
    "title", "year", "genres", "mood", "tempo",
    "visual_style", "star_power", "keywords",
    "budget_scale", "origin"
]


def load_movies(json_path: str = None) -> list[dict]:
    """
    크롤링된 영화 데이터를 JSON 파일에서 로드.

    Args:
        json_path: JSON 파일 경로 (None이면 config.DATA_PATH 사용)

    Returns:
        유효한 영화 dict 리스트
    """
    path = Path(json_path or config.DATA_PATH)
    if not path.exists():
        print(f"[오류] {path} 파일이 존재하지 않습니다.")
        return []

    with open(path, encoding="utf-8") as f:
        movies = json.load(f)

    valid = []
    for m in movies:
        if all(m.get(k) is not None for k in _REQUIRED_FIELDS):
            # 누락 가능한 수치 필드 기본값 처리
            if m.get("critic_score") is None:
                m["critic_score"] = 0.5
            if m.get("audience_score") is None:
                m["audience_score"] = 0.5
            # runtime=0은 결측치 (실제 상영시간 0분은 불가능)
            if m.get("runtime") is None or m["runtime"] == 0:
                m["runtime"] = None
            valid.append(m)

    print(f"[Stage 1] 데이터 로드 완료: {len(valid)}편 (전체 {len(movies)}편 중 유효)")
    return valid


def load_test_movies() -> list[dict]:
    """테스트 영화(2026년 개봉 예정작) 반환"""
    test = []
    for m in config.TEST_MOVIES:
        movie = dict(m)
        if movie.get("critic_score") is None:
            movie["critic_score"] = 0.5
        if movie.get("audience_score") is None:
            movie["audience_score"] = 0.5
        test.append(movie)
    return test


# ═══════════════════════════════════════════════════════════════════
# 데이터 필드 정보 (다이어그램용)
# ═══════════════════════════════════════════════════════════════════

def get_data_field_info() -> dict:
    """
    크롤링 데이터의 24개 필드와 임베딩 사용 여부 매핑 반환.
    visualizer.py의 데이터 필드 다이어그램에서 사용.

    Returns:
        {field_name: {"category": str, "embedding_dim": int, "description": str}}
    """
    fields = {
        # --- 임베딩에 사용되는 필드 ---
        "genres": {
            "category": "embedding",
            "encoding": "one-hot",
            "embedding_dim": len(config.ALL_GENRES),
            "description": f"장르 원-핫 인코딩 ({len(config.ALL_GENRES)}D)",
        },
        "keywords": {
            "category": "embedding",
            "encoding": "binary",
            "embedding_dim": len(config.ALL_KEYWORDS),
            "description": f"키워드 바이너리 인코딩 ({len(config.ALL_KEYWORDS)}D)",
        },
        "mood": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "분위기 (0.0=밝음 ~ 1.0=어두움)",
        },
        "tempo": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "전개 속도 (0.0=느림 ~ 1.0=빠름)",
        },
        "visual_style": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "시각 스타일 (0.0=사실적 ~ 1.0=환상적)",
        },
        "star_power": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "출연진 인지도 (0.0~1.0)",
        },
        "critic_score": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "평론가 점수 (0.0~1.0, None->0.5)",
        },
        "audience_score": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "관객 점수 (0.0~1.0, None->0.5)",
        },
        "budget_scale": {
            "category": "embedding",
            "encoding": "numeric",
            "embedding_dim": 1,
            "description": "제작 규모 (0.0=저예산 ~ 1.0=블록버스터)",
        },
        # --- 메타데이터 (평가에만 사용) ---
        "origin": {
            "category": "metadata",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "제작 국가 (평가 시 다양성 분석)",
        },
        "title": {
            "category": "identifier",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "영화 제목 (식별자)",
        },
        "year": {
            "category": "identifier",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "개봉 연도 (식별자)",
        },
        # --- 참고용 필드 (임베딩 미사용) ---
        "director": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "감독",
        },
        "actors": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "주요 출연진",
        },
        "runtime": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "상영 시간(분)",
        },
        "budget_usd": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "제작비(USD)",
        },
        "audience_count": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "누적 관객수",
        },
        "release_date": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "개봉일",
        },
        "original_title": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "원제",
        },
        "overview": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "줄거리 요약",
        },
        "tmdb_id": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "TMDB ID",
        },
        "tmdb_vote_avg": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "TMDB 평균 평점",
        },
        "tmdb_vote_count": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "TMDB 투표 수",
        },
        "tmdb_popularity": {
            "category": "reference",
            "encoding": "none",
            "embedding_dim": 0,
            "description": "TMDB 인기도",
        },
    }
    return fields
