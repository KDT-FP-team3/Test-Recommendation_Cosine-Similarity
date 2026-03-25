"""
config.py -- 중앙 파라미터 관리
====================================================================
모든 조정 가능한 파라미터를 한 곳에서 관리한다.
대시보드(dashboard.py)에서 동적으로 변경 가능하며,
main.py는 이 파일의 값을 기본값으로 사용한다.
"""

import os

# ═══════════════════════════════════════════════════════════════════
# 경로
# ═══════════════════════════════════════════════════════════════════
DATA_PATH = "data/movies.json"
RESULTS_DIR = "results"

# ═══════════════════════════════════════════════════════════════════
# 임베딩 특징 목록
# ═══════════════════════════════════════════════════════════════════
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Musical", "Documentary",
    "Family", "History"
]

ALL_KEYWORDS = [
    "superhero", "space", "time-travel", "dystopia", "revenge",
    "love", "friendship", "survival", "war", "heist",
    "coming-of-age", "technology", "monster", "magic", "detective",
    "family", "conspiracy", "sports", "music", "nature",
    "robot", "politics", "psychological", "historical", "comedy",
    "apocalypse", "alien", "spy", "martial-arts", "animation"
]

NUMERIC_FEATURES = [
    "mood", "tempo", "visual_style", "star_power",
    "critic_score", "audience_score", "budget_scale"
]

# 총 벡터 차원 = 17(장르) + 30(키워드) + 7(수치) = 54

# ═══════════════════════════════════════════════════════════════════
# 테스트 영화 (2026년 개봉 예정작)
# ═══════════════════════════════════════════════════════════════════
TEST_MOVIES = [
    # Heartman (하트맨) → 2026-02-11 개봉 완료 → 학습 데이터로 이동
    # Project Hail Mary → 2026-03-15 개봉 완료 → 학습 데이터로 이동
    {
        "title": "Iron Lung",
        "year": 2026,
        "genres": ["Sci-Fi", "Horror"],
        "mood": 0.95, "tempo": 0.4, "visual_style": 0.7,
        "star_power": 0.3, "critic_score": None, "audience_score": None,
        "keywords": ["survival", "space", "psychological", "apocalypse"],
        "budget_scale": 0.15, "origin": "US"
    },
    {
        "title": "HUMINT (휴민트)",
        "year": 2026,
        "genres": ["Action", "Thriller"],
        "mood": 0.75, "tempo": 0.85, "visual_style": 0.6,
        "star_power": 0.65, "critic_score": None, "audience_score": None,
        "keywords": ["spy", "conspiracy", "war", "politics"],
        "budget_scale": 0.5, "origin": "KR"
    },
    {
        "title": "Nuremberg",
        "year": 2026,
        "genres": ["Drama", "History", "Thriller"],
        "mood": 0.85, "tempo": 0.45, "visual_style": 0.55,
        "star_power": 0.8, "critic_score": None, "audience_score": None,
        "keywords": ["war", "historical", "psychological", "politics"],
        "budget_scale": 0.5, "origin": "US"
    },
    {
        "title": "Peaky Blinders: The Immortal Man",
        "year": 2026,
        "genres": ["Crime", "Drama", "History"],
        "mood": 0.85, "tempo": 0.65, "visual_style": 0.75,
        "star_power": 0.85, "critic_score": None, "audience_score": None,
        "keywords": ["conspiracy", "revenge", "historical", "politics"],
        "budget_scale": 0.65, "origin": "UK"
    },
    {
        "title": "Ready or Not 2",
        "year": 2026,
        "genres": ["Comedy", "Horror", "Thriller"],
        "mood": 0.65, "tempo": 0.8, "visual_style": 0.55,
        "star_power": 0.5, "critic_score": None, "audience_score": None,
        "keywords": ["survival", "comedy", "family", "psychological"],
        "budget_scale": 0.3, "origin": "US"
    },
]

# ═══════════════════════════════════════════════════════════════════
# 특징 가중치 (대시보드에서 조정 가능)
# ═══════════════════════════════════════════════════════════════════
WEIGHT_GENRE = 1.0
WEIGHT_KEYWORD = 1.0
WEIGHT_NUMERIC = 1.0

# ═══════════════════════════════════════════════════════════════════
# 차원 축소
# ═══════════════════════════════════════════════════════════════════
PCA_COMPONENTS = 3          # PCA 축소 차원 (2 or 3)
USE_TSNE = False            # True: t-SNE, False: PCA
TSNE_PERPLEXITY = 30.0
TSNE_LEARNING_RATE = 200.0
TSNE_N_ITER = 1000

# ═══════════════════════════════════════════════════════════════════
# 군집화
# ═══════════════════════════════════════════════════════════════════
CLUSTER_METHOD = "kmeans"   # "kmeans" or "dbscan"
KMEANS_N_CLUSTERS = 8
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# ═══════════════════════════════════════════════════════════════════
# 추천
# ═══════════════════════════════════════════════════════════════════
TOP_K = 5

# ═══════════════════════════════════════════════════════════════════
# 평가 임계값
# ═══════════════════════════════════════════════════════════════════
THRESHOLD_GENRE_PRECISION = 0.4
THRESHOLD_AVG_SIMILARITY = 0.6
THRESHOLD_3D_DISTANCE = 4.0

# ═══════════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════════
GENRE_COLORS = {
    "Action":      "#E74C3C",
    "Adventure":   "#E67E22",
    "Animation":   "#2ECC71",
    "Comedy":      "#F1C40F",
    "Crime":       "#8E44AD",
    "Drama":       "#3498DB",
    "Fantasy":     "#1ABC9C",
    "Horror":      "#2C3E50",
    "Mystery":     "#9B59B6",
    "Romance":     "#E91E63",
    "Sci-Fi":      "#00BCD4",
    "Thriller":    "#795548",
    "War":         "#607D8B",
    "Musical":     "#FF9800",
    "Documentary": "#4CAF50",
    "Family":      "#FF5722",
    "History":     "#9E9E9E",
}

FIGURE_WIDTH = 1800
FIGURE_HEIGHT = 1100

# ═══════════════════════════════════════════════════════════════════
# 대시보드
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050

# ═══════════════════════════════════════════════════════════════════
# 민감도 분석
# ═══════════════════════════════════════════════════════════════════
SENSITIVITY_TOP_K = 20                                  # 비교용 추천 목록 크기
SENSITIVITY_LEVELS = {                                  # 상/중/하 가중치 레벨
    "하": 0.5,
    "중": 1.0,
    "상": 1.5,
}
# 3^3 = 27가지 조합 (장르, 키워드, 수치)
SENSITIVITY_SWEEP_VALUES = [0.5, 1.0, 1.5]

# 자유 텍스트 검색 쿼리 (민감도 분석용)
SENSITIVITY_TEXT_QUERIES = [
    "어두운 분위기의 SF 우주 생존 영화",
    "밝은 로맨틱 코미디 사랑 이야기",
    "빠른 전개의 첩보 액션 스파이 스릴러",
    "역사적 전쟁 드라마 심리극",
    "화려한 블록버스터 판타지 모험 마법",
]
