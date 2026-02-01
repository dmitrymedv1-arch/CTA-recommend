import streamlit as st
import requests
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
import time
import sqlite3
import os
from pathlib import Path
import hashlib
import joblib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import io
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Image
import xlsxwriter
from PIL import Image as PILImage

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки приложения
st.set_page_config(
    page_title="CTA Article Recommender Pro",
    page_icon="logo.jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Кастомные стили (более компактные)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    
    .step-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #667eea;
        margin-bottom: 15px;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.04);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        border: 1px solid #e0e0e0;
        height: 100%;
        min-height: 90px;
    }
    
    .metric-card h4 {
        font-size: 0.85rem;
        margin: 0 0 8px 0;
        color: #666;
    }
    
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #333;
        line-height: 1.2;
    }
    
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        border-left: 3px solid #4CAF50;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    .compact-button {
        padding: 8px 16px !important;
        font-size: 0.9rem !important;
        margin: 5px 0 !important;
        border-radius: 6px !important;
    }
    
    .compact-textarea {
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    
    .compact-select {
        font-size: 0.9rem !important;
    }
    
    .compact-slider {
        margin: 5px 0 !important;
    }
    
    .back-button {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 100;
    }
    
    .progress-container {
        background: #f5f5f5;
        border-radius: 8px;
        height: 6px;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 15px 0;
        font-size: 0.85rem;
        color: #666;
    }
    
    .step-indicator .active {
        color: #667eea;
        font-weight: 600;
    }
    
    .filter-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        margin: 2px;
        background: #e3f2fd;
        border-radius: 16px;
        font-size: 0.8rem;
        color: #1565c0;
    }
    
    .info-message {
        background: linear-gradient(135deg, #2196F315 0%, #0D47A115 100%);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #2196F3;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FF980015 0%, #EF6C0015 100%);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #FF9800;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    
    .topic-card {
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .topic-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    .topic-card.selected {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-color: #667eea;
        border-left: 4px solid #667eea;
    }
    
    .dataframe th {
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
    }
    
    .dataframe td {
        padding: 6px 12px !important;
        font-size: 0.85rem !important;
    }
    
    .download-buttons {
        display: flex;
        gap: 10px;
        margin: 15px 0;
    }
    
    .download-button {
        flex: 1;
    }
    
    /* Новые стили для фильтров */
    .filter-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #dee2e6;
    }
    
    .filter-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #667eea;
    }
    
    .filter-stats {
        background: white;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #ced4da;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .citation-checkbox-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .citation-checkbox-item {
        flex: 1;
        text-align: center;
    }
    
    .year-checkbox-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 15px;
    }
    
    .year-checkbox-item {
        background: white;
        border-radius: 6px;
        padding: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .year-checkbox-item:hover {
        border-color: #667eea;
        background-color: #f8f9ff;
    }
    
    .year-checkbox-item.selected {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-color: #667eea;
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Конфигурация OpenAlex API
OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = "your-email@example.com"  # Замените на ваш email
POLITE_POOL_HEADER = {'User-Agent': f'CTA-App (mailto:{MAILTO})'}

# Настройки rate limit
RATE_LIMIT_PER_SECOND = 8
BATCH_SIZE = 50
CURSOR_PAGE_SIZE = 200
MAX_WORKERS_ASYNC = 3
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 60

# Настройки кэширования
CACHE_DIR = Path("./cache")
CACHE_DB = CACHE_DIR / "openalex_cache.db"
CACHE_EXPIRY_DAYS = 30

# Инициализация кэш директории
CACHE_DIR.mkdir(exist_ok=True)

# Инициализация стоп-слов
nltk.download('stopwords', quiet=True)
COMMON_WORDS = {
    'study', 'studies', 'research', 'paper', 'article', 'review', 'analysis', 'analyses',
    'investigation', 'investigations', 'effect', 'effects', 'property', 'properties',
    'performance', 'behavior', 'behaviour', 'characterization', 'characterisation',
    'synthesis', 'development', 'preparation', 'fabrication', 'application', 'applications',
    'method', 'methods', 'approach', 'approaches', 'result', 'results', 'discussion',
    'conclusion', 'conclusions', 'introduction', 'experimental', 'experiment', 'experiments',
    'measurement', 'measurements', 'observation', 'observations', 'technique', 'techniques',
    'technology', 'technologies', 'material', 'materials', 'system', 'systems',
    'process', 'processes', 'structure', 'structures', 'model', 'models',
    'based', 'using', 'used', 'use', 'high', 'low', 'temperature', 'temperatures',
    'pressure', 'different', 'various', 'several', 'important', 'significant',
    'novel', 'new', 'recent', 'current', 'potential', 'possible', 'first',
    'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
    'tenth', 'good', 'better', 'best', 'poor', 'higher', 'lower', 'strong',
    'weak', 'large', 'small', 'great', 'major', 'minor', 'main', 'primary',
    'secondary', 'critical', 'essential', 'general', 'specific', 'special',
    'particular', 'similar', 'different', 'various', 'several', 'multiple',
    'numerous', 'common', 'unusual', 'typical', 'atypical', 'standard',
    'advanced', 'basic', 'fundamental', 'theoretical', 'practical', 'experimental',
    'computational', 'numerical', 'analytical', 'theoretical', 'practical'
}

ALL_STOPWORDS = set(stopwords.words('english')).union(COMMON_WORDS)

# ============================================================================
# ХИМИЧЕСКИЕ КОНСТАНТЫ ДЛЯ АНАЛИЗА
# ============================================================================

# Химические элементы (базовый список)
CHEM_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Yb3+', 'Eu3+', 'Tb3+', 'Dy3+', 'Sm3+', 'Nd3+', 'Er3+', 'Tm3+', 'Ho3+', 'Pr3+', 'Ce3+', 'Gd3+',
    # Общие ионы
    'OH', 'CO3', 'SO4', 'NO3', 'PO4', 'SiO4', 'NH4', 'CH3', 'C2H5', 'CN', 'SCN', 'NCS',
    # Функциональные группы
    'Me', 'Et', 'Pr', 'Bu', 'Ph', 'Bn', 'Ac', 'Bz', 'Cp', 'Cy'
}

# Regex для химических формул (e.g., La2NiO4, La1.8Pr0.2Ni0.7Co0.3O4, TiO2)
CHEM_FORMULA_PATTERN = r'\b(?:[A-Z][a-z]?\d*(?:\.\d+)?)+(?:[A-Z][a-z]?\d*(?:\.\d+)?)*\b'

# Органические паттерны (IUPAC fusion, bis-, tris-, substituted- и т.п.)
ORGANIC_PATTERNS = {
    'fusion_bracket': r'\[\d+,\d+[-a-z,′′′\d]+[a-z]*\]',          # [1,2-a:2′,1′-c], [1,2,4]triazolo[4,3-a]
    'bis_tris': r'\b(bis|tris|tetrakis|pentakis|hexakis)\[.*?\]',  # bis[...], tris[...]
    'substituted': r'\([Ss]ubstituted[^)]+\)',                     # (Substituted-Aminomethyl)
    'heterocycle_endings': r'(?:quinoxaline|pyridine|pyrrole|imidazole|thiazole|oxazole|quinoline|indole|purine|pteridine|carbazole|acridine|pyrazole|triazole|tetrazole|oxadiazole|thiadiazole|azepine|diazepine|azepinone)\b',
    'common_organic': r'\b(?:benzene|toluene|xylene|aniline|phenol|naphthalene|anthracene|phenanthrene|pyrene|coronene|fullerene|graphene|graphene oxide|reduced graphene oxide|carbon nanotube|MOF|COF|ZIF|perovskite|zeolite|mesoporous)\b',
}

# Веса для разных типов химических паттернов
CHEMICAL_WEIGHT = 2.5
ORGANIC_PATTERN_WEIGHT = 2.8
FUSION_MATCH_BONUS = 3.5
BIS_MATCH_BONUS = 2.5
CHEM_FORMULA_BONUS = 3.0

# ============================================================================
# КЭШИРОВАНИЕ НА УРОВНЕ SQLite
# ============================================================================

def init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS works_cache (
            doi TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_works_cache (
            topic_id TEXT,
            cursor_key TEXT,
            data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            PRIMARY KEY (topic_id, cursor_key)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics_cache (
            topic_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_works_expires ON works_cache(expires_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_works_expires ON topic_works_cache(expires_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topics_expires ON topics_cache(expires_at)')
    
    conn.commit()
    conn.close()

def get_cache_key(prefix: str, key: str) -> str:
    return hashlib.md5(f"{prefix}:{key}".encode()).hexdigest()

@st.cache_resource
def get_db_connection():
    init_cache_db()
    return sqlite3.connect(CACHE_DB, check_same_thread=False)

def cache_work(doi: str, data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
    
    cursor.execute('''
        INSERT OR REPLACE INTO works_cache (doi, data, expires_at)
        VALUES (?, ?, ?)
    ''', (doi, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_work(doi: str) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT data FROM works_cache 
        WHERE doi = ? AND (expires_at IS NULL OR expires_at > ?)
    ''', (doi, datetime.now()))
    
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])
    return None

def cache_topic_works(topic_id: str, cursor_key: str, data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=7)
    
    cursor.execute('''
        INSERT OR REPLACE INTO topic_works_cache (topic_id, cursor_key, data, expires_at)
        VALUES (?, ?, ?, ?)
    ''', (topic_id, cursor_key, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_topic_works(topic_id: str, cursor_key: str) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT data FROM topic_works_cache 
        WHERE topic_id = ? AND cursor_key = ? 
        AND (expires_at IS NULL OR expires_at > ?)
    ''', (topic_id, cursor_key, datetime.now()))
    
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])
    return None

def cache_topic_stats(topic_id: str, data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=30)
    
    cursor.execute('''
        INSERT OR REPLACE INTO topics_cache (topic_id, data, expires_at)
        VALUES (?, ?, ?)
    ''', (topic_id, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_topic_stats(topic_id: str) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT data FROM topics_cache 
        WHERE topic_id = ? AND (expires_at IS NULL OR expires_at > ?)
    ''', (topic_id, datetime.now()))
    
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])
    return None

def clear_old_cache():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Преобразуем datetime в строку в ISO формате для SQLite
    now_str = datetime.now().isoformat(' ', 'seconds')
    
    cursor.execute('DELETE FROM works_cache WHERE expires_at <= ?', (now_str,))
    cursor.execute('DELETE FROM topic_works_cache WHERE expires_at <= ?', (now_str,))
    cursor.execute('DELETE FROM topics_cache WHERE expires_at <= ?', (now_str,))
    
    conn.commit()

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ПАРСИНГА ДИАПАЗОНОВ ЦИТИРОВАНИЙ
# ============================================================================

def parse_citation_ranges(range_str: str) -> List[Tuple[int, int]]:
    """
    Парсит строку диапазонов цитирований в список кортежей.
    
    Примеры:
    "0" -> [(0, 0)]
    "0-3" -> [(0, 3)]
    "1,3-5" -> [(1, 1), (3, 5)]
    "0-1,3-4" -> [(0, 1), (3, 4)]
    "0,1,2,3" -> [(0, 0), (1, 1), (2, 2), (3, 3)]
    """
    ranges = []
    
    if not range_str or range_str.strip() == "":
        return [(0, 10)]  # По умолчанию
    
    # Разделяем по запятым
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Диапазон
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                
                # Проверяем что end >= start и оба в пределах 0-10
                if start <= end and 0 <= start <= 10 and 0 <= end <= 10:
                    ranges.append((start, end))
                else:
                    logger.warning(f"Invalid range: {part}. Using default.")
                    ranges.append((0, 10))
            except ValueError:
                logger.warning(f"Could not parse range: {part}. Using default.")
                ranges.append((0, 10))
        else:
            # Одиночное значение
            try:
                value = int(part.strip())
                if 0 <= value <= 10:
                    ranges.append((value, value))
                else:
                    logger.warning(f"Value out of range: {value}. Using default.")
                    ranges.append((0, 10))
            except ValueError:
                logger.warning(f"Could not parse value: {part}. Using default.")
                ranges.append((0, 10))
    
    # Удаляем дубликаты и сортируем
    unique_ranges = list(set(ranges))
    unique_ranges.sort(key=lambda x: x[0])
    
    return unique_ranges if unique_ranges else [(0, 10)]

def format_citation_ranges(ranges: List[Tuple[int, int]]) -> str:
    """
    Форматирует список диапазонов в читаемую строку.
    """
    if not ranges:
        return "0-10"
    
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    
    return ", ".join(parts)

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ОБНОВЛЕННОГО АНАЛИЗА С ФИЛЬТРАЦИЕЙ НА СТОРОНЕ API
# ============================================================================

def build_openalex_filter(topic_id: str, selected_years: List[int], 
                         selected_citations: List[Tuple[int, int]]) -> str:
    """
    Строит фильтр для OpenAlex API на основе выбранных параметров.
    
    Args:
        topic_id: Идентификатор темы (например, "T10366")
        selected_years: Список выбранных годов [2022, 2023, 2024]
        selected_citations: Список диапазонов цитирований [(0,0), (1,2), ...]
    
    Returns:
        Строка фильтра для OpenAlex API
    """
    filter_parts = [f"topics.id:{topic_id}"]
    
    # Добавляем фильтр по годам
    if selected_years:
        years_str = "|".join(map(str, selected_years))
        filter_parts.append(f"publication_year:{years_str}")
    
    # Добавляем фильтр по цитированиям
    if selected_citations:
        cites_str_parts = []
        for start, end in selected_citations:
            if start == end:
                cites_str_parts.append(str(start))
            else:
                cites_str_parts.append(f"{start}-{end}")
        
        if cites_str_parts:
            cites_str = "|".join(cites_str_parts)
            filter_parts.append(f"cited_by_count:{cites_str}")
    
    return ",".join(filter_parts)

def get_topic_total_works_count(topic_id: str) -> int:
    """
    Получает общее количество работ по теме из OpenAlex.
    
    Args:
        topic_id: Идентификатор темы
    
    Returns:
        Общее количество работ по теме
    """
    # Проверяем кэш
    cache_key = f"topic_total_{topic_id}"
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT data FROM topics_cache 
        WHERE topic_id = ? AND (expires_at IS NULL OR expires_at > ?)
    ''', (cache_key, datetime.now()))
    
    result = cursor.fetchone()
    if result:
        return int(json.loads(result[0]))
    
    # Если нет в кэше, запрашиваем из API
    try:
        url = f"{OPENALEX_BASE_URL}/topics/{topic_id}"
        response = requests.get(url, headers=POLITE_POOL_HEADER, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            works_count = data.get('works_count', 0)
            
            # Сохраняем в кэш
            expires_at = datetime.now() + timedelta(days=7)
            cursor.execute('''
                INSERT OR REPLACE INTO topics_cache (topic_id, data, expires_at)
                VALUES (?, ?, ?)
            ''', (cache_key, str(works_count), expires_at))
            conn.commit()
            
            return works_count
        else:
            logger.error(f"Error fetching topic stats: {response.status_code}")
            return 0
    except Exception as e:
        logger.error(f"Error in get_topic_total_works_count: {str(e)}")
        return 0

def fetch_filtered_works_by_topic(
    topic_id: str,
    years_filter: List[int],
    citations_filter: List[Tuple[int, int]],
    max_results: Optional[int] = None,
    progress_callback=None
) -> Tuple[List[dict], int]:
    """
    Загружает работы по теме с фильтрацией на стороне API.
    
    Args:
        topic_id: Идентификатор темы
        years_filter: Список годов для фильтрации
        citations_filter: Список диапазонов цитирований
        max_results: Максимальное количество результатов (None = все)
        progress_callback: Функция обратного вызова для прогресса
    
    Returns:
        Кортеж (список работ, общее количество после фильтров)
    """
    # Строим фильтр для API
    filter_str = build_openalex_filter(topic_id, years_filter, citations_filter)
    
    # Ключ кэша на основе фильтров
    cache_key = f"filtered_{topic_id}_{hashlib.md5(filter_str.encode()).hexdigest()[:16]}"
    
    # Проверяем кэш
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT data FROM topic_works_cache 
        WHERE topic_id = ? AND cursor_key = ? 
        AND (expires_at IS NULL OR expires_at > ?)
    ''', (topic_id, cache_key, datetime.now()))
    
    result = cursor.fetchone()
    if result:
        cached_data = json.loads(result[0])
        works_list = cached_data.get('works', [])
        total_count = cached_data.get('total_count', 0)
        
        if max_results and len(works_list) >= max_results:
            logger.info(f"Using cached filtered data for topic {topic_id}")
            return works_list[:max_results] if max_results else works_list, total_count
    
    # Если нет в кэше, загружаем с API
    logger.info(f"Fetching filtered works for topic {topic_id}")
    logger.info(f"Filter: {filter_str}")
    
    all_works = []
    cursor_param = "*"
    page_count = 0
    total_count = 0
    
    try:
        while True:
            if max_results and len(all_works) >= max_results:
                break
                
            page_count += 1
            
            # Формируем URL с фильтрами
            params = {
                "filter": filter_str,
                "per-page": CURSOR_PAGE_SIZE,
                "cursor": cursor_param,
                "mailto": MAILTO
            }
            
            url = f"{OPENALEX_BASE_URL}/works"
            response = requests.get(url, params=params, headers=POLITE_POOL_HEADER, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Error fetching works: {response.status_code}")
                break
            
            data = response.json()
            
            # Получаем общее количество на первой странице
            if page_count == 1:
                total_count = data.get('meta', {}).get('count', 0)
                logger.info(f"Total works after filters: {total_count}")
                
                if total_count == 0:
                    return [], 0
            
            works = data.get('results', [])
            if not works:
                break
            
            all_works.extend(works)
            
            # Вызываем callback прогресса
            if progress_callback and total_count > 0:
                progress = min(len(all_works) / min(total_count, max_results or total_count), 1.0)
                progress_callback(progress, len(all_works), page_count, total_count)
            
            logger.info(f"Page {page_count}: got {len(works)} works, total: {len(all_works)}/{total_count}")
            
            # Получаем следующий курсор
            next_cursor = data.get('meta', {}).get('next_cursor')
            if not next_cursor:
                break
            
            cursor_param = next_cursor
            
            # Небольшая задержка для соблюдения rate limit
            time.sleep(0.1)
        
        # Сохраняем в кэш
        if all_works:
            cache_data = {
                'works': all_works,
                'total_count': total_count,
                'filter': filter_str,
                'timestamp': datetime.now().isoformat()
            }
            
            expires_at = datetime.now() + timedelta(days=3)
            cursor.execute('''
                INSERT OR REPLACE INTO topic_works_cache (topic_id, cursor_key, data, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (topic_id, cache_key, json.dumps(cache_data), expires_at))
            conn.commit()
        
        # Ограничиваем результаты если нужно
        result_works = all_works[:max_results] if max_results else all_works
        
        return result_works, total_count
        
    except Exception as e:
        logger.error(f"Error in fetch_filtered_works_by_topic: {str(e)}")
        return all_works, total_count

# ============================================================================
# ASYNCIO + AIOHTTP
# ============================================================================

class OpenAlexAsyncClient:
    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_WORKERS_ASYNC)
        self.request_count = 0
        self.start_time = time.time()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=POLITE_POOL_HEADER,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=INITIAL_DELAY, max=MAX_DELAY),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def make_request(self, url: str) -> Optional[dict]:
        async with self.semaphore:
            elapsed = time.time() - self.start_time
            expected_time = self.request_count / RATE_LIMIT_PER_SECOND
            
            if elapsed < expected_time:
                wait_time = expected_time - elapsed
                await asyncio.sleep(wait_time)
            
            try:
                async with self.session.get(url) as response:
                    self.request_count += 1
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429
                        )
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        logger.warning(f"Resource not found: {url}")
                        return None
                    else:
                        logger.error(f"HTTP {response.status}: {url}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout: {url}")
                raise
            except Exception as e:
                logger.error(f"Error: {url} - {str(e)}")
                raise
    
    async def fetch_works_by_dois_batch(self, dois: List[str]) -> List[Optional[dict]]:
        if not dois:
            return []
        
        cached_results = []
        uncached_dois = []
        
        for doi in dois:
            cached = get_cached_work(doi)
            if cached:
                cached_results.append(cached)
            else:
                uncached_dois.append(doi)
        
        if not uncached_dois:
            return cached_results
        
        logger.info(f"Fetching {len(uncached_dois)} works via batch API")
        
        doi_filter = "|".join(uncached_dois)
        url = f"{OPENALEX_BASE_URL}/works?filter=doi:{doi_filter}&per-page=200"
        
        try:
            data = await self.make_request(url)
            if data and 'results' in data:
                results = data['results']
                
                for work in results:
                    doi = work.get('doi', '').replace('https://doi.org/', '')
                    if doi:
                        cache_work(doi, work)
                
                doi_to_work = {w.get('doi', '').replace('https://doi.org/', ''): w for w in results}
                batch_results = []
                
                for doi in uncached_dois:
                    if doi in doi_to_work:
                        batch_results.append(doi_to_work[doi])
                    else:
                        try:
                            work_data = await self.fetch_single_work(doi)
                            batch_results.append(work_data)
                        except:
                            batch_results.append(None)
                
                return cached_results + batch_results
            else:
                return cached_results + [None] * len(uncached_dois)
                
        except Exception as e:
            logger.error(f"Batch fetch error: {str(e)}")
            return cached_results + [None] * len(uncached_dois)
    
    async def fetch_single_work(self, doi: str) -> Optional[dict]:
        cached = get_cached_work(doi)
        if cached:
            return cached
        
        url = f"{OPENALEX_BASE_URL}/works/https://doi.org/{doi}"
        data = await self.make_request(url)
        
        if data:
            cache_work(doi, data)
        
        return data
    
    async def fetch_works_by_topic_cursor(self, topic_id: str, max_results: int = 2000, 
                                         progress_callback=None) -> List[dict]:
        all_works = []
        cursor = "*"
        page_count = 0
        
        cache_key = f"{topic_id}_cursor_{cursor}"
        cached = get_cached_topic_works(topic_id, cache_key)
        
        if cached and len(cached) >= max_results:
            logger.info(f"Using cached data for topic {topic_id}")
            return cached[:max_results]
        
        logger.info(f"Fetching works for topic {topic_id} (max: {max_results})")
        
        try:
            while len(all_works) < max_results and cursor:
                page_count += 1
                
                url = (f"{OPENALEX_BASE_URL}/works?"
                      f"filter=topics.id:{topic_id}&"
                      f"per-page={CURSOR_PAGE_SIZE}&"
                      f"cursor={cursor}&"
                      f"sort=publication_date:desc")
                
                data = await self.make_request(url)
                
                if not data or 'results' not in data:
                    break
                
                works = data['results']
                if not works:
                    break
                
                all_works.extend(works)
                
                # Call progress callback
                if progress_callback:
                    progress = min(len(all_works) / max_results, 1.0)
                    progress_callback(progress, len(all_works), page_count)
                
                meta = data.get('meta', {})
                cursor = meta.get('next_cursor')
                
                logger.info(f"Page {page_count}: got {len(works)} works, total: {len(all_works)}")
                
                cache_key = f"{topic_id}_cursor_{cursor or 'end'}"
                cache_topic_works(topic_id, cache_key, all_works)
                
                if not cursor or page_count >= 10:
                    break
                
                await asyncio.sleep(0.5)
            
            result = all_works[:max_results]
            cache_topic_works(topic_id, "final", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching topic works: {str(e)}")
            return all_works
    
    async def fetch_topic_stats(self, topic_id: str) -> Optional[dict]:
        cached = get_cached_topic_stats(topic_id)
        if cached:
            return cached
        
        url = f"{OPENALEX_BASE_URL}/topics/{topic_id}"
        data = await self.make_request(url)
        
        if data:
            cache_topic_stats(topic_id, data)
        
        return data

# ============================================================================
# СИНХРОННЫЕ ОБЕРТКИ
# ============================================================================

def run_async(coro):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def fetch_works_by_dois_sync(dois: List[str]) -> Tuple[List[dict], int, int]:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batches = [dois[i:i + BATCH_SIZE] for i in range(0, len(dois), BATCH_SIZE)]
    all_results = []
    successful = 0
    failed = 0
    
    async def process_batches():
        nonlocal all_results, successful, failed
        async with OpenAlexAsyncClient() as client:
            for i, batch in enumerate(batches):
                progress = (i + 1) / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Batch {i+1}/{len(batches)}: {len(batch)} DOI")
                
                results = await client.fetch_works_by_dois_batch(batch)
                
                for result in results:
                    if result:
                        successful += 1
                        all_results.append({
                            'data': result,
                            'success': True
                        })
                    else:
                        failed += 1
                        all_results.append({
                            'data': None,
                            'success': False
                        })
                
                if i < len(batches) - 1:
                    await asyncio.sleep(1)
    
    run_async(process_batches())
    
    progress_bar.empty()
    status_text.empty()
    
    return all_results, successful, failed

def fetch_works_by_topic_sync(topic_id: str, max_results: int = 2000) -> List[dict]:
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_works = []
    
    def update_progress(progress, count, page):
        progress_bar.progress(progress)
        status_text.text(f"Page {page}: {count}/{max_results} works fetched")
    
    async def fetch():
        async with OpenAlexAsyncClient() as client:
            return await client.fetch_works_by_topic_cursor(
                topic_id, max_results, update_progress
            )
    
    result = run_async(fetch())
    progress_bar.empty()
    status_text.empty()
    return result

def fetch_topic_stats_sync(topic_id: str) -> Optional[dict]:
    async def fetch():
        async with OpenAlexAsyncClient() as client:
            return await client.fetch_topic_stats(topic_id)
    
    return run_async(fetch())

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def normalize_word(word: str) -> str:
    word_lower = word.lower()
    
    if len(word_lower) < 4:
        return ''
    
    plural_exceptions = {
        'analyses': 'analysis', 'bases': 'base', 'criteria': 'criterion',
        'hypotheses': 'hypothesis', 'phenomena': 'phenomenon',
        'properties': 'property', 'activities': 'activity',
        'efficiencies': 'efficiency', 'performances': 'performance'
    }
    
    if word_lower in plural_exceptions:
        return plural_exceptions[word_lower]
    
    if word_lower.endswith('ies'):
        base = word_lower[:-3] + 'y'
        if len(base) >= 4:
            return base
    elif word_lower.endswith('es'):
        if word_lower.endswith(('ches', 'shes', 'xes', 'zes', 'sses')):
            base = word_lower[:-2]
            if len(base) >= 4:
                return base
    elif word_lower.endswith('s') and not word_lower.endswith(('ss', 'us', 'is', 'ys', 'as')):
        base = word_lower[:-1]
        if len(base) >= 4:
            return base
    
    return word_lower

def extract_keywords_from_title(title: str) -> List[str]:
    if not title:
        return []
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', title)
    filtered_words = []
    
    for word in words:
        word_lower = word.lower()
        
        if word_lower in ALL_STOPWORDS:
            continue
        
        if re.search(r'\d', word_lower):
            continue
        
        normalized = normalize_word(word_lower)
        if normalized:
            filtered_words.append(normalized)
    
    return filtered_words

def parse_chemical_elements(formula: str) -> Set[str]:
    """
    Парсит элементы из химической формулы, игнорируя коэффициенты.
    
    Примеры:
        "La2NiO4" → {'La', 'Ni', 'O'}
        "La1.8Pr0.2Ni0.7Co0.3O4" → {'La', 'Pr', 'Ni', 'Co', 'O'}
        "TiO2" → {'Ti', 'O'}
        "C60" → {'C'}
    
    Args:
        formula: Химическая формула
    
    Returns:
        Множество химических элементов в формуле
    """
    if not formula or len(formula) < 2:
        return set()
    
    elements = set()
    i = 0
    n = len(formula)
    
    while i < n:
        # Начинается с заглавной буквы
        if formula[i].isupper():
            elem = formula[i]
            i += 1
            
            # Проверяем строчную букву (двухбуквенные символы)
            if i < n and formula[i].islower():
                elem += formula[i]
                i += 1
            
            # Проверяем что элемент существует
            if elem in CHEM_ELEMENTS:
                elements.add(elem)
            elif len(elem) == 2 and elem[0] in CHEM_ELEMENTS:
                # Проверяем однобуквенные варианты
                elements.add(elem[0])
        else:
            i += 1  # Пропускаем цифры, точки, скобки и т.д.
    
    return elements

def extract_chemical_formulas_from_text(text: str) -> List[str]:
    """
    Извлекает химические формулы из текста.
    
    Args:
        text: Текст для анализа
    
    Returns:
        Список найденных химических формул
    """
    if not text:
        return []
    
    # Ищем формулы по шаблону
    formulas = re.findall(CHEM_FORMULA_PATTERN, text)
    
    # Фильтруем: минимум 2 символа, содержит хотя бы одну заглавную букву и цифру
    valid_formulas = []
    for formula in formulas:
        # Проверяем что это похоже на химическую формулу
        if len(formula) >= 2 and any(c.isupper() for c in formula):
            # Проверяем что содержит хотя бы один химический элемент
            elements = parse_chemical_elements(formula)
            if len(elements) >= 1:
                valid_formulas.append(formula)
    
    return valid_formulas

def extract_organic_patterns_from_text(text: str) -> Dict[str, List[str]]:
    """
    Извлекает органические паттерны из текста.
    
    Args:
        text: Текст для анализа
    
    Returns:
        Словарь с типами паттернов и найденными значениями
    """
    if not text:
        return {}
    
    patterns_found = {}
    
    for pattern_type, regex_pattern in ORGANIC_PATTERNS.items():
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if matches:
            # Убираем дубликаты
            unique_matches = list(set(matches))
            patterns_found[pattern_type] = unique_matches
    
    return patterns_found

def calculate_chemical_similarity(formula1: str, formula2: str) -> float:
    """
    Вычисляет химическую схожесть между двумя формулами.
    
    Args:
        formula1: Первая формула
        formula2: Вторая формула
    
    Returns:
        Коэффициент схожести от 0.0 до 1.0
    """
    elements1 = parse_chemical_elements(formula1)
    elements2 = parse_chemical_elements(formula2)
    
    if not elements1 or not elements2:
        return 0.0
    
    # Вычисляем коэффициент Жаккара
    intersection = elements1.intersection(elements2)
    union = elements1.union(elements2)
    
    if not union:
        return 0.0
    
    similarity = len(intersection) / len(union)
    
    # Бонус за полное совпадение
    if elements1 == elements2:
        similarity = min(1.0, similarity + 0.3)
    
    return similarity

def extract_numeric_from_doi(doi: str) -> int:
    """
    Extract numeric suffix from DOI for comparison.
    
    Examples:
        "10.5281/zenodo.17747567" -> 17747567
        "10.1002/anie.202000001" -> 202000001
        "10.1038/nature12345" -> 12345
    
    Args:
        doi: DOI string
    
    Returns:
        Integer value of the numeric suffix, or 0 if no number found
    """
    if not doi:
        return 0
    
    # Try to find the last numeric sequence in the DOI
    # First, split by common separators
    parts = doi.replace('.', '/').replace('-', '/').split('/')
    
    # Look for numeric parts from the end
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    
    # If no pure numeric parts, try to extract numbers from mixed strings
    numbers = re.findall(r'\d+', doi)
    if numbers:
        # Use the last number found (often version or ID)
        return int(numbers[-1])
    
    return 0

def parse_doi_input(text: str) -> List[str]:
    """
    Extract DOI identifiers from text handling various formats:
    1. https://doi.org/10.1002/fuce.70042
    2. 10.1002/fuce.70042
    3. https://dx.doi.org/10.1002/fuce.70042
    4. doi:10.1002/fuce.70042
    5. DOI:10.1002/fuce.70042
    6. Full citations with doi:10.1002/cphc.201000936
    """
    if not text or not text.strip():
        return []
    
    # More comprehensive DOI pattern that handles various formats
    # Matches DOI after common prefixes and URLs
    doi_patterns = [
        r'(?i)https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)',  # URL format
        r'(?i)(?:doi|DOI)[:\s]+(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)',            # doi: prefix format
        r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b'                               # Raw DOI format
    ]
    
    all_dois = []
    
    for pattern in doi_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):  # Some patterns may return groups
                doi = match[0] if match else ''
            else:
                doi = match
            
            if doi:
                # Clean up the DOI - remove any trailing punctuation
                doi = doi.strip()
                # Remove trailing punctuation (.,;:)
                doi = re.sub(r'[.,;:]+$', '', doi)
                # Remove any angle brackets or parentheses
                doi = doi.strip('<>()[]{}')
                all_dois.append(doi)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dois = []
    for doi in all_dois:
        # Normalize DOI to lowercase for comparison
        doi_lower = doi.lower()
        if doi_lower not in seen:
            seen.add(doi_lower)
            unique_dois.append(doi)
    
    return unique_dois[:300]

def analyze_keywords_parallel(titles: List[str]) -> Counter:
    all_keywords = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_keywords_from_title, title) for title in titles]
        for future in as_completed(futures):
            all_keywords.extend(future.result())
    
    return Counter(all_keywords)

def enrich_work_data(work: dict) -> dict:
    if not work:
        return {}
    
    doi_raw = work.get('doi')
    doi_clean = ''
    if doi_raw:
        doi_clean = str(doi_raw).replace('https://doi.org/', '')
    
    enriched = {
        'id': work.get('id', ''),
        'doi': doi_clean,
        'title': work.get('title', ''),
        'publication_date': work.get('publication_date', ''),
        'publication_year': work.get('publication_year', 0),
        'cited_by_count': work.get('cited_by_count', 0),
        'type': work.get('type', ''),
        'abstract': (work.get('abstract') or '')[:500],
        'doi_url': f"https://doi.org/{doi_clean}" if doi_clean else '',
    }
    
    authorships = work.get('authorships', [])
    authors = []
    institutions = set()
    
    for authorship in authorships:
        if authorship:
            author = authorship.get('author', {})
            if author:
                author_name = author.get('display_name', '')
                if author_name:
                    authors.append(author_name)
            
            for inst in authorship.get('institutions', []):
                if inst:
                    inst_name = inst.get('display_name', '')
                    if inst_name:
                        institutions.add(inst_name)
    
    enriched['authors'] = authors[:5]
    enriched['institutions'] = list(institutions)
    
    primary_location = work.get('primary_location')
    if primary_location:
        source = primary_location.get('source', {})
        enriched['journal_name'] = source.get('display_name', '') if source else ''
        enriched['journal_type'] = (source or {}).get('type', '')
    else:
        enriched['journal_name'] = ''
        enriched['journal_type'] = ''
    
    open_access = work.get('open_access')
    enriched['is_oa'] = open_access.get('is_oa', False) if open_access else False
    
    topics = work.get('topics', [])
    if topics:
        sorted_topics = sorted(topics, key=lambda x: x.get('score', 0) if x else 0, reverse=True)
        primary_topic = sorted_topics[0] if sorted_topics else {}
        enriched['primary_topic'] = primary_topic.get('display_name', '') if primary_topic else ''
        topic_id = primary_topic.get('id', '') if primary_topic else ''
        enriched['topic_id'] = topic_id.split('/')[-1] if topic_id else ''
    else:
        enriched['primary_topic'] = ''
        enriched['topic_id'] = ''
    
    return enriched

# ============================================================================
# НОВЫЙ КЛАСС ДЛЯ УЛУЧШЕННОГО АНАЛИЗА КЛЮЧЕВЫХ СЛОВ
# ============================================================================

class TitleKeywordsAnalyzer:
    def __init__(self):
        # Initialize stopwords and lemmatizer
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Load necessary NLTK resources
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-eng', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            except:
                pass
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Правила для специальных случаев
            self.irregular_plurals = {
                'analyses': 'analysis', 'axes': 'axis', 'bases': 'basis',
                'crises': 'crisis', 'criteria': 'criterion', 'data': 'datum',
                'diagnoses': 'diagnosis', 'ellipses': 'ellipsis', 'emphases': 'emphasis',
                'genera': 'genus', 'hypotheses': 'hypothesis', 'indices': 'index',
                'media': 'medium', 'memoranda': 'memorandum', 'parentheses': 'parenthesis',
                'phenomena': 'phenomenon', 'prognoses': 'prognosis', 'radii': 'radius',
                'stimuli': 'stimulus', 'syntheses': 'synthesis', 'theses': 'thesis',
                'vertebrae': 'vertebra', 'oxides': 'oxide', 'composites': 'composite',
                'applications': 'application', 'materials': 'material', 'methods': 'method',
                'systems': 'system', 'techniques': 'technique', 'properties': 'property',
                'structures': 'structure', 'devices': 'device', 'processes': 'process',
                'mechanisms': 'mechanism', 'models': 'model', 'approaches': 'approach',
                'frameworks': 'framework', 'strategies': 'strategy', 'solutions': 'solution',
                'technologies': 'technology', 'materials': 'material', 'nanoparticles': 'nanoparticle',
                'nanostructures': 'nanostructure', 'polymers': 'polymer', 'composites': 'composite',
                'ceramics': 'ceramic', 'alloys': 'alloy', 'coatings': 'coating', 'films': 'film',
                'layers': 'layer', 'interfaces': 'interface', 'surfaces': 'surface',
                'catalysts': 'catalyst', 'sensors': 'sensor', 'actuators': 'actuator',
                'transistors': 'transistor', 'diodes': 'diode', 'circuits': 'circuit',
                'networks': 'network', 'algorithms': 'algorithm', 'protocols': 'protocol',
                'databases': 'database', 'architectures': 'architecture', 'platforms': 'platform',
                'environments': 'environment', 'simulations': 'simulation', 'experiments': 'experiment',
                'measurements': 'measurement', 'observations': 'observation', 'analyses': 'analysis',
                'evaluations': 'evaluation', 'assessments': 'assessment', 'comparisons': 'comparison',
                'classifications': 'classification', 'predictions': 'prediction', 'optimizations': 'optimization',
                'characterizations': 'characterization', 'syntheses': 'synthesis', 'fabrications': 'fabrication',
                'preparations': 'preparation', 'treatments': 'treatment', 'modifications': 'modification',
                'enhancements': 'enhancement', 'improvements': 'improvement', 'developments': 'development',
                'innovations': 'innovation', 'discoveries': 'discovery', 'inventions': 'invention',
                'applications': 'application', 'implementations': 'implementation', 'utilizations': 'utilization',
                'integrations': 'integration', 'combinations': 'combination', 'interactions': 'interaction',
                'relationships': 'relationship', 'dependencies': 'dependency', 'correlations': 'correlation',
                'associations': 'association', 'connections': 'connection', 'communications': 'communication',
                'collaborations': 'collaboration', 'cooperations': 'cooperation', 'competitions': 'competition',
                'conflicts': 'conflict', 'challenges': 'challenge', 'problems': 'problem', 'solutions': 'solution',
                'alternatives': 'alternative', 'options': 'option', 'variants': 'variant', 'versions': 'version',
                'editions': 'edition', 'releases': 'release', 'updates': 'update', 'revisions': 'revision',
                'modifications': 'modification', 'adaptations': 'adaptation', 'customizations': 'customization',
                'personalizations': 'personalization', 'localizations': 'localization', 'internationalizations': 'internationalization',
                'standardizations': 'standardization', 'normalizations': 'normalization', 'optimizations': 'optimization',
                'maximizations': 'maximization', 'minimizations': 'minimization', 'reductions': 'reduction',
                'increases': 'increase', 'improvements': 'improvement', 'enhancements': 'enhancement',
                'advancements': 'advancement', 'progresses': 'progress', 'developments': 'development',
                'evolutions': 'evolution', 'revolutions': 'revolution', 'transformations': 'transformation',
                'changes': 'change', 'variations': 'variation', 'fluctuations': 'fluctuation', 'oscillations': 'oscillation',
                'vibrations': 'vibration', 'rotations': 'rotation', 'translations': 'translation', 'movements': 'movement',
                'motions': 'motion', 'dynamics': 'dynamic', 'kinematics': 'kinematic', 'mechanics': 'mechanic',
                'thermodynamics': 'thermodynamic', 'electrodynamics': 'electrodynamic', 'hydrodynamics': 'hydrodynamic',
                'aerodynamics': 'aerodynamic', 'biomechanics': 'biomechanic', 'geomechanics': 'geomechanic',
                'chemomechanics': 'chemomechanic', 'tribology': 'tribology', 'rheology': 'rheology',
                'viscoelasticity': 'viscoelastic', 'plasticity': 'plastic', 'elasticity': 'elastic',
                'viscosity': 'viscous', 'conductivity': 'conductive', 'resistivity': 'resistive',
                'permeability': 'permeable', 'porosity': 'porous', 'density': 'dense', 'hardness': 'hard',
                'stiffness': 'stiff', 'strength': 'strong', 'toughness': 'tough', 'brittleness': 'brittle',
                'ductility': 'ductile', 'malleability': 'malleable', 'flexibility': 'flexible', 'rigidity': 'rigid',
                'stability': 'stable', 'instability': 'unstable', 'reliability': 'reliable', 'durability': 'durable',
                'sustainability': 'sustainable', 'efficiency': 'efficient', 'effectiveness': 'effective',
                'performance': 'perform', 'productivity': 'productive', 'quality': 'qualitative',
                'quantity': 'quantitative', 'accuracy': 'accurate', 'precision': 'precise', 'reliability': 'reliable',
                'validity': 'valid', 'reproducibility': 'reproducible', 'repeatability': 'repeatable',
                'consistency': 'consistent', 'homogeneity': 'homogeneous', 'heterogeneity': 'heterogeneous',
                'isotropy': 'isotropic', 'anisotropy': 'anisotropic', 'symmetry': 'symmetric',
                'asymmetry': 'asymmetric', 'regularity': 'regular', 'irregularity': 'irregular',
                'periodicity': 'periodic', 'aperiodicity': 'aperiodic', 'randomness': 'random',
                'determinism': 'deterministic', 'stochasticity': 'stochastic', 'probability': 'probable',
                'statistics': 'statistic', 'distributions': 'distribution', 'functions': 'function',
                'equations': 'equation', 'formulas': 'formula', 'theorems': 'theorem', 'lemmas': 'lemma',
                'corollaries': 'corollary', 'proofs': 'proof', 'demonstrations': 'demonstration',
                'verifications': 'verification', 'validations': 'validation', 'confirmations': 'confirmation',
                'tests': 'test', 'experiments': 'experiment', 'trials': 'trial', 'studies': 'study',
                'investigations': 'investigation', 'examinations': 'examination', 'inspections': 'inspection',
                'audits': 'audit', 'reviews': 'review', 'surveys': 'survey', 'polls': 'poll',
                'questionnaires': 'questionnaire', 'interviews': 'interview', 'observations': 'observation',
                'measurements': 'measurement', 'calculations': 'calculation', 'computations': 'computation',
                'simulations': 'simulation', 'modelings': 'modeling', 'analyses': 'analysis', 'syntheses': 'synthesis',
                'evaluations': 'evaluation', 'assessments': 'assessment', 'appraisals': 'appraisal',
                'estimations': 'estimation', 'approximations': 'approximation', 'predictions': 'prediction',
                'forecasts': 'forecast', 'projections': 'projection', 'extrapolations': 'extrapolation',
                'interpolations': 'interpolation', 'regressions': 'regression', 'correlations': 'correlation',
                'classifications': 'classification', 'clusters': 'cluster', 'segments': 'segment', 'groups': 'group',
                'categories': 'category', 'types': 'type', 'classes': 'class', 'kinds': 'kind', 'sorts': 'sort',
                'varieties': 'variety', 'forms': 'form', 'shapes': 'shape', 'sizes': 'size', 'dimensions': 'dimension',
                'volumes': 'volume', 'areas': 'area', 'lengths': 'length', 'widths': 'width', 'heights': 'height',
                'depths': 'depth', 'thicknesses': 'thickness', 'diameters': 'diameter', 'radii': 'radius',
                'circumferences': 'circumference', 'perimeters': 'perimeter', 'surfaces': 'surface',
                'interfaces': 'interface', 'boundaries': 'boundary', 'edges': 'edge', 'corners': 'corner',
                'vertices': 'vertex', 'nodes': 'node', 'points': 'point', 'lines': 'line', 'curves': 'curve',
                'planes': 'plane', 'spaces': 'space', 'regions': 'region', 'zones': 'zone', 'sectors': 'sector',
                'segments': 'segment', 'parts': 'part', 'components': 'component', 'elements': 'element',
                'units': 'unit', 'modules': 'module', 'blocks': 'block', 'pieces': 'piece', 'fragments': 'fragment',
                'particles': 'particle', 'atoms': 'atom', 'molecules': 'molecule', 'ions': 'ion', 'electrons': 'electron',
                'protons': 'proton', 'neutrons': 'neutron', 'photons': 'photon', 'quarks': 'quark', 'leptons': 'lepton',
                'bosons': 'boson', 'fermions': 'fermion', 'hadrons': 'hadron', 'mesons': 'meson', 'baryons': 'baryon',
                'nuclei': 'nucleus', 'isotopes': 'isotope', 'elements': 'element', 'compounds': 'compound',
                'mixtures': 'mixture', 'solutions': 'solution', 'suspensions': 'suspension', 'colloids': 'colloid',
                'emulsions': 'emulsion', 'foams': 'foam', 'gels': 'gel', 'solids': 'solid', 'liquids': 'liquid',
                'gases': 'gas', 'plasmas': 'plasma', 'crystals': 'crystal', 'amorphous': 'amorphous', 'polymers': 'polymer',
                'monomers': 'monomer', 'oligomers': 'oligomer', 'copolymers': 'copolymer', 'homopolymers': 'homopolymer',
                'biopolymers': 'biopolymer', 'proteins': 'protein', 'enzymes': 'enzyme', 'antibodies': 'antibody',
                'antigens': 'antigen', 'vaccines': 'vaccine', 'drugs': 'drug', 'medicines': 'medicine',
                'therapies': 'therapy', 'treatments': 'treatment', 'diagnoses': 'diagnosis', 'prognoses': 'prognosis',
                'symptoms': 'symptom', 'diseases': 'disease', 'disorders': 'disorder', 'conditions': 'condition',
                'syndromes': 'syndrome', 'infections': 'infection', 'inflammations': 'inflammation', 'tumors': 'tumor',
                'cancers': 'cancer', 'metastases': 'metastasis', 'remissions': 'remission', 'recurrences': 'recurrence',
                'survivals': 'survival', 'mortality': 'mortal', 'morbidity': 'morbid', 'epidemiology': 'epidemiologic',
                'pathology': 'pathologic', 'physiology': 'physiologic', 'anatomy': 'anatomic', 'histology': 'histologic',
                'cytology': 'cytologic', 'genetics': 'genetic', 'genomics': 'genomic', 'proteomics': 'proteomic',
                'metabolomics': 'metabolomic', 'transcriptomics': 'transcriptomic', 'epigenetics': 'epigenetic',
                'bioinformatics': 'bioinformatic', 'biotechnology': 'biotechnologic', 'nanotechnology': 'nanotechnologic',
                'microtechnology': 'microtechnologic', 'microfabrication': 'microfabricate', 'nanofabrication': 'nanofabricate',
                'lithography': 'lithographic', 'photolithography': 'photolithographic', 'electron-beam': 'electron-beam',
                'ion-beam': 'ion-beam', 'focused-ion-beam': 'focused-ion-beam', 'atomic-force': 'atomic-force',
                'scanning-tunneling': 'scanning-tunneling', 'transmission-electron': 'transmission-electron',
                'scanning-electron': 'scanning-electron', 'optical': 'optical', 'confocal': 'confocal',
                'fluorescence': 'fluorescent', 'phosphorescence': 'phosphorescent', 'luminescence': 'luminescent',
                'chemiluminescence': 'chemiluminescent', 'bioluminescence': 'bioluminescent', 'electroluminescence': 'electroluminescent',
                'photoluminescence': 'photoluminescent', 'cathodoluminescence': 'cathodoluminescent',
                'thermoluminescence': 'thermoluminescent', 'radioluminescence': 'radioluminescent',
                'sonoluminescence': 'sonoluminescent', 'triboluminescence': 'triboluminescent',
                'crystalloluminescence': 'crystalloluminescent', 'electroluminescence': 'electroluminescent',
                'magnetoluminescence': 'magnetoluminescent',
            }
            
            # Суффиксы, которые нужно преобразовать
            self.suffix_replacements = {
                'ies': 'y',
                'es': '',
                's': '',
                'ed': '',
                'ing': '',
                'ly': '',
                'ally': 'al',
                'ically': 'ic',
                'ization': 'ize',
                'isation': 'ise',
                'ment': '',
                'ness': '',
                'ity': '',
                'ty': '',
                'ic': '',
                'ical': '',
                'ive': '',
                'ous': '',
                'ful': '',
                'less': '',
                'est': '',
                'er': '',
                'ors': 'or',
                'ors': 'or',
                'ings': 'ing',
                'ments': 'ment',
            }
            
        except:
            # Fallback if nltk not available
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            self.lemmatizer = None
            self.irregular_plurals = {}
            self.suffix_replacements = {}
        
        # Scientific stopwords (уже лемматизированные)
        self.scientific_stopwords = {
            'activate', 'adapt', 'advance', 'analyze', 'apply',
            'approach', 'architect', 'artificial', 'assess',
            'base', 'behave', 'capacity', 'characterize',
            'coat', 'compare', 'compute', 'composite',
            'control', 'cycle', 'damage', 'data', 'density', 'design',
            'detect', 'develop', 'device', 'diagnose', 'discover',
            'dynamic', 'economic', 'effect', 'efficacy',
            'efficient', 'energy', 'engineer', 'enhance', 'environment',
            'evaluate', 'experiment', 'explore', 'factor', 'fail',
            'fabricate', 'field', 'film', 'flow', 'framework', 'frequency',
            'function', 'grow', 'high', 'impact', 'improve',
            'induce', 'influence', 'inform', 'innovate', 'intelligent',
            'interact', 'interface', 'investigate', 'know',
            'layer', 'learn', 'magnetic', 'manage', 'material',
            'measure', 'mechanism', 'medical',
            'method', 'model', 'modify', 'modulate',
            'molecule', 'monitor', 'motion', 'nanoparticle',
            'nanostructure', 'network', 'neural', 'new', 'nonlinear',
            'novel', 'numerical', 'optical', 'optimize', 'pattern', 'perform',
            'phenomenon', 'potential', 'power', 'predict', 'prepare', 'process',
            'produce', 'progress', 'property', 'quality', 'regulate', 'relate',
            'reliable', 'remote', 'repair', 'research', 'resist', 'respond',
            'review', 'risk', 'role', 'safe', 'sample', 'scale', 'screen',
            'separate', 'signal', 'simulate', 'specific', 'stable', 'state',
            'store', 'strain', 'strength', 'stress', 'structure', 'study',
            'sustain', 'synergy', 'synthesize', 'system', 'target',
            'technique', 'technology', 'test', 'theoretical', 'therapy',
            'thermal', 'tissue', 'tolerate', 'toxic', 'transform', 'transition',
            'transmit', 'transport', 'type', 'understand', 'use', 'validate',
            'value', 'vary', 'virtual', 'waste', 'wave',
            'application', 'approach', 'assessment', 'behavior', 'capability',
            'characterization', 'comparison', 'concept', 'condition', 'configuration',
            'construction', 'contribution', 'demonstration', 'description', 'detection',
            'determination', 'development', 'effectiveness', 'efficiency', 'evaluation',
            'examination', 'experimentation', 'explanation', 'exploration', 'fabrication',
            'formation', 'implementation', 'improvement', 'indication', 'investigation',
            'management', 'manufacture', 'measurement', 'modification', 'observation',
            'operation', 'optimization', 'performance', 'preparation', 'presentation',
            'production', 'realization', 'recognition', 'regulation', 'representation',
            'simulation', 'solution', 'specification', 'synthesis', 'transformation',
            'treatment', 'utilization', 'validation', 'verification'
        }

        # Химические константы
        self.chemical_elements = CHEM_ELEMENTS
        self.chemical_weight = CHEMICAL_WEIGHT
        self.organic_weight = ORGANIC_PATTERN_WEIGHT
        self.chem_formula_bonus = CHEM_FORMULA_BONUS
        
    def _get_lemma(self, word: str) -> str:
        """Get word lemma considering special rules"""
        if not word or len(word) < 3:
            return word
        
        # Convert to lowercase for processing
        lower_word = word.lower()
        
        # Check irregular plurals FIRST
        if lower_word in self.irregular_plurals:
            return self.irregular_plurals[lower_word]
        
        # Check regular plurals
        # If word ends with 's' or 'es' but not 'ss' or 'us'
        if lower_word.endswith('s') and not (lower_word.endswith('ss') or lower_word.endswith('us')):
            # Try to remove 's' or 'es'
            if lower_word.endswith('es') and len(lower_word) > 2:
                base_word = lower_word[:-2]
                # Check that after removing 'es' word not too short
                if len(base_word) >= 3:
                    return base_word
            elif len(lower_word) > 1:
                base_word = lower_word[:-1]
                # Check that after removing 's' word not too short
                if len(base_word) >= 3:
                    return base_word
        
        # Use lemmatizer if available
        if self.lemmatizer:
            # Try different parts of speech
            for pos in ['n', 'v', 'a', 'r']:  # noun, verb, adjective, adverb
                lemma = self.lemmatizer.lemmatize(lower_word, pos=pos)
                if lemma != lower_word:
                    return lemma
        
        # Apply suffix rules in reverse order (long to short)
        sorted_suffixes = sorted(self.suffix_replacements.keys(), key=len, reverse=True)
        for suffix in sorted_suffixes:
            if lower_word.endswith(suffix) and len(lower_word) > len(suffix) + 2:
                replacement = self.suffix_replacements[suffix]
                base = lower_word[:-len(suffix)] + replacement
                # Check result not too short
                if len(base) >= 3:
                    # Also check base doesn't end with double consonant
                    if len(base) >= 4 and base[-1] == base[-2]:
                        base = base[:-1]
                    return base
        
        return lower_word
    
    def _get_base_form(self, word: str) -> str:
        """Get base word form with aggressive lemmatization"""
        lemma = self._get_lemma(word)
        
        # Additional rules for scientific terms
        if lemma.endswith('isation'):
            return lemma[:-7] + 'ize'
        elif lemma.endswith('ization'):
            return lemma[:-7] + 'ize'
        elif lemma.endswith('ication'):
            return lemma[:-7] + 'y'
        elif lemma.endswith('ation'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ition'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ution'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ment'):
            return lemma[:-4]
        elif lemma.endswith('ness'):
            return lemma[:-4]
        elif lemma.endswith('ity'):
            return lemma[:-3] + 'e'
        elif lemma.endswith('ty'):
            base = lemma[:-2]
            if base.endswith('i'):
                return base[:-1] + 'y'
            return base
        elif lemma.endswith('ic'):
            return lemma[:-2] + 'y'
        elif lemma.endswith('al'):
            return lemma[:-2]
        elif lemma.endswith('ive'):
            return lemma[:-3] + 'e'
        elif lemma.endswith('ous'):
            return lemma[:-3]
        
        return lemma
    
    def preprocess_content_words(self, text: str) -> List[Dict]:
        """Clean and normalize content words, return dictionaries with lemmas and forms"""
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error']:
            return []

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        content_words = []

        for word in words:
            # EXCLUDE word "sub"
            if word == 'sub':
                continue
            if '-' in word:
                continue
            if len(word) > 2 and word not in self.stop_words:
                lemma = self._get_base_form(word)
                if lemma not in self.scientific_stopwords:
                    content_words.append({
                        'original': word,
                        'lemma': lemma,
                        'type': 'content'
                    })

        return content_words

    def extract_compound_words(self, text: str) -> List[Dict]:
        """Extract hyphenated compound words"""
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error']:
            return []

        text = text.lower()
        compound_words = re.findall(r'\b[a-z]{2,}-[a-z]{2,}(?:-[a-z]{2,})*\b', text)

        compounds = []
        for word in compound_words:
            parts = word.split('-')
            if not any(part in self.stop_words for part in parts):
                # For compound words lemmatize each part
                lemmatized_parts = []
                for part in parts:
                    lemma = self._get_base_form(part)
                    lemmatized_parts.append(lemma)
                
                compounds.append({
                    'original': word,
                    'lemma': '-'.join(lemmatized_parts),
                    'type': 'compound'
                })

        return compounds

    def extract_chemical_formulas(self, text: str) -> List[Dict]:
        """
        Извлекает химические формулы как специальные compound words.
        
        Args:
            text: Текст для анализа
        
        Returns:
            Список словарей с информацией о формулах
        """
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error']:
            return []
        
        formulas = extract_chemical_formulas_from_text(text)
        chem_formulas = []
        
        for formula in formulas:
            elements = parse_chemical_elements(formula)
            if elements:  # Хотя бы один элемент
                # Создаем лемму для сравнения (сортированные элементы через дефис)
                sorted_elements = sorted(elements)
                lemma = '-'.join(sorted_elements)
                
                chem_formulas.append({
                    'original': formula,
                    'lemma': lemma,
                    'type': 'chemical',
                    'elements': elements,
                    'element_count': len(elements),
                    'weight': self.chemical_weight * (1 + 0.1 * len(elements))  # Вес зависит от сложности
                })
        
        return chem_formulas
    
    def extract_organic_patterns(self, text: str) -> List[Dict]:
        """
        Извлекает IUPAC-подобные органические паттерны для повышенного веса.
        
        Args:
            text: Текст для анализа
        
        Returns:
            Список словарей с информацией об органических паттернах
        """
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error']:
            return []
        
        patterns_found = []
        text_lower = text.lower()
        
        for pattern_type, regex in ORGANIC_PATTERNS.items():
            matches = re.findall(regex, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Берем первую группу если match - tuple
                
                # Нормализуем для сравнения
                normalized = re.sub(r'[\s()[\]]', '', match.lower())
                
                # Определяем вес в зависимости от типа паттерна
                weight_modifier = self.organic_weight
                if pattern_type == 'fusion_bracket':
                    weight_modifier *= 1.4  # Fusion паттерны важнее
                elif pattern_type == 'bis_tris':
                    weight_modifier *= 1.2
                elif pattern_type == 'heterocycle_endings':
                    weight_modifier *= 1.1
                
                patterns_found.append({
                    'original': match,
                    'lemma': normalized,
                    'type': 'organic_pattern',
                    'subtype': pattern_type,
                    'weight': weight_modifier
                })
        
        return patterns_found
    
    def extract_all_chemical_info(self, text: str) -> Dict:
        """
        Извлекает всю химическую информацию из текста.
        
        Args:
            text: Текст для анализа
        
        Returns:
            Словарь со всей химической информацией
        """
        return {
            'formulas': self.extract_chemical_formulas(text),
            'organic_patterns': self.extract_organic_patterns(text),
            'has_chemical_content': False  # Будет обновлено ниже
        }

    def _are_similar_lemmas(self, lemma1: str, lemma2: str) -> bool:
        """Check if lemmas are similar (e.g., singular/plural)"""
        if lemma1 == lemma2:
            return True
        
        # Check if they are forms of the same word
        # Example: "composite" and "composites"
        if lemma1.endswith('s') and lemma1[:-1] == lemma2:
            return True
        if lemma2.endswith('s') and lemma2[:-1] == lemma1:
            return True
        
        # Check if they are forms with different suffixes
        # Example: "characterization" and "characterize"
        common_prefix = self._get_common_prefix(lemma1, lemma2)
        if len(common_prefix) >= 5:  # If common prefix long enough
            # Check length difference
            if abs(len(lemma1) - len(lemma2)) <= 3:
                return True
        
        return False
    
    def _get_common_prefix(self, str1: str, str2: str) -> str:
        """Return common prefix of two strings"""
        min_length = min(len(str1), len(str2))
        common_prefix = []
        
        for i in range(min_length):
            if str1[i] == str2[i]:
                common_prefix.append(str1[i])
            else:
                break
        
        return ''.join(common_prefix)

class EnhancedKeywordAnalyzer:
    def __init__(self):
        self.title_analyzer = TitleKeywordsAnalyzer()
        
        # Веса для разных типов слов
        self.weights = {
            'content': 1.0,
            'compound': 1.5,
            'scientific': 0.7,
            'chemical': 2.3,
            'organic_pattern': 2.5
        }
    
    def extract_weighted_keywords(self, titles: List[str]) -> Dict[str, float]:
        """Извлечение ключевых слов с весами, включая химические паттерны"""
        weighted_counter = Counter()
        chemical_formulas_count = 0
        organic_patterns_count = 0
        
        for title in titles:
            if not title:
                continue
                
            # Извлекаем все типы слов
            content_words = self.title_analyzer.preprocess_content_words(title)
            compound_words = self.title_analyzer.extract_compound_words(title)
            chemical_formulas = self.title_analyzer.extract_chemical_formulas(title)
            organic_patterns = self.title_analyzer.extract_organic_patterns(title)
            
            # Учитываем веса
            for word_info in content_words:
                lemma = word_info['lemma']
                if lemma:
                    weighted_counter[lemma] += self.weights['content']
            
            for word_info in compound_words:
                lemma = word_info['lemma']
                if lemma:
                    weighted_counter[lemma] += self.weights['compound']
            
            for formula_info in chemical_formulas:
                lemma = formula_info['lemma']
                if lemma:
                    # Используем индивидуальный вес для формулы
                    weight = formula_info.get('weight', self.weights['chemical'])
                    weighted_counter[lemma] += weight
                    chemical_formulas_count += 1
            
            for pattern_info in organic_patterns:
                lemma = pattern_info['lemma']
                if lemma:
                    # Используем индивидуальный вес для паттерна
                    weight = pattern_info.get('weight', self.weights['organic_pattern'])
                    weighted_counter[lemma] += weight
                    organic_patterns_count += 1
        
        # Добавляем маркеры для химического контента
        if chemical_formulas_count > 0:
            weighted_counter['__CHEMICAL_FORMULAS__'] = chemical_formulas_count
        if organic_patterns_count > 0:
            weighted_counter['__ORGANIC_PATTERNS__'] = organic_patterns_count
        
        return weighted_counter
    
    def extract_chemical_keywords(self, keywords: List[str]) -> Dict[str, Dict]:
        """
        Извлекает химические ключевые слова из списка.
        
        Args:
            keywords: Список ключевых слов
        
        Returns:
            Словарь с химическими ключевыми словами по типам
        """
        chemical_keywords = {
            'formulas': [],
            'elements': [],
            'organic_patterns': [],
            'common_chemicals': []
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Проверяем на химическую формулу
            formulas = extract_chemical_formulas_from_text(keyword)
            if formulas:
                for formula in formulas:
                    chemical_keywords['formulas'].append({
                        'original': formula,
                        'elements': parse_chemical_elements(formula)
                    })
            
            # Проверяем на органические паттерны
            organic_patterns = extract_organic_patterns_from_text(keyword)
            for pattern_type, matches in organic_patterns.items():
                for match in matches:
                    chemical_keywords['organic_patterns'].append({
                        'original': match,
                        'type': pattern_type
                    })
            
            # Проверяем на химические элементы
            # Разделяем на слова и проверяем каждый
            words = re.findall(r'\b[a-zA-Z]+\b', keyword)
            for word in words:
                if word in CHEM_ELEMENTS or (len(word) == 2 and word[0].isupper() and word[1].islower() and word in CHEM_ELEMENTS):
                    chemical_keywords['elements'].append(word)
        
        return chemical_keywords

def calculate_enhanced_relevance(work: dict, keywords: Dict[str, float], 
                                 analyzer: TitleKeywordsAnalyzer) -> Tuple[float, List[str], Dict]:
    """
    Расчет релевантности с учетом семантической близости и химических паттернов.
    
    Возвращает:
        - Оценку релевантности (0-10)
        - Список совпавших ключевых слов
        - Словарь с дополнительной химической информацией
    """
    
    title = work.get('title', '').lower()
    abstract = work.get('abstract', '').lower()
    full_text = title + " " + abstract
    
    if not title:
        return 0.0, [], {}
    
    score = 0.0
    matched_keywords = []
    chemical_info = {
        'has_chemical_formula': False,
        'has_organic_pattern': False,
        'chemical_bonus': 0.0,
        'organic_bonus': 0.0,
        'formula_matches': [],
        'organic_matches': [],
        'element_matches': []
    }
    
    # ========== БАЗОВЫЙ АНАЛИЗ (существующий) ==========
    
    # Извлекаем слова из заголовка анализируемой работы
    title_words = analyzer.preprocess_content_words(title)
    compound_words = analyzer.extract_compound_words(title)
    
    title_lemmas = {w['lemma'] for w in title_words}
    compound_lemmas = {w['lemma'] for w in compound_words}
    all_title_lemmas = title_lemmas.union(compound_lemmas)
    
    # Проверяем каждое ключевое слово
    for keyword, weight in keywords.items():
        # Пропускаем системные маркеры
        if keyword.startswith('__') and keyword.endswith('__'):
            continue
            
        keyword_lower = keyword.lower()
        keyword_base = analyzer._get_base_form(keyword_lower)
        
        # Проверяем точное совпадение в заголовке
        if keyword_lower in title:
            score += weight * 3.0  # Высокий вес для точного совпадения
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)
        
        # Проверяем точное совпадение в аннотации
        elif abstract and keyword_lower in abstract:
            score += weight * 1.0  # Меньший вес для аннотации
            if f"{keyword}*" not in matched_keywords:
                matched_keywords.append(f"{keyword}*")
        
        else:
            # Проверяем лемматизированные формы в заголовке
            for lemma in all_title_lemmas:
                if analyzer._are_similar_lemmas(keyword_base, lemma):
                    score += weight * 2.0  # Средний вес для семантической близости
                    if f"{keyword}~{lemma}" not in matched_keywords:
                        matched_keywords.append(f"{keyword}~{lemma}")
                    break
    
    # ========== ХИМИЧЕСКИЙ АНАЛИЗ ==========
    
    # Извлекаем химические формулы из целевой работы
    target_formulas = analyzer.extract_chemical_formulas(full_text)
    target_organic = analyzer.extract_organic_patterns(full_text)
    
    # Обновляем химическую информацию
    if target_formulas:
        chemical_info['has_chemical_formula'] = True
        chemical_info['formula_matches'] = [f['original'] for f in target_formulas]
    
    if target_organic:
        chemical_info['has_organic_pattern'] = True
        chemical_info['organic_matches'] = [p['original'] for p in target_organic]
    
    # Извлекаем химические элементы из формул
    target_elements = set()
    for formula_info in target_formulas:
        target_elements.update(formula_info['elements'])
    
    # Ищем химические формулы в ключевых словах
    input_chemical_elements = set()
    input_chemical_formulas = []
    input_organic_patterns = []
    
    for keyword in keywords.keys():
        # Пропускаем системные маркеры
        if keyword.startswith('__') and keyword.endswith('__'):
            continue
            
        # Проверяем на химическую формулу
        formulas = extract_chemical_formulas_from_text(keyword)
        if formulas:
            input_chemical_formulas.extend(formulas)
            for formula in formulas:
                input_chemical_elements.update(parse_chemical_elements(formula))
        
        # Проверяем на органические паттерны
        organic_patterns = extract_organic_patterns_from_text(keyword)
        for pattern_type, matches in organic_patterns.items():
            for match in matches:
                input_organic_patterns.append({
                    'original': match,
                    'type': pattern_type
                })
        
        # Проверяем на химические элементы (отдельные слова)
        words = keyword.split()
        for word in words:
            if word in CHEM_ELEMENTS:
                input_chemical_elements.add(word)
    
    # ========== РАСЧЕТ ХИМИЧЕСКОГО БОНУСА ==========
    
    # Бонус за совпадение химических элементов
    if input_chemical_elements and target_elements:
        common_elements = input_chemical_elements.intersection(target_elements)
        if common_elements:
            element_match_ratio = len(common_elements) / len(input_chemical_elements) if input_chemical_elements else 0
            
            if element_match_ratio >= 0.7:
                chemical_bonus = 3.0 * element_match_ratio
                chemical_info['chemical_bonus'] = chemical_bonus
                chemical_info['element_matches'] = list(common_elements)
                score += chemical_bonus
                
                matched_keywords.append(f"Chemical elements: {', '.join(sorted(common_elements))}")
    
    # Бонус за совпадение химических формул
    if input_chemical_formulas and target_formulas:
        for input_formula in input_chemical_formulas:
            input_elements = parse_chemical_elements(input_formula)
            
            for target_formula_info in target_formulas:
                target_elements_formula = target_formula_info['elements']
                
                # Вычисляем схожесть формул
                similarity = calculate_chemical_similarity(
                    input_formula, 
                    target_formula_info['original']
                )
                
                if similarity >= 0.5:  # Порог схожести
                    formula_bonus = CHEM_FORMULA_BONUS * similarity
                    chemical_info['chemical_bonus'] += formula_bonus
                    score += formula_bonus
                    
                    matched_keywords.append(f"Formula match: {input_formula} ≈ {target_formula_info['original']}")
                    break
    
    # ========== РАСЧЕТ БОНУСА ЗА ОРГАНИЧЕСКИЕ ПАТТЕРНЫ ==========
    
    if input_organic_patterns and target_organic:
        for input_pattern in input_organic_patterns:
            input_original = input_pattern['original'].lower()
            input_type = input_pattern['type']
            
            for target_pattern in target_organic:
                target_original = target_pattern['original'].lower()
                target_type = target_pattern['type']
                
                # Нормализуем для сравнения (убираем пробелы и скобки)
                input_norm = re.sub(r'[\s()[\]]', '', input_original)
                target_norm = re.sub(r'[\s()[\]]', '', target_original)
                
                # Проверяем совпадение
                if input_norm == target_norm or input_norm in target_norm or target_norm in input_norm:
                    organic_bonus = 0.0
                    
                    if input_type == 'fusion_bracket':
                        organic_bonus = FUSION_MATCH_BONUS
                        matched_keywords.append(f"Fusion pattern: {input_original}")
                    elif input_type == 'bis_tris':
                        organic_bonus = BIS_MATCH_BONUS
                        matched_keywords.append(f"Bis/tris pattern: {input_original}")
                    elif input_type == 'heterocycle_endings':
                        organic_bonus = 2.0
                        matched_keywords.append(f"Heterocycle: {input_original}")
                    else:
                        organic_bonus = 1.5
                        matched_keywords.append(f"Organic pattern: {input_original}")
                    
                    chemical_info['organic_bonus'] += organic_bonus
                    score += organic_bonus
                    break
    
    # ========== ДОПОЛНИТЕЛЬНЫЕ БОНУСЫ ==========
    
    # Бонус за составные слова
    if compound_words:
        score += len(compound_words) * 0.5
    
    # Бонус за наличие химической формулы в заголовке
    if chemical_info['has_chemical_formula']:
        score += 1.0
        matched_keywords.append("Contains chemical formula")
    
    # Бонус за наличие органических паттернов
    if chemical_info['has_organic_pattern']:
        score += 0.5
        matched_keywords.append("Contains organic patterns")
    
    # ========== НОРМАЛИЗАЦИЯ И ОГРАНИЧЕНИЕ ==========
    
    # Ограничиваем максимальный балл
    max_score = 10.0
    normalized_score = min(score, max_score)
    
    # Округляем до одного знака после запятой
    normalized_score = round(normalized_score, 1)
    
    # Обновляем общий бонус в химической информации
    total_chemical_bonus = chemical_info.get('chemical_bonus', 0) + chemical_info.get('organic_bonus', 0)
    chemical_info['total_chemical_bonus'] = total_chemical_bonus
    
    return normalized_score, matched_keywords, chemical_info

def passes_filters(work: dict, year_filter: List[int], 
                   citation_ranges: List[Tuple[int, int]]) -> bool:
    """Проверяет работу на соответствие фильтрам"""
    
    cited_by_count = work.get('cited_by_count', 0)
    publication_year = work.get('publication_year', 0)
    
    # Фильтр по годам
    if year_filter and publication_year not in year_filter:
        return False
    
    # Фильтр по цитированиям
    if citation_ranges:
        in_range = False
        for min_cit, max_cit in citation_ranges:
            if min_cit <= cited_by_count <= max_cit:
                in_range = True
                break
        if not in_range:
            return False
    
    return True

def analyze_works_for_topic(
    topic_id: str,
    keywords: List[str],
    max_citations: int = 10,  # ← Этот параметр теперь не используется!
    max_works: int = 2000,
    top_n: int = 100,
    year_filter: List[int] = None,
    citation_ranges: List[Tuple[int, int]] = None
) -> List[dict]:
    """
    Analyze works for a specific topic with filtering of input DOIs and duplicate titles.
    """
    
    with st.spinner(f"Loading up to {max_works} works..."):
        works = fetch_works_by_topic_sync(topic_id, max_works)
    
    if not works:
        return []
    
    current_year = datetime.now().year
    if year_filter is None:
        year_filter = [current_year - 2, current_year - 1, current_year]
    
    if citation_ranges is None:
        citation_ranges = [(0, 10)]
    
    # Get input DOIs from session state to exclude them from recommendations
    input_dois = set()
    if 'dois' in st.session_state:
        # Normalize input DOIs (remove https://doi.org/ prefix for comparison)
        for doi in st.session_state.dois:
            if doi.startswith('https://doi.org/'):
                clean_doi = doi.replace('https://doi.org/', '').lower()
            else:
                clean_doi = doi.lower()
            input_dois.add(clean_doi)
        logger.info(f"Excluding {len(input_dois)} input DOIs from recommendations")
    
    # Инициализация анализаторов
    title_analyzer = TitleKeywordsAnalyzer()
    keyword_analyzer = EnhancedKeywordAnalyzer()
    
    # Преобразуем ключевые слова в взвешенный словарь
    keywords_lower = [kw.lower() for kw in keywords]
    weighted_keywords = keyword_analyzer.extract_weighted_keywords(keywords_lower)
    
    # Добавляем исходные ключевые слова с весом
    for keyword in keywords:
        keyword_lower = keyword.lower()
        keyword_base = title_analyzer._get_base_form(keyword_lower)
        if keyword_base:
            weighted_keywords[keyword_base] = weighted_keywords.get(keyword_base, 0) + 2.0
    
    # Нормализуем веса
    if weighted_keywords:
        max_weight = max(weighted_keywords.values())
        normalized_keywords = {k: v/max_weight for k, v in weighted_keywords.items()}
    else:
        normalized_keywords = {}
    
    # Track duplicate titles to keep only one version (with highest DOI number)
    title_to_work_map = {}
    
    with st.spinner(f"Analyzing {len(works)} works with enhanced algorithm..."):
        analyzed = []
        
        for work in works:
            # ========== ИСПРАВЛЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ==========
            # Проверяем работу фильтрами
            if not passes_filters(work, year_filter, citation_ranges):
                continue
            # ========== ИСПРАВЛЕНИЕ ЗАКАНЧИВАЕТСЯ ЗДЕСЬ ==========
            
            title = work.get('title', '')
            
            if not title:  # Skip works without title
                continue
            
            # Extract and clean DOI for comparison
            doi_raw = work.get('doi', '')
            doi_clean = ''
            if doi_raw:
                doi_clean = str(doi_raw).replace('https://doi.org/', '').lower()
            
            # RULE 1: Exclude works that match input DOIs
            if doi_clean and doi_clean in input_dois:
                logger.debug(f"Excluding work with input DOI: {doi_clean}")
                continue
            
            # Calculate enhanced relevance score
            relevance_score, matched_keywords = calculate_enhanced_relevance(
                work, normalized_keywords, title_analyzer
            )
            
            if relevance_score > 0:
                enriched = enrich_work_data(work)
                enriched.update({
                    'relevance_score': relevance_score,
                    'matched_keywords': matched_keywords,
                    'analysis_time': datetime.now().isoformat()
                })
                
                # RULE 2: Handle duplicate titles
                title_normalized = title.strip().lower()
                
                if title_normalized in title_to_work_map:
                    # We have a duplicate title, compare DOIs
                    existing_work = title_to_work_map[title_normalized]
                    existing_doi = existing_work.get('doi', '').lower()
                    current_doi = enriched.get('doi', '').lower()
                    
                    # Extract numeric parts from DOIs for comparison
                    existing_numeric = extract_numeric_from_doi(existing_doi)
                    current_numeric = extract_numeric_from_doi(current_doi)
                    
                    # Keep the work with higher numeric DOI (or higher score if DOIs equal)
                    if current_numeric > existing_numeric:
                        # Replace with current work
                        title_to_work_map[title_normalized] = enriched
                        logger.debug(f"Replacing duplicate title '{title[:50]}...' with higher DOI")
                    elif current_numeric == existing_numeric:
                        # If DOIs are equal, keep the one with higher relevance score
                        if enriched['relevance_score'] > existing_work['relevance_score']:
                            title_to_work_map[title_normalized] = enriched
                            logger.debug(f"Replacing duplicate title '{title[:50]}...' with higher score")
                    # else: keep existing work
                else:
                    # First occurrence of this title
                    title_to_work_map[title_normalized] = enriched
        
        # Convert map back to list
        analyzed = list(title_to_work_map.values())
        
        # Многокритериальная сортировка
        analyzed.sort(key=lambda x: (
            -x['relevance_score'],          # 1. Релевантность
            -x.get('publication_year', 0),  # 2. Новизна
            -x.get('cited_by_count', 0)     # 3. Цитирования (в пределах диапазона)
        ))
        
        # Apply top_n limit
        result = analyzed[:top_n]
        
        # Log summary statistics
        logger.info(f"Found {len(result)} unique works after filtering")
        logger.info(f"Removed {len(works) - len(analyzed)} works due to filters")
        if len(analyzed) > len(result):
            logger.info(f"Limited from {len(analyzed)} to {len(result)} works by top_n parameter")
        
        return result

# ============================================================================
# НОВАЯ ФУНКЦИЯ ДЛЯ УЛУЧШЕННОГО АНАЛИЗА С ФИЛЬТРАЦИЕЙ НА СТОРОНЕ API
# ============================================================================

def analyze_filtered_works_for_topic(
    topic_id: str,
    keywords: List[str],
    selected_years: List[int],
    selected_citations: List[Tuple[int, int]],
    max_works: Optional[int] = None,
    top_n: int = 100
) -> Tuple[List[dict], int]:
    """
    Analyze works for a specific topic with server-side filtering.
    
    Args:
        topic_id: Идентификатор темы
        keywords: Список ключевых слов для анализа
        selected_years: Список выбранных годов
        selected_citations: Список диапазонов цитирований
        max_works: Максимальное количество работ для загрузки (None = все)
        top_n: Количество топ результатов для возврата
    
    Returns:
        Кортеж (список релевантных работ, общее количество работ после фильтров)
    """
    # Get input DOIs from session state to exclude them from recommendations
    input_dois = set()
    if 'dois' in st.session_state:
        # Normalize input DOIs (remove https://doi.org/ prefix for comparison)
        for doi in st.session_state.dois:
            if doi.startswith('https://doi.org/'):
                clean_doi = doi.replace('https://doi.org/', '').lower()
            else:
                clean_doi = doi.lower()
            input_dois.add(clean_doi)
        logger.info(f"Excluding {len(input_dois)} input DOIs from recommendations")
    
    # Загружаем отфильтрованные работы
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, count, page, total):
        progress_bar.progress(progress)
        status_text.text(f"Page {page}: {count}/{total if total > 0 else '?'} works fetched")
    
    works, total_count = fetch_filtered_works_by_topic(
        topic_id=topic_id,
        years_filter=selected_years,
        citations_filter=selected_citations,
        max_results=max_works,
        progress_callback=update_progress
    )
    
    progress_bar.empty()
    status_text.empty()
    
    if not works:
        logger.warning(f"No works found for topic {topic_id} with given filters")
        return [], total_count
    
    logger.info(f"Loaded {len(works)} works (total after filters: {total_count})")
    
    # Инициализация анализаторов
    title_analyzer = TitleKeywordsAnalyzer()
    keyword_analyzer = EnhancedKeywordAnalyzer()
    
    # Преобразуем ключевые слова в взвешенный словарь
    keywords_lower = [kw.lower() for kw in keywords]
    weighted_keywords = keyword_analyzer.extract_weighted_keywords(keywords_lower)
    
    # Добавляем исходные ключевые слова с весом
    for keyword in keywords:
        keyword_lower = keyword.lower()
        keyword_base = title_analyzer._get_base_form(keyword_lower)
        if keyword_base:
            weighted_keywords[keyword_base] = weighted_keywords.get(keyword_base, 0) + 2.0
    
    # Нормализуем веса
    if weighted_keywords:
        max_weight = max(weighted_keywords.values())
        normalized_keywords = {k: v/max_weight for k, v in weighted_keywords.items()}
    else:
        normalized_keywords = {}
    
    # Track duplicate titles to keep only one version (with highest DOI number)
    title_to_work_map = {}
    
    with st.spinner(f"Analyzing {len(works)} works with enhanced algorithm..."):
        analyzed = []
        
        for work in works:
            title = work.get('title', '')
            
            if not title:  # Skip works without title
                continue
            
            # Extract and clean DOI for comparison
            doi_raw = work.get('doi', '')
            doi_clean = ''
            if doi_raw:
                doi_clean = str(doi_raw).replace('https://doi.org/', '').lower()
            
            # RULE 1: Exclude works that match input DOIs
            if doi_clean and doi_clean in input_dois:
                logger.debug(f"Excluding work with input DOI: {doi_clean}")
                continue
            
            # Calculate enhanced relevance score
            relevance_score, matched_keywords, chemical_info = calculate_enhanced_relevance(
                work, normalized_keywords, title_analyzer
            )
            
            # Затем обновите enriched:
            enriched = enrich_work_data(work)
            enriched.update({
                'relevance_score': relevance_score,
                'matched_keywords': matched_keywords,
                'analysis_time': datetime.now().isoformat(),
                'has_chemical_formula': chemical_info.get('has_chemical_formula', False),
                'has_organic_pattern': chemical_info.get('has_organic_pattern', False),
                'chemical_bonus': chemical_info.get('total_chemical_bonus', 0.0)
            })
            
            if relevance_score > 0:
                enriched = enrich_work_data(work)
                enriched.update({
                    'relevance_score': relevance_score,
                    'matched_keywords': matched_keywords,
                    'analysis_time': datetime.now().isoformat()
                })
                
                # RULE 2: Handle duplicate titles
                title_normalized = title.strip().lower()
                
                if title_normalized in title_to_work_map:
                    # We have a duplicate title, compare DOIs
                    existing_work = title_to_work_map[title_normalized]
                    existing_doi = existing_work.get('doi', '').lower()
                    current_doi = enriched.get('doi', '').lower()
                    
                    # Extract numeric parts from DOIs for comparison
                    existing_numeric = extract_numeric_from_doi(existing_doi)
                    current_numeric = extract_numeric_from_doi(current_doi)
                    
                    # Keep the work with higher numeric DOI (or higher score if DOIs equal)
                    if current_numeric > existing_numeric:
                        # Replace with current work
                        title_to_work_map[title_normalized] = enriched
                        logger.debug(f"Replacing duplicate title '{title[:50]}...' with higher DOI")
                    elif current_numeric == existing_numeric:
                        # If DOIs are equal, keep the one with higher relevance score
                        if enriched['relevance_score'] > existing_work['relevance_score']:
                            title_to_work_map[title_normalized] = enriched
                            logger.debug(f"Replacing duplicate title '{title[:50]}...' with higher score")
                    # else: keep existing work
                else:
                    # First occurrence of this title
                    title_to_work_map[title_normalized] = enriched
        
        # Convert map back to list
        analyzed = list(title_to_work_map.values())
        
        # Многокритериальная сортировка
        analyzed.sort(key=lambda x: (
            -x['relevance_score'],          # 1. Релевантность
            -x.get('publication_year', 0),  # 2. Новизна
            -x.get('cited_by_count', 0)     # 3. Цитирования (в пределах диапазона)
        ))
        
        # Apply top_n limit
        result = analyzed[:top_n]
        
        # Log summary statistics
        logger.info(f"Found {len(result)} unique works after filtering")
        logger.info(f"Removed {len(works) - len(analyzed)} works due to filters")
        if len(analyzed) > len(result):
            logger.info(f"Limited from {len(analyzed)} to {len(result)} works by top_n parameter")
        
        return result, total_count

# ============================================================================
# ФУНКЦИИ ЭКСПОРТА
# ============================================================================

def generate_csv(data: List[dict]) -> str:
    """Генерация CSV файла"""
    # Существующий код:
    display_data = []
    for work in data:
        # Существующие поля:
        row = {
            'Title': work.get('title', ''),
            'Authors': '; '.join(work.get('authors', [])),
            'Journal': work.get('journal_name', ''),
            'Year': work.get('publication_year', ''),
            'Citations': work.get('cited_by_count', 0),
            'Relevance Score': work.get('relevance_score', 0),
            'DOI': work.get('doi', ''),
            'Open Access': 'Yes' if work.get('is_oa') else 'No',
            'Abstract': work.get('abstract', '')[:200],
            'Matched Keywords': '; '.join(work.get('matched_keywords', [])),
            
            # ДОБАВЬТЕ ЗДЕСЬ (после 'Matched Keywords'):
            'Has Chemical Formula': 'Yes' if work.get('has_chemical_formula') else 'No',
            'Has Organic Pattern': 'Yes' if work.get('has_organic_pattern') else 'No',
            'Chemical Bonus': work.get('chemical_bonus', 0.0),
            'Primary Topic': work.get('primary_topic', ''),
            'Institutions': '; '.join(work.get('institutions', []))
        }
        display_data.append(row)
    
    df = pd.DataFrame(display_data)
    return df.to_csv(index=False, encoding='utf-8-sig')

def generate_excel(data: List[dict]) -> bytes:
    """Генерация Excel файла"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Существующий код создания display_data:
        display_data = []
        for work in data:
            row = {
                'Title': work.get('title', ''),
                'Authors': '; '.join(work.get('authors', [])),
                'Journal': work.get('journal_name', ''),
                'Year': work.get('publication_year', ''),
                'Citations': work.get('cited_by_count', 0),
                'Relevance Score': work.get('relevance_score', 0),
                'DOI': work.get('doi', ''),
                'Open Access': 'Yes' if work.get('is_oa') else 'No',
                'Abstract': work.get('abstract', '')[:200],
                'Matched Keywords': '; '.join(work.get('matched_keywords', [])),
                
                # ДОБАВЬТЕ ЗДЕСЬ (после 'Matched Keywords'):
                'Has Chemical Formula': 'Yes' if work.get('has_chemical_formula') else 'No',
                'Has Organic Pattern': 'Yes' if work.get('has_organic_pattern') else 'No',
                'Chemical Bonus': work.get('chemical_bonus', 0.0),
                'Primary Topic': work.get('primary_topic', ''),
                'Institutions': '; '.join(work.get('institutions', []))
            }
            display_data.append(row)
        
        df = pd.DataFrame(display_data)
        df.to_excel(writer, sheet_name='Papers', index=False)
        
        # Добавляем заголовок
        workbook = writer.book
        worksheet = writer.sheets['Papers']
        
        # Форматирование
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#667eea',
            'font_color': 'white',
            'border': 1
        })
        
        # Применяем к заголовкам
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Авто-ширина колонок
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(column_len, 50))
    
    return output.getvalue()

def generate_pdf(data: List[dict], topic_name: str) -> bytes:
    """Генерация PDF файла с улучшенным дизайном и активными гиперссылками"""
    
    # Вспомогательная функция для очистки текста
    def clean_text(text):
        if not text:
            return ""
        # Заменяем HTML сущности и теги
        text = re.sub(r'<[^>]+>', '', text)  # Удаляем HTML теги
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return text
    
    buffer = io.BytesIO()
    
    # Используем A4 для большего пространства
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        topMargin=1*cm,
        bottomMargin=1*cm,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm
    )
    
    styles = getSampleStyleSheet()
    
    # ========== СОЗДАНИЕ КАСТОМНЫХ СТИЛЕЙ ==========
    
    # Стиль для заголовка
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Стиль для подзаголовка
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    # Стиль для информации о теме
    topic_style = ParagraphStyle(
        'CustomTopic',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#16A085'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Стиль для мета-информации
    meta_style = ParagraphStyle(
        'CustomMeta',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Стиль для названия статьи (обычный, не ссылка)
    paper_title_style = ParagraphStyle(
        'CustomPaperTitle',
        parent=styles['Heading4'],
        fontSize=11,
        textColor=colors.HexColor('#2980B9'),
        spaceAfter=4,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    # Стиль для авторов
    authors_style = ParagraphStyle(
        'CustomAuthors',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    # Стиль для деталей статьи
    details_style = ParagraphStyle(
        'CustomDetails',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    # Стиль для метрик
    metrics_style = ParagraphStyle(
        'CustomMetrics',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#27AE60'),
        spaceAfter=0,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    # Стиль для ключевых слов
    keywords_style = ParagraphStyle(
        'CustomKeywords',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#E74C3C'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica-Oblique'
    )
    
    # Стиль для нижнего колонтитула
    footer_style = ParagraphStyle(
        'CustomFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#95A5A6'),
        spaceBefore=15,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Стиль для разделителя
    separator_style = ParagraphStyle(
        'CustomSeparator',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#BDC3C7'),
        spaceAfter=10,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Стиль для ссылок
    link_style = ParagraphStyle(
        'CustomLink',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.blue,
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica',
        underline=True
    )
    
    story = []
    
    # ========== ЗАГОЛОВОЧНАЯ СТРАНИЦА ==========
    
    story.append(Spacer(1, 1*cm))

    # Добавляем логотип
    try:
        # Пробуем несколько возможных путей
        possible_paths = [
            "logo.png",  # Текущая директория
            "./logo.png",  # Относительный путь
            "app/logo.png",  # Если в поддиректории
            os.path.join(os.path.dirname(__file__), "logo.png"),  # Абсолютный путь
            os.path.join(os.getcwd(), "logo.png")  # Текущая рабочая директория
        ]
        
        logo_path = None
        for path in possible_paths:
            if os.path.exists(path):
                logo_path = path
                break
        
        if logo_path:
            # Проверяем, что файл действительно является изображением
            try:
                # Проверяем с помощью PIL
                pil_img = PILImage.open(logo_path)
                pil_img.verify()  # Проверяем целостность файла
                
                # Используем Image из reportlab
                logo = Image(logo_path, width=160, height=80)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 0.5*cm))
                logger.info(f"Logo loaded successfully from: {logo_path}")
            except Exception as img_error:
                logger.warning(f"Invalid image file at {logo_path}: {img_error}")
                raise ValueError("Invalid image file")
        else:
            logger.warning("Logo file 'logo.png' not found in any expected location")
            raise FileNotFoundError("Logo not found")
            
    except Exception as e:
        logger.error(f"Could not load logo: {e}")
        # Если логотип не загрузился, показываем эмодзи
        story.append(Paragraph("🔬", ParagraphStyle(
            'LogoEmoji',
            parent=styles['Heading1'],
            fontSize=40,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER
        )))
        story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("CTA Article Recommender Pro", title_style))
    story.append(Paragraph("Fresh Papers Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.8*cm))
    
    # Информация о теме
    story.append(Paragraph(f"RESEARCH TOPIC:", topic_style))
    story.append(Paragraph(f"{topic_name.upper()}", subtitle_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Мета-информация
    current_date = datetime.now().strftime('%B %d, %Y at %H:%M')
    story.append(Paragraph(f"Generated on {current_date}", meta_style))
    story.append(Paragraph(f"Total papers analyzed: {len(data)}", meta_style))
    
    # Расчет статистик
    if data:
        avg_citations = np.mean([w.get('cited_by_count', 0) for w in data])
        oa_count = sum(1 for w in data if w.get('is_oa'))
        recent_count = sum(1 for w in data if w.get('publication_year', 0) >= datetime.now().year - 2)
        
        stats_text = f"""
        Average citations: {avg_citations:.1f} | 
        Open Access papers: {oa_count} | 
        Recent papers (≤2 years): {recent_count}
        """
        story.append(Paragraph(stats_text, meta_style))
    
    story.append(Spacer(1, 1.5*cm))
    
    # Копирайт информация
    story.append(Paragraph("© CTA - Chimica Techno Acta", footer_style))
    story.append(Paragraph("https://chimicatechnoacta.ru", footer_style))
    story.append(Paragraph("Developed by daM©", footer_style))
    
    # Разделитель страниц
    story.append(PageBreak())
    
    # ========== INITIAL DATA ==========
    story.append(Paragraph("INITIAL DATA", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Получаем данные из сессии
    initial_dois = st.session_state.get('dois', [])
    selected_topic = st.session_state.get('selected_topic', 'Not selected')
    selected_years = st.session_state.get('selected_years', [])
    selected_ranges = st.session_state.get('selected_ranges', [(0, 10)])
    
    # Создаем таблицу с основными параметрами
    initial_data = [
        ["Parameter", "Value"],
        ["Total Input DOIs", len(initial_dois)],
        ["Selected Topic", clean_text(selected_topic)],
        ["Publication Years", ", ".join(map(str, selected_years))],
        ["Citation Ranges", format_citation_ranges(selected_ranges)],
        ["Analysis Date", current_date],
        ["Papers Found", len(data)]
    ]
    
    initial_table = Table(initial_data, colWidths=[doc.width/2.5, doc.width*3/5])
    initial_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5DBDB')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F4F4')]),
        ('WORDWRAP', (0, 0), (-1, -1), 'LTR'),
    ]))
    
    story.append(initial_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Отображаем DOI в виде кликабельного списка
    if initial_dois:
        story.append(Paragraph("<b>Input DOIs:</b>", ParagraphStyle(
            'DOIsHeader',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        )))
        
        # Функция для создания ссылок из DOI
        def create_doi_link(doi):
            # Проверяем, есть ли уже https://doi.org/ в начале
            if doi.startswith('10.'):
                doi_url = f"https://doi.org/{doi}"
            elif doi.startswith('https://doi.org/'):
                doi_url = doi
            else:
                doi_url = f"https://doi.org/{doi}"
            
            # Экранируем для XML
            doi_url_clean = doi_url.replace('&', '&amp;')
            
            # Создаем строку с ссылкой (используем <a> вместо <link> для лучшей совместимости)
            return f"<a href='{doi_url_clean}' color='blue'>{doi_url}</a>"
        
        max_dois_to_show = min(300, len(initial_dois))
        for i, doi in enumerate(initial_dois[:max_dois_to_show], 1):
            doi_link = create_doi_link(doi)
            story.append(Paragraph(f"{i}. {doi_link}", ParagraphStyle(
                'DOILink',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.blue,
                spaceAfter=2,
                leftIndent=10,
                fontName='Helvetica',
                underline=True
            )))
        
        # Если DOI больше 25, показываем информацию
        if len(initial_dois) > max_dois_to_show:
            story.append(Paragraph(
                f"... and {len(initial_dois) - max_dois_to_show} more DOI entries", 
                ParagraphStyle(
                    'DOIsMore',
                    parent=styles['Normal'],
                    fontSize=8,
                    textColor=colors.gray,
                    spaceAfter=10,
                    leftIndent=10,
                    fontName='Helvetica-Oblique'
                )
            ))
    
    story.append(Spacer(1, 1*cm))
    
    # ========== TABLE OF CONTENTS ==========
    story.append(Paragraph("TABLE OF CONTENTS", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Создаем оглавление
    toc_items = []
    for i in range(min(20, len(data))):  # Ограничиваем 20 записями для читаемости
        title = data[i].get('title', 'Untitled')
        # Удаляем HTML-теги
        title_clean = re.sub(r'<[^>]+>', '', title)
        # Экранируем специальные символы
        title_clean = title_clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        toc_items.append(f"{i+1}. {title_clean[:60]}...")
    
    toc_text = "<br/>".join(toc_items[:15])  # Первые 15 в оглавлении
    story.append(Paragraph(toc_text, details_style))
    
    if len(data) > 15:
        story.append(Paragraph(f"... and {len(data)-15} more papers", details_style))
    
    story.append(PageBreak())
    
    # ========== ДЕТАЛЬНЫЙ ОТЧЕТ ПО СТАТЬЯМ ==========
    
    story.append(Paragraph("DETAILED PAPER ANALYSIS", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Вспомогательная функция для очистки текста
    def clean_text(text):
        if not text:
            return ""
        # Заменяем HTML сущности и теги
        text = re.sub(r'<[^>]+>', '', text)  # Удаляем HTML теги
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return text
    
    # Обрабатываем каждую статью (ограничиваем 30 для читаемости)
    for i, work in enumerate(data[:30], 1):
        # Существующий код для заголовка, авторов, метрик...
        
        # Добавьте химическую информацию после ключевых слов:
        if work.get('has_chemical_formula') or work.get('has_organic_pattern'):
            chem_text = ""
            if work.get('has_chemical_formula'):
                chem_text += "🧪 Chemical formula "
            if work.get('has_organic_pattern'):
                chem_text += "⚗️ Organic pattern "
            if work.get('chemical_bonus', 0) > 0:
                chem_text += f"(Bonus: {work.get('chemical_bonus', 0):.1f})"
            
            story.append(Paragraph(f"<b>Chemical Info:</b> {clean_text(chem_text)}", keywords_style))
            
        # Заголовок статьи
        title = clean_text(work.get('title', 'No title available'))
        story.append(Paragraph(f"{i}. {title}", paper_title_style))
        
        # Авторы
        authors = work.get('authors', [])
        if authors:
            authors_text = ', '.join(authors[:3])
            if len(authors) > 3:
                authors_text += f' et al. ({len(authors)} authors)'
            story.append(Paragraph(f"<b>Authors:</b> {clean_text(authors_text)}", authors_style))
        
        # Основные метрики
        citations = work.get('cited_by_count', 0)
        year = work.get('publication_year', 'N/A')
        relevance = work.get('relevance_score', 0)
        journal = clean_text(work.get('journal_name', 'N/A')[:40])
        
        metrics_text = f"""
        <b>Citations:</b> {citations} | 
        <b>Year:</b> {year} | 
        <b>Relevance Score:</b> {relevance}/10 | 
        <b>Journal:</b> {journal} | 
        <b>Open Access:</b> {'Yes' if work.get('is_oa') else 'No'}
        """
        story.append(Paragraph(metrics_text, metrics_style))
        
        # Ключевые слова (если есть)
        if work.get('matched_keywords'):
            keywords = ', '.join(work.get('matched_keywords', [])[:5])
            story.append(Paragraph(f"<b>Matched Keywords:</b> {clean_text(keywords)}", keywords_style))
        
        # DOI и ссылка
        doi = work.get('doi', '')
        doi_url = work.get('doi_url', '')
        
        if doi:
            if doi_url:
                # Добавляем ссылку как отдельный параграф
                story.append(Paragraph(f"<b>DOI:</b> {clean_text(doi)}", details_style))
                story.append(Paragraph(f"<b>Link:</b> {clean_text(doi_url)}", link_style))
            else:
                story.append(Paragraph(f"<b>DOI:</b> {clean_text(doi)}", details_style))
        
        # Разделитель между статьями
        if i < min(30, len(data)):
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("─" * 50, separator_style))
            story.append(Spacer(1, 0.3*cm))
        else:
            story.append(Spacer(1, 0.3*cm))
    
    # ========== СТАТИСТИЧЕСКАЯ СТРАНИЦА ==========
    
    if len(data) > 10:
        story.append(PageBreak())
        story.append(Paragraph("STATISTICAL SUMMARY", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Подготовка данных для статистики
        citations_list = [w.get('cited_by_count', 0) for w in data]
        years_list = [w.get('publication_year', 0) for w in data if w.get('publication_year', 0) > 1900]
        
        if citations_list and years_list:
            # Базовая статистика
            stats_data = [
                ["Metric", "Value"],
                ["Total Papers", len(data)],
                ["Average Citations", f"{np.mean(citations_list):.2f}"],
                ["Median Citations", f"{np.median(citations_list):.2f}"],
                ["Min Citations", min(citations_list)],
                ["Max Citations", max(citations_list)],
                ["Open Access Papers", sum(1 for w in data if w.get('is_oa'))],
                ["Average Year", f"{np.mean(years_list):.1f}"],
                ["Most Recent Year", max(years_list) if years_list else "N/A"],
                ["Average Relevance", f"{np.mean([w.get('relevance_score', 0) for w in data]):.2f}/10"]
            ]
            
            # Создаем таблицу статистики
            stats_table = Table(stats_data, colWidths=[doc.width/2.5, doc.width/3])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5DBDB')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F4F4')]),
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 1*cm))
            
            # Распределение по годам
            if years_list:
                year_counts = {}
                for year in years_list:
                    year_counts[year] = year_counts.get(year, 0) + 1
                
                sorted_years = sorted(year_counts.items())
                year_data = [["Year", "Number of Papers"]] + [[str(y), str(c)] for y, c in sorted_years[-10:]]  # Последние 10 лет
                
                if len(year_data) > 1:
                    story.append(Paragraph("Publications by Year (last 3 years)", subtitle_style))
                    year_table = Table(year_data, colWidths=[doc.width/4, doc.width/4])
                    year_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5DBDB')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F4F4')]),
                    ]))
                    story.append(year_table)
    
    # ========== ЗАКЛЮЧЕНИЕ ==========
    
    story.append(PageBreak())
    story.append(Paragraph("CONCLUSION & RECOMMENDATIONS", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Рекомендации на основе анализа
    conclusions = [
        f"This report analyzed {len(data)} fresh papers in the field of '{topic_name}'.",
        "Papers with low citation counts often represent emerging ideas or niche research areas.",
        "Consider these papers for:",
        "• Literature reviews of emerging topics",
        "• Identifying research gaps",
        "• Finding novel methodologies",
        "• Cross-disciplinary connections"
    ]
    
    for conclusion in conclusions:
        story.append(Paragraph(clean_text(conclusion), ParagraphStyle(
            'Conclusion',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=4,
            leftIndent=20 if conclusion.startswith('•') else 0,
            fontName='Helvetica'
        )))
    
    story.append(Spacer(1, 1*cm))
    
    # Заключительные замечания
    story.append(Paragraph("FINAL NOTES", subtitle_style))
    final_notes = [
        "This report was generated automatically by CTA Article Recommender Pro.",
        "All data is sourced from OpenAlex API and is subject to their terms of use.",
        "For the most current data, please visit the original sources via the provided DOIs.",
        "Citation counts are as of the report generation date and may change over time."
    ]
    
    for note in final_notes:
        story.append(Paragraph(f"• {clean_text(note)}", details_style))
    
    # Нижний колонтитул на последней странице
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("© CTA Article Recommender Pro - https://chimicatechnoacta.ru", footer_style))
    story.append(Paragraph(f"Report ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}", 
                         ParagraphStyle(
                             'ReportID',
                             parent=styles['Normal'],
                             fontSize=7,
                             textColor=colors.HexColor('#BDC3C7'),
                             alignment=TA_CENTER
                         )))
    
    # ========== ГЕНЕРАЦИЯ PDF ==========
    
    doc.build(story)
    
    return buffer.getvalue()

def generate_txt(data: List[dict], topic_name: str) -> str:
    """Генерация TXT файла с улучшенным форматированием и структурой"""
    
    output = []
    
    # ========== ЗАГОЛОВОК ==========
    output.append("=" * 80)
    output.append("CTA Article Recommender Pro")
    output.append("Under-Cited Papers Analysis Report")
    output.append("=" * 80)
    output.append("")
    
    # ========== ИНФОРМАЦИЯ О ТЕМЕ ==========
    output.append("RESEARCH TOPIC:")
    output.append(f"  {topic_name.upper()}")
    output.append("")
    
    # ========== МЕТА-ИНФОРМАЦИЯ ==========
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output.append("REPORT INFORMATION:")
    output.append(f"  Generated: {current_date}")
    output.append(f"  Papers analyzed: {len(data)}")
    
    if data:
        avg_citations = np.mean([w.get('cited_by_count', 0) for w in data])
        oa_count = sum(1 for w in data if w.get('is_oa'))
        recent_count = sum(1 for w in data if w.get('publication_year', 0) >= datetime.now().year - 2)
        
        output.append(f"  Average citations: {avg_citations:.2f}")
        output.append(f"  Open Access papers: {oa_count}")
        output.append(f"  Recent papers (≤2 years): {recent_count}")
    
    output.append("")
    output.append("© CTA - Chemical Technology Acta")
    output.append("https://chimicatechnoacta.ru")
    output.append("Developed by daM©")
    output.append("")
    output.append("=" * 80)
    output.append("")

    # ========== INITIAL DATA ==========
    output.append("INITIAL DATA")
    output.append("=" * 40)
    
    # Получаем данные из сессии
    initial_dois = st.session_state.get('dois', [])
    selected_topic = st.session_state.get('selected_topic', 'Not selected')
    selected_years = st.session_state.get('selected_years', [])
    selected_ranges = st.session_state.get('selected_ranges', [(0, 10)])
    
    # Основные параметры
    output.append(f"  Total Input DOIs: {len(initial_dois)}")
    output.append(f"  Selected Topic: {selected_topic}")
    output.append(f"  Publication Years: {', '.join(map(str, selected_years))}")
    output.append(f"  Citation Ranges: {format_citation_ranges(selected_ranges)}")
    output.append(f"  Analysis Date: {current_date}")
    output.append(f"  Papers Found: {len(data)}")
    
    # Список DOI
    if initial_dois:
        output.append("")
        output.append("  Input DOIs:")
        output.append("  " + "-" * 36)
        
        max_dois_to_show = min(300, len(initial_dois))
        for i, doi in enumerate(initial_dois[:max_dois_to_show], 1):
            # Форматируем DOI в полный URL
            if doi.startswith('10.'):
                doi_url = f"https://doi.org/{doi}"
            elif doi.startswith('https://doi.org/'):
                doi_url = doi
            else:
                doi_url = f"https://doi.org/{doi}"
            
            output.append(f"  {i:3d}. {doi_url}")
        
        if len(initial_dois) > max_dois_to_show:
            output.append(f"  ... and {len(initial_dois) - max_dois_to_show} more")
        
        output.append("  " + "-" * 36)
    
    output.append("")
    output.append("=" * 80)
    output.append("")
    
    # ========== ОГЛАВЛЕНИЕ ==========
    output.append("TABLE OF CONTENTS")
    output.append("-" * 40)
    
    # Группируем статьи по релевантности
    high_relevance = [w for w in data if w.get('relevance_score', 0) >= 8]
    medium_relevance = [w for w in data if 5 <= w.get('relevance_score', 0) < 8]
    low_relevance = [w for w in data if w.get('relevance_score', 0) < 5]
    
    output.append(f"  High Relevance (Score ≥ 8): {len(high_relevance)} papers")
    output.append(f"  Medium Relevance (5-7): {len(medium_relevance)} papers")
    output.append(f"  Low Relevance (Score < 5): {len(low_relevance)} papers")
    output.append("")
    
    # Быстрый обзор по годам
    if data:
        years = [w.get('publication_year', 0) for w in data if w.get('publication_year', 0) > 1900]
        if years:
            output.append("PUBLICATION YEAR DISTRIBUTION:")
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
            
            for year in sorted(year_counts.keys(), reverse=True)[:5]:  # Топ 5 последних лет
                output.append(f"  {year}: {year_counts[year]} papers")
            output.append("")
    
    output.append("=" * 80)
    output.append("")
    
    # ========== ДЕТАЛЬНЫЙ АНАЛИЗ СТАТЕЙ ==========
    output.append("DETAILED PAPER ANALYSIS")
    output.append("=" * 80)
    output.append("")
    
    for i, work in enumerate(data, 1):
        # Номер и релевантность
        relevance_score = work.get('relevance_score', 0)
        relevance_stars = "★" * min(int(relevance_score), 5) + "☆" * max(5 - int(relevance_score), 0)
        
        output.append(f"PAPER #{i:03d}")
        output.append(f"Relevance: {relevance_score}/10 {relevance_stars}")
        output.append("-" * 40)
        
        # Заголовок
        title = work.get('title', 'No title available')
        output.append(f"TITLE: {title}")
        
        # Авторы
        authors = work.get('authors', [])
        if authors:
            output.append(f"AUTHORS: {', '.join(authors[:3])}")
            if len(authors) > 3:
                output.append(f"         + {len(authors) - 3} more authors")
        
        # Основные метрики
        citations = work.get('cited_by_count', 0)
        year = work.get('publication_year', 'N/A')
        journal = work.get('journal_name', 'N/A')
        
        output.append("METRICS:")
        output.append(f"  • Citations: {citations}")
        output.append(f"  • Year: {year}")
        output.append(f"  • Journal/Conference: {journal}")
        output.append(f"  • Open Access: {'Yes' if work.get('is_oa') else 'No'}")
        
        # Ключевые слова
        if work.get('matched_keywords'):
            keywords = work.get('matched_keywords', [])
            output.append(f"KEYWORDS: {', '.join(keywords[:5])}")
            if len(keywords) > 5:
                output.append(f"          + {len(keywords) - 5} more keywords")
        
        # DOI и ссылка
        doi = work.get('doi', '')
        doi_url = work.get('doi_url', '')
        
        if doi:
            output.append(f"DOI: {doi}")
            if doi_url:
                output.append(f"LINK: {doi_url}")
        
        # Абстракт (если есть и короткий)
        abstract = work.get('abstract', '')
        if abstract and len(abstract) < 300:
            output.append("ABSTRACT:")
            # Форматируем абстракт с переносами строк
            words = abstract.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 70:
                    current_line += " " + word if current_line else word
                else:
                    lines.append("  " + current_line)
                    current_line = word
            if current_line:
                lines.append("  " + current_line)
            output.extend(lines)
        
        # Разделитель между статьями
        if i < len(data):
            output.append("")
            output.append("─" * 60)
            output.append("")
    
    output.append("=" * 80)
    output.append("")
    
    # ========== СТАТИСТИЧЕСКАЯ СВОДКА ==========
    if len(data) > 5:
        output.append("STATISTICAL SUMMARY")
        output.append("=" * 80)
        output.append("")
        
        citations_list = [w.get('cited_by_count', 0) for w in data]
        relevance_list = [w.get('relevance_score', 0) for w in data]
        
        if citations_list:
            output.append("CITATION ANALYSIS:")
            output.append(f"  Average: {np.mean(citations_list):.2f}")
            output.append(f"  Median: {np.median(citations_list):.2f}")
            output.append(f"  Minimum: {min(citations_list)}")
            output.append(f"  Maximum: {max(citations_list)}")
            output.append(f"  Standard Deviation: {np.std(citations_list):.2f}")
            output.append("")
            
            # Распределение по количеству цитирований
            output.append("CITATION DISTRIBUTION:")
            ranges = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 1000)]
            for min_cit, max_cit in ranges:
                count = sum(1 for w in data if min_cit <= w.get('cited_by_count', 0) <= max_cit)
                if count > 0:
                    if min_cit == max_cit:
                        range_str = f"Exactly {min_cit}"
                    else:
                        range_str = f"{min_cit}-{max_cit}"
                    percentage = (count / len(data)) * 100
                    output.append(f"  {range_str:12} citations: {count:3d} papers ({percentage:5.1f}%)")
            output.append("")
        
        if relevance_list:
            output.append("RELEVANCE SCORE ANALYSIS:")
            output.append(f"  Average: {np.mean(relevance_list):.2f}/10")
            output.append(f"  Median: {np.median(relevance_list):.2f}/10")
            
            # Распределение по релевантности
            relevance_counts = {score: 0 for score in range(1, 11)}
            for score in relevance_list:
                rounded = min(int(score), 10)
                relevance_counts[rounded] = relevance_counts.get(rounded, 0) + 1
            
            output.append("  Distribution:")
            for score in range(10, 0, -1):
                count = relevance_counts.get(score, 0)
                if count > 0:
                    percentage = (count / len(data)) * 100
                    stars = "★" * min(score, 5) + "☆" * max(5 - score, 0)
                    output.append(f"    Score {score:2d}/10 {stars}: {count:3d} papers ({percentage:5.1f}%)")
            output.append("")
    
    # ========== ТОП РЕКОМЕНДАЦИЙ ==========
    if len(data) > 10:
        output.append("TOP RECOMMENDATIONS")
        output.append("=" * 80)
        output.append("")
        
        # Сортируем по релевантности, затем по годам (новые первыми)
        sorted_data = sorted(data, key=lambda x: (-x.get('relevance_score', 0), 
                                                  -x.get('publication_year', 0)))
        
        output.append("Highest Relevance & Most Recent:")
        for i, work in enumerate(sorted_data[:5], 1):
            title = work.get('title', '')[:70] + "..." if len(work.get('title', '')) > 70 else work.get('title', '')
            output.append(f"  {i}. {title}")
            output.append(f"     Year: {work.get('publication_year', 'N/A')}, "
                         f"Citations: {work.get('cited_by_count', 0)}, "
                         f"Score: {work.get('relevance_score', 0)}/10")
        
        output.append("")
        output.append("Most Cited (among under-cited):")
        # Берем статьи с ненулевыми цитированиями
        cited_papers = [w for w in data if w.get('cited_by_count', 0) > 0]
        if cited_papers:
            most_cited = sorted(cited_papers, key=lambda x: -x.get('cited_by_count', 0))
            for i, work in enumerate(most_cited[:3], 1):
                title = work.get('title', '')[:70] + "..." if len(work.get('title', '')) > 70 else work.get('title', '')
                output.append(f"  {i}. {title}")
                output.append(f"     Citations: {work.get('cited_by_count', 0)}, "
                             f"Year: {work.get('publication_year', 'N/A')}")
        
        output.append("")
        output.append("Newest Publications:")
        recent_papers = sorted(data, key=lambda x: -x.get('publication_year', 0))
        for i, work in enumerate(recent_papers[:3], 1):
            title = work.get('title', '')[:70] + "..." if len(work.get('title', '')) > 70 else work.get('title', '')
            output.append(f"  {i}. {title}")
            output.append(f"     Year: {work.get('publication_year', 'N/A')}, "
                         f"Citations: {work.get('cited_by_count', 0)}")
    
    # ========== ЗАКЛЮЧЕНИЕ ==========
    output.append("=" * 80)
    output.append("CONCLUSION")
    output.append("=" * 80)
    output.append("")
    
    conclusions = [
        f"This analysis identified {len(data)} under-cited papers in '{topic_name}'.",
        "",
        "KEY INSIGHTS:",
        "• These papers may represent emerging research trends",
        "• Low citation counts don't necessarily indicate low quality",
        "• Consider these for literature reviews and gap analysis",
        "• They may contain novel methodologies or cross-disciplinary insights",
        "",
        "RECOMMENDED ACTIONS:",
        "1. Review high-relevance papers for potential citations",
        "2. Use as starting points for systematic reviews",
        "3. Identify research gaps and opportunities",
        "4. Track emerging authors in this field",
        "",
        "REPORT METADATA:",
        f"• Generated by: CTA Article Recommender Pro",
        f"• Report ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12].upper()}",
        f"• Data source: OpenAlex API",
        f"• Analysis date: {current_date}",
        "",
        "© CTA - Chemical Technology Acta | https://chimicatechnoacta.ru",
        "This report is for research purposes only.",
        "Always verify information with original sources.",
        "",
        "End of Report"
    ]
    
    output.extend(conclusions)
    
    return "\n".join(output)

# ============================================================================
# КОМПОНЕНТЫ ИНТЕРФЕЙСА
# ============================================================================

def create_progress_bar(current_step: int, total_steps: int):
    """Создает прогресс бар мастер-процесса"""
    progress = current_step / total_steps
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress * 100}%"></div>
    </div>
    <div class="step-indicator">
        <span class="{'active' if current_step >= 1 else ''}">📥 Data Input</span>
        <span class="{'active' if current_step >= 2 else ''}">🔍 Analysis</span>
        <span class="{'active' if current_step >= 3 else ''}">🎯 Topic Selection</span>
        <span class="{'active' if current_step >= 4 else ''}">⚙️ Filters</span>
        <span class="{'active' if current_step >= 5 else ''}">📊 Results</span>
    </div>
    """, unsafe_allow_html=True)

def create_back_button():
    """Создает кнопку возврата назад"""
    if st.session_state.current_step > 1:
        if st.button("← Back", key="back_button", use_container_width=False):
            # При возврате на шаг 4 или 5, сбрасываем кэш результатов, чтобы фильтры применились заново
            if st.session_state.current_step in [4, 5]:
                if 'filtered_works' in st.session_state:
                    del st.session_state['filtered_works']
                if 'filtered_total_count' in st.session_state:
                    del st.session_state['filtered_total_count']
                if 'filter_stats' in st.session_state:
                    del st.session_state['filter_stats']
                if 'top_keywords' in st.session_state:
                    del st.session_state['top_keywords']
            
            st.session_state.current_step -= 1
            st.rerun()

def create_metric_card_compact(title: str, value, icon: str = "📊"):
    """Создает компактную карточку с метрикой"""
    st.markdown(f"""
    <div class="metric-card">
        <h4>{icon} {title}</h4>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def create_result_card_compact(work: dict, index: int):
    """Создает компактную карточку результата с химическими маркерами"""
    import html
    import re
    
    def safe_html(text: str) -> str:
        """Экранирует HTML, оставляя только безопасные теги"""
        if not text:
            return ""
        
        # Сохраняем безопасные теги
        replacements = {
            '<sub>': '___SUB_OPEN___',
            '</sub>': '___SUB_CLOSE___',
            '<sup>': '___SUP_OPEN___',
            '</sup>': '___SUP_CLOSE___',
            '<i>': '___I_OPEN___',
            '</i>': '___I_CLOSE___',
            '<b>': '___B_OPEN___',
            '</b>': '___B_CLOSE___',
            '<em>': '___EM_OPEN___',
            '</em>': '___EM_CLOSE___',
            '<strong>': '___STRONG_OPEN___',
            '</strong>': '___STRONG_CLOSE___'
        }
        
        # Заменяем безопасные теги на маркеры
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        # Экранируем все HTML
        text = html.escape(text)
        
        # Восстанавливаем безопасные теги
        for original, replacement in replacements.items():
            text = text.replace(replacement, original)
        
        return text
    
    citation_count = work.get('cited_by_count', 0)
    
    # Определяем цвет баджа цитирования
    if citation_count == 0:
        badge_color = "#4CAF50"
        badge_text = "0 citations"
    elif citation_count <= 3:
        badge_color = "#4CAF50"
        badge_text = f"{citation_count} citation{'s' if citation_count > 1 else ''}"
    elif citation_count <= 10:
        badge_color = "#FF9800"
        badge_text = f"{citation_count} citations"
    else:
        badge_color = "#f44336"
        badge_text = f"{citation_count} citations"
    
    # Химические маркеры
    chemical_markers = []
    if work.get('has_chemical_formula'):
        chemical_markers.append("🧪")
    if work.get('has_organic_pattern'):
        chemical_markers.append("⚗️")
    
    oa_badge = '🔓' if work.get('is_oa') else '🔒'
    doi_url = work.get('doi_url', '')
    
    # Используем безопасный HTML
    title = safe_html(work.get('title', 'No title'))
    
    authors_list = work.get('authors', [])
    authors_safe = []
    for author in authors_list[:2]:
        authors_safe.append(safe_html(author))
    
    authors = ', '.join(authors_safe)
    if len(authors_list) > 2:
        authors += ' et al.'
    
    journal_name = safe_html(work.get('journal_name', ''))[:30]
    year = work.get('publication_year', '')
    relevance_score = work.get('relevance_score', 0)
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <span style="font-weight: 600; color: #667eea; margin-right: 8px;">#{index}</span>
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                    {badge_text}
                </span>
                <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 5px;">
                    Score: {relevance_score}
                </span>
                {' '.join(chemical_markers)}
            </div>
            <span style="color: #666; font-size: 0.8rem;">{year}</span>
        </div>
        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 5px; line-height: 1.3;">{title}</div>
        <div style="color: #555; font-size: 0.85rem; margin-bottom: 5px;">👤 {authors}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
            <span>{oa_badge} {journal_name}</span>
            <a href="{doi_url}" target="_blank" style="color: #2196F3; text-decoration: none; font-size: 0.85rem;">
                🔗 View Article
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_topic_selection_ui():
    """Интерфейс выбора темы"""
    st.markdown("<h4>🎯 Select Research Topic</h4>", unsafe_allow_html=True)
    
    topics = st.session_state.topic_counter.most_common()
    
    # Показываем первые 8 тем в компактном виде
    cols = st.columns(2)
    for idx, (topic, count) in enumerate(topics[:8]):
        with cols[idx % 2]:
            is_selected = st.session_state.get('selected_topic') == topic
            st.markdown(f"""
            <div class="topic-card {'selected' if is_selected else ''}" 
                 onclick="this.style.background='#667eea10';">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; font-size: 0.9rem;">{topic[:70]}{'...' if len(topic) > 70 else ''}</div>
                    <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                        {count} papers
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select", key=f"select_{idx}", 
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"):
                st.session_state.selected_topic = topic
                
                # Находим ID темы из данных
                for work in st.session_state.works_data:
                    if work.get('primary_topic') == topic:
                        topic_id = work.get('topic_id')
                        if topic_id:
                            st.session_state.selected_topic_id = topic_id
                            break
                
                st.rerun()
    
    # Кнопка продолжения
    if 'selected_topic' in st.session_state:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚙️ Configure Filters", type="primary", use_container_width=True, key="configure_filters"):
                st.session_state.current_step = 4
                st.rerun()

# ============================================================================
# НОВЫЙ ШАГ ДЛЯ ФИЛЬТРАЦИИ
# ============================================================================

def step_filters():
    """Шаг 4: Настройка фильтров"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">⚙️ Step 4: Configure Filters</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Set publication years and citation ranges for analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'selected_topic_id' not in st.session_state:
        st.error("❌ Topic not selected. Please go back to Step 3.")
        return
    
    topic_id = st.session_state.selected_topic_id
    topic_name = st.session_state.get('selected_topic', 'Selected Topic')
    
    # Получаем общее количество работ по теме
    with st.spinner("Getting topic statistics..."):
        total_works = get_topic_total_works_count(topic_id)
    
    if total_works == 0:
        st.error(f"❌ No works found for topic: {topic_name}")
        return
    
    st.markdown(f"""
    <div class="filter-stats">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>📊 Topic Statistics</strong><br>
                <span style="font-size: 0.9rem; color: #666;">{topic_name}</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{total_works:,}</span><br>
                <span style="font-size: 0.8rem; color: #666;">total works</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Инициализация состояния фильтров
    if 'selected_years' not in st.session_state:
        current_year = datetime.now().year
        st.session_state.selected_years = [current_year - 2, current_year - 1, current_year]
    
    if 'selected_citations' not in st.session_state:
        st.session_state.selected_citations = [(0, 0), (1, 1), (2, 2)]
    
    # Секция фильтра по годам
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
    st.markdown("<div class='filter-header'>📅 Publication Years</div>", unsafe_allow_html=True)
    
    current_year = datetime.now().year
    years_options = list(range(current_year - 2, current_year + 1))  # Только последние 3 года
    
    # Отображаем чекбоксы для годов в 3 колонки
    st.markdown("<div class='year-checkbox-container'>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    selected_years = []
    
    for idx, year in enumerate(years_options):
        col_idx = idx % 3
        with cols[col_idx]:
            is_selected = year in st.session_state.selected_years
            if st.checkbox(f"{year}", value=is_selected, key=f"year_{year}"):
                selected_years.append(year)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Если ничего не выбрано, используем значения по умолчанию
    if not selected_years:
        selected_years = years_options
        # Обновляем чекбоксы
        for year in years_options:
            st.session_state[f"year_{year}"] = True
    
    st.session_state.selected_years = selected_years
    st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 10px;'>Selected years: {', '.join(map(str, selected_years))}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Секция фильтра по цитированиям
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
    st.markdown("<div class='filter-header'>📈 Citation Counts</div>", unsafe_allow_html=True)
    
    citation_options = list(range(0, 11))  # 0-10
    
    # Первый ряд: 0-5
    st.markdown("<div class='citation-checkbox-row'>", unsafe_allow_html=True)
    cols1 = st.columns(6)
    selected_citation_values = []
    
    for i in range(6):  # 0-5
        with cols1[i]:
            citation_value = i
            is_selected = any(start <= citation_value <= end for start, end in st.session_state.selected_citations)
            if st.checkbox(f"{citation_value}", value=is_selected, key=f"citation_{citation_value}"):
                selected_citation_values.append(citation_value)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Второй ряд: 6-10 + "Select all"
    st.markdown("<div class='citation-checkbox-row'>", unsafe_allow_html=True)
    cols2 = st.columns(6)
    
    for i in range(5):  # 6-10
        with cols2[i]:
            citation_value = i + 6
            is_selected = any(start <= citation_value <= end for start, end in st.session_state.selected_citations)
            if st.checkbox(f"{citation_value}", value=is_selected, key=f"citation_{citation_value}"):
                selected_citation_values.append(citation_value)
    
    # Колонка для "Select all"
    with cols2[5]:
        # Используем ключ для select all и обрабатываем логику позже
        select_all = st.checkbox("Select all", key="citation_all")
        
        # Если выбран select_all, то выбираем все значения
        if select_all:
            selected_citation_values = list(range(0, 11))
        # Если select_all снят и выбраны все значения, снимаем select_all
        elif len(selected_citation_values) == 11:
            # Обновляем состояние чекбокса через callback
            st.session_state.citation_all = True
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Преобразуем выбранные значения в диапазоны
    if selected_citation_values:
        selected_citation_values.sort()
        citation_ranges = []
        start = selected_citation_values[0]
        end = selected_citation_values[0]
        
        for i in range(1, len(selected_citation_values)):
            if selected_citation_values[i] == end + 1:
                end = selected_citation_values[i]
            else:
                citation_ranges.append((start, end))
                start = selected_citation_values[i]
                end = selected_citation_values[i]
        
        citation_ranges.append((start, end))
        st.session_state.selected_citations = citation_ranges
    
    # Если ничего не выбрано, используем значения по умолчанию
    if not selected_citation_values:
        st.session_state.selected_citations = [(0, 2)]
        # Обновляем чекбоксы
        for i in range(3):
            st.session_state[f"citation_{i}"] = True
    
    st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 10px;'>Selected citation ranges: {format_citation_ranges(st.session_state.selected_citations)}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Кнопка запуска анализа
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Start Filtered Analysis", type="primary", use_container_width=True, key="start_filtered_analysis"):
            # Сбрасываем кэш предыдущих результатов
            if 'filtered_works' in st.session_state:
                del st.session_state['filtered_works']
            if 'filtered_total_count' in st.session_state:
                del st.session_state['filtered_total_count']
            if 'filter_stats' in st.session_state:
                del st.session_state['filter_stats']
            if 'top_keywords' in st.session_state:
                del st.session_state['top_keywords']
            
            # Сохраняем статистику фильтров для отображения
            st.session_state.filter_stats = {
                'total_works': total_works,
                'selected_years': st.session_state.selected_years,
                'selected_citations': st.session_state.selected_citations
            }
            
            st.session_state.current_step = 5
            st.rerun()

# ============================================================================
# ШАГИ МАСТЕР-ПРОЦЕССА
# ============================================================================

def step_data_input():
    """Шаг 1: Ввод данных (компактный)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">📥 Step 1: Input Research DOIs</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Enter DOI identifiers to analyze topics and keywords.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Компактный ввод
    doi_input = st.text_area(
        "**DOI Input** (one per line or comma-separated):",
        height=150,
        placeholder="Examples:\n10.1038/nmat1849\nhttps://doi.org/10.1038/nmat1849\nGeim, A., Novoselov, K. The rise of graphene. Nature Mater 6, 183–191 (2007). https://doi.org/10.1038/nmat1849",
        help="Enter up to 300 DOI identifiers"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if doi_input:
                dois = parse_doi_input(doi_input)
                if dois:
                    st.session_state.dois = dois
                    st.session_state.current_step = 2
                    st.rerun()
                else:
                    st.error("❌ No valid DOI identifiers found.")
            else:
                st.error("❌ Please enter at least one DOI")
    
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()

def step_analysis():
    """Шаг 2: Анализ (компактный)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">🔍 Step 2: Analysis in Progress</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Fetching data from OpenAlex...</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'dois' not in st.session_state:
        st.error("❌ No data to analyze. Please go back to Step 1.")
        return
    
    dois = st.session_state.dois
    
    # Компактные метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card_compact("DOIs", len(dois), "🔢")
    with col2:
        create_metric_card_compact("Est. Time", f"{len(dois)//10}s", "⏱️")
    with col3:
        create_metric_card_compact("API Rate", "8/sec", "⚡")
    
    # Загрузка данных
    with st.spinner("Fetching data..."):
        results, successful, failed = fetch_works_by_dois_sync(dois)
    
    # Обработка результатов
    works_data = []
    topic_counter = Counter()
    titles = []
    
    for result in results:
        if result.get('success') and result.get('data'):
            work = result['data']
            enriched = enrich_work_data(work)
            
            if enriched.get('primary_topic'):
                topic_counter[enriched['primary_topic']] += 1
            
            works_data.append(enriched)
            titles.append(enriched.get('title', ''))
    
    # Анализ ключевых слов
    keyword_counter = analyze_keywords_parallel(titles)
    
    # Сохранение результатов
    st.session_state.works_data = works_data
    st.session_state.topic_counter = topic_counter
    st.session_state.keyword_counter = keyword_counter
    st.session_state.successful = successful
    st.session_state.failed = failed
    
    # Результаты анализа
    st.markdown(f"""
    <div class="info-message">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>✅ Analysis Complete!</strong><br>
                Successfully processed {successful} papers
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Статистика
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card_compact("Successful", successful, "✅")
    with col2:
        create_metric_card_compact("Failed", failed, "❌")
    with col3:
        create_metric_card_compact("Topics", len(topic_counter), "🏷️")
    
    # Кнопка продолжения
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Continue to Topic Selection", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

def step_topic_selection():
    """Шаг 3: Выбор темы (компактный)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">🎯 Step 3: Select Research Topic</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Choose a topic for deep analysis of fresh papers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.works_data:
        st.error("❌ No data available. Please start from Step 1.")
        return
    
    create_topic_selection_ui()

def step_results():
    """Шаг 5: Результаты (обновленный с фильтрацией на стороне API)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">📊 Step 5: Analysis Results</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Fresh papers in your research area with server-side filtering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'selected_topic_id' not in st.session_state:
        st.error("❌ Topic not selected. Please go back to Step 3.")
        return
    
    # Получаем фильтры
    selected_years = st.session_state.get('selected_years', [])
    if not selected_years:
        current_year = datetime.now().year
        selected_years = [current_year - 2, current_year - 1, current_year]
        st.session_state.selected_years = selected_years
    
    selected_citations = st.session_state.get('selected_citations', [])
    if not selected_citations:
        selected_citations = [(0, 2)]
        st.session_state.selected_citations = selected_citations
    
    # Показываем статистику фильтров
    if 'filter_stats' in st.session_state:
        stats = st.session_state.filter_stats
        st.markdown(f"""
        <div class="filter-stats">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>📊 Filter Summary</strong><br>
                    <span style="font-size: 0.9rem; color: #666;">
                        Years: {', '.join(map(str, stats['selected_years']))} | 
                        Citations: {format_citation_ranges(stats['selected_citations'])}
                    </span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 1.2rem; font-weight: 700; color: #667eea;">{stats['total_works']:,}</span><br>
                    <span style="font-size: 0.8rem; color: #666;">total works in topic</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Анализ работ по теме с фильтрацией на стороне API
    if 'filtered_works' not in st.session_state:
        with st.spinner("Searching for fresh papers with server-side filtering..."):
            # Получаем топ-10 ключевых слов
            top_keywords = [kw for kw, _ in st.session_state.keyword_counter.most_common(10)]
            
            # Сохраняем ключевые слова в сессии
            st.session_state.top_keywords = top_keywords
            
            # Выполняем улучшенный анализ с фильтрацией на стороне API
            relevant_works, filtered_total_count = analyze_filtered_works_for_topic(
                topic_id=st.session_state.selected_topic_id,
                keywords=top_keywords,
                selected_years=selected_years,
                selected_citations=selected_citations,
                max_works=5000,  # Увеличили лимит для полноты
                top_n=100
            )
        
        st.session_state.filtered_works = relevant_works
        st.session_state.filtered_total_count = filtered_total_count
    else:
        relevant_works = st.session_state.filtered_works
        filtered_total_count = st.session_state.filtered_total_count
    
    # Статистика
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        create_metric_card_compact("Filtered Works", f"{filtered_total_count:,}", "📄")
    with col2:
        create_metric_card_compact("Papers Found", len(relevant_works), "🎯")
    with col3:
        if relevant_works:
            avg_citations = np.mean([w.get('cited_by_count', 0) for w in relevant_works])
            create_metric_card_compact("Avg Citations", f"{avg_citations:.1f}", "📈")
        else:
            create_metric_card_compact("Avg Citations", "0", "📈")
    with col4:
        oa_count = sum(1 for w in relevant_works if w.get('is_oa'))
        create_metric_card_compact("Open Access", oa_count, "🔓")
    with col5:
        current_year = datetime.now().year
        recent_count = sum(1 for w in relevant_works if w.get('publication_year', 0) >= current_year - 2)
        create_metric_card_compact("Recent (≤2y)", recent_count, "🕒")

    if relevant_works:
        chem_count = sum(1 for w in relevant_works if w.get('has_chemical_formula') or w.get('has_organic_pattern'))
        create_metric_card_compact("Chemical", f"{chem_count}", "🧪")
        
    if not relevant_works:
        # Добавляем отладочную информацию
        st.markdown(f"""
        <div class="warning-message">
            <strong>⚠️ No papers match your filters</strong><br>
            <strong>Debug info:</strong><br>
            - Topic ID: {st.session_state.get('selected_topic_id', 'Not set')}<br>
            - Years filter: {selected_years}<br>
            - Citation ranges: {format_citation_ranges(selected_citations)}<br>
            - Total works after filters: {filtered_total_count}<br>
            <br>
            This might happen when:<br>
            1. Current year selected with high citation threshold (papers might not have enough citations yet)<br>
            2. Very specific citation range selected<br>
            3. Topic has limited publications in selected years<br>
            <br>
            Try adjusting your filters in Step 4.
        </div>
        """, unsafe_allow_html=True)
        
        # Для отладки также покажем логи
        logger.warning(f"No relevant works found for topic {st.session_state.get('selected_topic_id')}")
        logger.warning(f"Filters: years={selected_years}, citation_ranges={selected_citations}")
        logger.warning(f"Total works after filters: {filtered_total_count}")
    else:
        # Результаты в виде карточек
        st.markdown("<h4>🎯 Recommended Papers:</h4>", unsafe_allow_html=True)
        
        for idx, work in enumerate(relevant_works[:10], 1):
            create_result_card_compact(work, idx)
        
        # Таблица для детального просмотра
        st.markdown("<h4>📋 Detailed View:</h4>", unsafe_allow_html=True)
        
        display_data = []
        for i, work in enumerate(relevant_works, 1):
            doi_url = work.get('doi_url', '')
            title = work.get('title', '')
            
            display_data.append({
                '#': i,
                'Title': title[:60] + '...' if len(title) > 60 else title,
                'Citations': work.get('cited_by_count', 0),
                'Relevance': work.get('relevance_score', 0),
                'Year': work.get('publication_year', ''),
                'Journal': work.get('journal_name', '')[:20],
                'DOI': doi_url if doi_url else 'N/A',
                'OA': '✅' if work.get('is_oa') else '❌',
                'Authors': ', '.join(work.get('authors', [])[:2])
            })
        
        df = pd.DataFrame(display_data)
        
        # Используем column_config без LinkColumn для чистых URL
        st.dataframe(
            df,
            use_container_width=True,
            height=300,
            column_config={
                "DOI": st.column_config.TextColumn(
                    "DOI",
                    help="Click to copy or open in browser",
                    width="medium"
                ),
                "Relevance": st.column_config.ProgressColumn(
                    "Relevance",
                    help="Relevance score (higher is better)",
                    format="%d",
                    min_value=1,
                    max_value=10
                )
            }
        )
        
        # Экспорт в разные форматы
        st.markdown("<h4>📥 Export Results:</h4>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv = generate_csv(relevant_works)
            st.download_button(
                label="📊 CSV",
                data=csv,
                file_name=f"under_cited_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            excel_data = generate_excel(relevant_works)
            st.download_button(
                label="📈 Excel",
                data=excel_data,
                file_name=f"under_cited_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            txt_data = generate_txt(relevant_works, st.session_state.get('selected_topic', 'Results'))
            st.download_button(
                label="📝 TXT",
                data=txt_data,
                file_name=f"under_cited_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col4:
            pdf_data = generate_pdf(relevant_works[:50], st.session_state.get('selected_topic', 'Results'))
            st.download_button(
                label="📄 PDF",
                data=pdf_data,
                file_name=f"under_cited_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        # Кнопка нового анализа
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                # Очищаем все данные сессии
                keys_to_clear = [
                    'filtered_works', 'filtered_total_count', 'filter_stats',
                    'selected_topic', 'selected_topic_id', 'selected_years', 
                    'selected_citations', 'top_keywords', 'works_data', 
                    'topic_counter', 'keyword_counter', 'successful', 
                    'failed', 'dois'
                ]
                
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Сбрасываем все чекбоксы
                current_year = datetime.now().year
                for year in range(current_year - 2, current_year + 1):
                    if f"year_{year}" in st.session_state:
                        del st.session_state[f"year_{year}"]
                
                for i in range(11):
                    if f"citation_{i}" in st.session_state:
                        del st.session_state[f"citation_{i}"]
                
                if "citation_all" in st.session_state:
                    del st.session_state["citation_all"]
                
                st.session_state.current_step = 1
                st.rerun()

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция приложения"""
    
    # Инициализация состояния
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Заголовок (компактный)
    st.markdown("""
    <h1 class="main-header">🔬 CTA Article Recommender Pro</h1>
    <p style="font-size: 1rem; color: #666; margin-bottom: 1.5rem;">
    Discover fresh papers using AI-powered analysis with server-side filtering
    </p>
    """, unsafe_allow_html=True)
    
    # Прогресс бар (обновлен для 5 шагов)
    create_progress_bar(st.session_state.current_step, 5)
    
    # Очистка старого кэша
    clear_old_cache()
    
    # Отображение текущего шага
    if st.session_state.current_step == 1:
        step_data_input()
    elif st.session_state.current_step == 2:
        step_analysis()
    elif st.session_state.current_step == 3:
        step_topic_selection()
    elif st.session_state.current_step == 4:
        step_filters()
    elif st.session_state.current_step == 5:
        step_results()
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem; margin-top: 1rem;">
        <p>© CTA, https://chimicatechnoacta.ru / developed by daM©</p>
        <p style="font-size: 0.7rem; color: #aaa;">v2.0 with server-side filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()







