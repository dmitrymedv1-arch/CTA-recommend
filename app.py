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
import xlsxwriter

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

def analyze_works_for_topic(
    topic_id: str,
    keywords: List[str],
    max_citations: int = 10,
    max_works: int = 2000,
    top_n: int = 100,
    year_filter: List[int] = None,
    citation_ranges: List[Tuple[int, int]] = None
) -> List[dict]:
    
    with st.spinner(f"Loading up to {max_works} works..."):
        works = fetch_works_by_topic_sync(topic_id, max_works)
    
    if not works:
        return []
    
    current_year = datetime.now().year
    if year_filter is None:
        year_filter = [current_year - 2, current_year - 1, current_year]
    
    if citation_ranges is None:
        citation_ranges = [(0, 10)]
    
    with st.spinner(f"Analyzing {len(works)} works..."):
        analyzed = []
        
        for work in works:
            cited_by_count = work.get('cited_by_count', 0)
            publication_year = work.get('publication_year', 0)
            
            # Фильтр по годам
            if publication_year not in year_filter:
                continue
            
            # Фильтр по цитированиям (диапазоны)
            in_range = False
            for min_cit, max_cit in citation_ranges:
                if min_cit <= cited_by_count <= max_cit:
                    in_range = True
                    break
            
            if not in_range:
                continue
            
            title = work.get('title', '')
            abstract = work.get('abstract', '')
            
            if title:
                title_lower = title.lower()
                abstract_lower = abstract.lower() if abstract else ''
                
                score = 0
                matched = []
                
                for keyword in keywords:
                    kw_lower = keyword.lower()
                    if kw_lower in title_lower:
                        score += 3
                        matched.append(keyword)
                    elif abstract and kw_lower in abstract_lower:
                        score += 1
                        matched.append(f"{keyword}*")
                
                if score > 0:
                    enriched = enrich_work_data(work)
                    enriched.update({
                        'relevance_score': score,
                        'matched_keywords': matched,
                        'analysis_time': datetime.now().isoformat()
                    })
                    analyzed.append(enriched)
        
        analyzed.sort(key=lambda x: x['relevance_score'], reverse=True)
        return analyzed[:top_n]

# ============================================================================
# ФУНКЦИИ ЭКСПОРТА
# ============================================================================

def generate_csv(data: List[dict]) -> str:
    """Генерация CSV файла"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

def generate_excel(data: List[dict]) -> bytes:
    """Генерация Excel файла"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df = pd.DataFrame(data)
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
    
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("CTA Article Recommender Pro", title_style))
    story.append(Paragraph("Fresh Papers Analysis Report", subtitle_style))
    story.append(Spacer(1, 1*cm))
    
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
                    story.append(Paragraph("Publications by Year (Last 10 years)", subtitle_style))
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
        <span class="{'active' if current_step >= 4 else ''}">📊 Results</span>
    </div>
    """, unsafe_allow_html=True)

def create_back_button():
    """Создает кнопку возврата назад"""
    if st.session_state.current_step > 1:
        if st.button("← Back", key="back_button", use_container_width=False):
            # При возврате на шаг 3, сбрасываем кэш результатов, чтобы фильтры применились заново
            if st.session_state.current_step == 4:
                if 'relevant_works' in st.session_state:
                    del st.session_state['relevant_works']
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
    """Создает компактную карточку результата"""
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
    
    oa_badge = '🔓' if work.get('is_oa') else '🔒'
    doi_url = work.get('doi_url', '')
    title = work.get('title', 'No title')
    authors = ', '.join(work.get('authors', [])[:2])
    if len(work.get('authors', [])) > 2:
        authors += ' et al.'
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <span style="font-weight: 600; color: #667eea; margin-right: 8px;">#{index}</span>
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                    {badge_text}
                </span>
                <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 5px;">
                    Score: {work.get('relevance_score', 0)}
                </span>
            </div>
            <span style="color: #666; font-size: 0.8rem;">{work.get('publication_year', '')}</span>
        </div>
        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 5px; line-height: 1.3;">{title}</div>
        <div style="color: #555; font-size: 0.85rem; margin-bottom: 5px;">👤 {authors}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
            <span>{oa_badge} {work.get('journal_name', '')[:30]}</span>
            <a href="{doi_url}" target="_blank" style="color: #2196F3; text-decoration: none; font-size: 0.85rem;">
                🔗 View Article
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_topic_selection_ui():
    """Интерфейс выбора темы с фильтрами"""
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
    
    # Фильтры для анализа (появляются после выбора темы)
    if 'selected_topic' in st.session_state:
        st.markdown("---")
        st.markdown("<h4>⚙️ Analysis Filters</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Фильтр по годам - ТОЛЬКО последние 3 года
            current_year = datetime.now().year
            years = list(range(current_year - 2, current_year + 1))  # Только 3 последних года
            
            selected_years = st.multiselect(
                "Publication Years:",
                options=years,
                default=[current_year - 2, current_year - 1, current_year],
                help="Select publication years (last 3 years only)"
            )
            
            if not selected_years:
                st.warning("Please select at least one year")
                selected_years = [current_year - 2, current_year - 1, current_year]
            
            st.session_state.selected_years = selected_years
        
        with col2:
            # Фильтр по цитированиям (обновленные опции)
            citation_options = [
                ("0 citations", "0"),
                ("1 citation", "1"),
                ("2 citations", "2"),
                ("3 citations", "3"),
                ("4 citations", "4"),
                ("5 citations", "5"),
                ("6 citations", "6"),
                ("7 citations", "7"),
                ("8 citations", "8"),
                ("9 citations", "9"),
                ("10 citations", "10"),
                ("0-2 citations", "0-2"),
                ("0-3 citations", "0-3"),
                ("0-5 citations", "0-5"),
                ("1-3 citations", "1-3"),
                ("1-5 citations", "1-5"),
                ("2-5 citations", "2-5"),
                ("3-5 citations", "3-5"),
                ("5-10 citations", "5-10"),
                ("0-1,3-4 (multiple ranges)", "0-1,3-4"),
                ("Custom...", "custom")
            ]
            
            selected_option = st.selectbox(
                "Citation Ranges:",
                options=[opt[0] for opt in citation_options],
                index=12,
                help="Select citation ranges (0-10 only)"
            )
            
            # Определяем выбранные диапазоны
            if selected_option == "Custom...":
                custom_input = st.text_input(
                    "Enter custom ranges (e.g., '0-2,4,5-7'):",
                    value="0-2",
                    help="Enter comma-separated values or ranges (0-10 only)"
                )
                if custom_input:
                    citation_ranges = parse_citation_ranges(custom_input)
                    # Проверяем что все значения в пределах 0-10
                    valid_ranges = []
                    for start, end in citation_ranges:
                        if 0 <= start <= 10 and 0 <= end <= 10:
                            valid_ranges.append((start, end))
                        else:
                            st.warning(f"Range {start}-{end} is outside 0-10. It will be ignored.")
                    
                    if valid_ranges:
                        st.session_state.selected_ranges = valid_ranges
                        st.info(f"Selected ranges: {format_citation_ranges(valid_ranges)}")
                    else:
                        st.session_state.selected_ranges = [(0, 2)]
                        st.warning("Using default range: 0-2")
                else:
                    st.session_state.selected_ranges = [(0, 2)]
            else:
                # Получаем строку диапазона из выбранной опции
                range_str = next(opt[1] for opt in citation_options if opt[0] == selected_option)
                citation_ranges = parse_citation_ranges(range_str)
                st.session_state.selected_ranges = citation_ranges
                st.info(f"Selected: {selected_option}")
        
        # Кнопка запуска анализа
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Start Deep Analysis", type="primary", use_container_width=True, key="start_analysis"):
                # Сбрасываем кэш предыдущих результатов
                if 'relevant_works' in st.session_state:
                    del st.session_state['relevant_works']
                if 'top_keywords' in st.session_state:
                    del st.session_state['top_keywords']
                
                st.session_state.current_step = 4
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
    """Шаг 4: Результаты (компактный)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">📊 Step 4: Analysis Results</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Fresh papers in your research area.</p>
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
    
    selected_ranges = st.session_state.get('selected_ranges', [])
    if not selected_ranges:
        selected_ranges = [(0, 2)]
        st.session_state.selected_ranges = selected_ranges
    
    # Анализ работ по теме (только если еще не анализировали или фильтры изменились)
    if 'relevant_works' not in st.session_state:
        with st.spinner("Searching for fresh papers..."):
            # Получаем топ-10 ключевых слов
            top_keywords = [kw for kw, _ in st.session_state.keyword_counter.most_common(10)]
            
            # Сохраняем ключевые слова в сессии
            st.session_state.top_keywords = top_keywords
            
            # Выполняем анализ
            relevant_works = analyze_works_for_topic(
                st.session_state.selected_topic_id,
                top_keywords,
                max_citations=10,
                max_works=2000,
                top_n=100,
                year_filter=selected_years,
                citation_ranges=selected_ranges
            )
        
        st.session_state.relevant_works = relevant_works
    else:
        relevant_works = st.session_state.relevant_works
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card_compact("Papers Found", len(relevant_works), "📄")
    with col2:
        if relevant_works:
            avg_citations = np.mean([w.get('cited_by_count', 0) for w in relevant_works])
            create_metric_card_compact("Avg Citations", f"{avg_citations:.1f}", "📈")
        else:
            create_metric_card_compact("Avg Citations", "0", "📈")
    with col3:
        oa_count = sum(1 for w in relevant_works if w.get('is_oa'))
        create_metric_card_compact("Open Access", oa_count, "🔓")
    with col4:
        current_year = datetime.now().year
        recent_count = sum(1 for w in relevant_works if w.get('publication_year', 0) >= current_year - 2)
        create_metric_card_compact("Recent (≤2y)", recent_count, "🕒")
    
    # Показываем активные фильтры
    st.markdown(f"""
    <div style="margin: 10px 0; font-size: 0.85rem; color: #666;">
        <strong>Active filters:</strong> Years: {', '.join(map(str, selected_years))} | 
        Citation ranges: {format_citation_ranges(selected_ranges)}
    </div>
    """, unsafe_allow_html=True)
    
    if not relevant_works:
        st.warning("""
        <div class="warning-message">
            <strong>⚠️ No papers match your filters</strong><br>
            This might happen when:<br>
            1. Current year selected with high citation threshold (papers might not have enough citations yet)<br>
            2. Very specific citation range selected<br>
            3. Topic has limited publications in selected years<br>
            <br>
            Try adjusting your filters in Step 3.
        </div>
        """, unsafe_allow_html=True)
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
                'DOI': doi_url if doi_url else 'N/A',  # Исправлено: убираем markdown
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
                for key in ['relevant_works', 'selected_topic', 'selected_topic_id', 
                          'selected_years', 'selected_ranges', 'top_keywords',
                          'works_data', 'topic_counter', 'keyword_counter',
                          'successful', 'failed', 'dois']:
                    if key in st.session_state:
                        del st.session_state[key]
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
    st.markdown(f"""
    <h1 class="main-header">
        <img src="logo.jpg" style="height: 40px; width: auto; vertical-align: middle; margin-right: 15px;"/>
        CTA Article Recommender Pro
    </h1>
    <p style="font-size: 1rem; color: #666; margin-bottom: 1.5rem;">
    Discover fresh papers using AI-powered analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Прогресс бар
    create_progress_bar(st.session_state.current_step, 4)
    
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
        step_results()
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem; margin-top: 1rem;">
        <p>© CTA, https://chimicatechnoacta.ru / developed by daM©</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()











