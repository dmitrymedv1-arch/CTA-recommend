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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import xlsxwriter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки приложения
st.set_page_config(
    page_title="CTA Research Explorer Pro",
    page_icon="🔬",
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
    
    cursor.execute('DELETE FROM works_cache WHERE expires_at <= ?', (datetime.now(),))
    cursor.execute('DELETE FROM topic_works_cache WHERE expires_at <= ?', (datetime.now(),))
    cursor.execute('DELETE FROM topics_cache WHERE expires_at <= ?', (datetime.now(),))
    
    conn.commit()

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
    if not text or not text.strip():
        return []
    
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    dois = re.findall(doi_pattern, text, re.IGNORECASE)
    
    cleaned_dois = []
    for doi in dois:
        doi = doi.strip()
        if doi:
            if doi.startswith('https://doi.org/'):
                doi = doi[16:]
            elif doi.startswith('http://doi.org/'):
                doi = doi[15:]
            elif doi.startswith('doi.org/'):
                doi = doi[8:]
            cleaned_dois.append(doi)
    
    return list(set(cleaned_dois))[:300]

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
        enriched['venue_name'] = source.get('display_name', '') if source else ''
        enriched['venue_type'] = (source or {}).get('type', '')
    else:
        enriched['venue_name'] = ''
        enriched['venue_type'] = ''
    
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
    """Генерация PDF файла"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Кастомные стили
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    story = []
    
    # Заголовок
    story.append(Paragraph("CTA Research Explorer Pro", title_style))
    story.append(Paragraph(f"Topic: {topic_name}", header_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Paragraph("© CTA, https://chimicatechnoacta.ru / developed by daM©", normal_style))
    story.append(Spacer(1, 20))
    
    # Таблица
    table_data = [['#', 'Title', 'Authors', 'Year', 'Citations', 'Relevance', 'DOI']]
    
    for i, work in enumerate(data[:50], 1):  # Ограничиваем 50 записями для PDF
        authors = ', '.join(work.get('authors', [])[:2])
        if len(work.get('authors', [])) > 2:
            authors += ' et al.'
        
        table_data.append([
            str(i),
            work.get('title', '')[:80],
            authors[:50],
            str(work.get('publication_year', '')),
            str(work.get('cited_by_count', 0)),
            str(work.get('relevance_score', 0)),
            work.get('doi', '')[:40]
        ])
    
    table = Table(table_data, colWidths=[30, 200, 100, 40, 50, 50, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(table)
    doc.build(story)
    
    return buffer.getvalue()

def generate_txt(data: List[dict], topic_name: str) -> str:
    """Генерация TXT файла"""
    output = []
    output.append("=" * 80)
    output.append("CTA Research Explorer Pro")
    output.append(f"Topic: {topic_name}")
    output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output.append("© CTA, https://chimicatechnoacta.ru / developed by daM©")
    output.append("=" * 80)
    output.append("")
    
    for i, work in enumerate(data, 1):
        output.append(f"{i}. {work.get('title', 'No title')}")
        output.append(f"   Authors: {', '.join(work.get('authors', ['Unknown']))[:100]}")
        output.append(f"   Year: {work.get('publication_year', 'N/A')}")
        output.append(f"   Citations: {work.get('cited_by_count', 0)}")
        output.append(f"   Relevance Score: {work.get('relevance_score', 0)}")
        output.append(f"   DOI: {work.get('doi', 'N/A')}")
        output.append(f"   Journal: {work.get('venue_name', 'N/A')}")
        output.append(f"   Open Access: {'Yes' if work.get('is_oa') else 'No'}")
        if work.get('matched_keywords'):
            output.append(f"   Matched Keywords: {', '.join(work.get('matched_keywords', []))}")
        output.append("-" * 80)
    
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
            <span>{oa_badge} {work.get('venue_name', '')[:30]}</span>
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
                    <div style="font-weight: 600; font-size: 0.9rem;">{topic[:25]}{'...' if len(topic) > 25 else ''}</div>
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
            # Фильтр по годам
            current_year = datetime.now().year
            years = list(range(current_year - 10, current_year + 1))
            selected_years = st.multiselect(
                "Publication Years:",
                options=years,
                default=[current_year - 2, current_year - 1, current_year],
                help="Select years to include in analysis"
            )
            
            if not selected_years:
                st.warning("Please select at least one year")
                selected_years = [current_year - 2, current_year - 1, current_year]
            
            st.session_state.selected_years = selected_years
        
        with col2:
            # Фильтр по цитированиям (диапазоны)
            citation_options = [
                ("Exactly 0", [(0, 0)]),
                ("0-2 citations", [(0, 2)]),
                ("0-5 citations", [(0, 5)]),
                ("2-5 citations", [(2, 5)]),
                ("Exactly 3", [(3, 3)]),
                ("3-5 citations", [(3, 5)]),
                ("5-10 citations", [(5, 10)]),
                ("0-2 OR 4", [(0, 2), (4, 4)]),  # Пример сложного диапазона
                ("Custom...", "custom")
            ]
            
            selected_option = st.selectbox(
                "Citation Ranges:",
                options=[opt[0] for opt in citation_options],
                index=2,
                help="Select citation ranges to include"
            )
            
            if selected_option == "Custom...":
                st.text_input("Enter ranges (e.g., '0-2,4,5-10'):", key="custom_ranges")
            else:
                selected_ranges = next(opt[1] for opt in citation_options if opt[0] == selected_option)
                st.session_state.selected_ranges = selected_ranges
        
        # Кнопка запуска анализа
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Start Deep Analysis", type="primary", use_container_width=True):
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
        placeholder="Example:\n10.1016/j.jpowsour.2020.228660\n10.1038/s41560-020-00734-0\n10.1021/acsenergylett.1c00123",
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
        <p style="margin: 5px 0; font-size: 0.9rem;">Choose a topic for deep analysis of under-cited papers.</p>
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
        <p style="margin: 5px 0; font-size: 0.9rem;">Under-cited papers in your research area.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'selected_topic_id' not in st.session_state:
        st.error("❌ Topic not selected. Please go back to Step 3.")
        return
    
    # Получаем фильтры
    selected_years = st.session_state.get('selected_years', [datetime.now().year - 2, datetime.now().year - 1, datetime.now().year])
    selected_ranges = st.session_state.get('selected_ranges', [(0, 10)])
    
    # Анализ работ по теме
    if 'relevant_works' not in st.session_state:
        with st.spinner("Searching for under-cited papers..."):
            relevant_works = analyze_works_for_topic(
                st.session_state.selected_topic_id,
                [kw for kw, _ in st.session_state.keyword_counter.most_common(10)],
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
        avg_citations = np.mean([w.get('cited_by_count', 0) for w in relevant_works]) if relevant_works else 0
        create_metric_card_compact("Avg Citations", f"{avg_citations:.1f}", "📈")
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
        Citation ranges: {', '.join([f'{min_cit}-{max_cit}' for min_cit, max_cit in selected_ranges])}
    </div>
    """, unsafe_allow_html=True)
    
    if not relevant_works:
        st.warning("""
        <div class="warning-message">
            <strong>⚠️ No papers match your filters</strong><br>
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
                'Journal': work.get('venue_name', '')[:20],
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
                          'selected_years', 'selected_ranges', 'top_keywords']:
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
    st.markdown("""
    <h1 class="main-header">🔬 CTA Research Explorer Pro</h1>
    <p style="font-size: 1rem; color: #666; margin-bottom: 1.5rem;">
    Discover under-cited papers using AI-powered analysis.
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
        <p>Powered by OpenAlex API • Polite pool enabled • Local caching</p>
        <p>© CTA, https://chimicatechnoacta.ru / developed by daM©</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

