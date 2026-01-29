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

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .step-card {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 15px;
        padding: 25px;
        border-left: 5px solid #667eea;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2196F3;
    }
    
    .doi-link {
        color: #2196F3;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .doi-link:hover {
        color: #0d47a1;
        text-decoration: underline;
    }
    
    .citation-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .low-citation { background: #4CAF50; color: white; }
    .medium-citation { background: #FF9800; color: white; }
    .high-citation { background: #f44336; color: white; }
    
    .progress-container {
        background: #f5f5f5;
        border-radius: 10px;
        height: 8px;
        margin: 30px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .filter-chip {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        margin: 3px;
        background: #e3f2fd;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #1565c0;
    }
    
    .success-message {
        background: linear-gradient(135deg, #4CAF5020 0%, #2E7D3220 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FF980020 0%, #EF6C0020 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #FF9800;
    }
    
    .info-message {
        background: linear-gradient(135deg, #2196F320 0%, #0D47A120 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2196F3;
    }
    
    .tooltip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        background: #e0e0e0;
        border-radius: 50%;
        margin-left: 5px;
        cursor: help;
        font-size: 0.7rem;
    }
    
    .stButton > button {
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .secondary-button {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
    }
    
    .dataframe {
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: left !important;
    }
    
    .dataframe tr:hover {
        background-color: #f5f5f5 !important;
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
# КЭШИРОВАНИЕ НА УРОВНЕ SQLite (остается без изменений)
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
# ASYNCIO + AIOHTTP (остается без изменений)
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
    
    async def fetch_works_by_topic_cursor(self, topic_id: str, max_results: int = 2000) -> List[dict]:
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
                status_text.text(f"Batch {i+1}/{len(batches)}: processing {len(batch)} DOI")
                
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
    async def fetch():
        async with OpenAlexAsyncClient() as client:
            return await client.fetch_works_by_topic_cursor(topic_id, max_results)
    
    return run_async(fetch())

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
        enriched['venue_type'] = source.get('type', '')
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
    top_n: int = 100
) -> List[dict]:
    
    with st.spinner(f"Loading up to {max_works} works..."):
        works = fetch_works_by_topic_sync(topic_id, max_works)
    
    if not works:
        return []
    
    with st.spinner(f"Analyzing {len(works)} works..."):
        analyzed = []
        
        for work in works:
            cited_by_count = work.get('cited_by_count', 0)
            
            if cited_by_count <= max_citations:
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
# КОМПОНЕНТЫ ИНТЕРФЕЙСА
# ============================================================================

def create_progress_bar(current_step: int, total_steps: int):
    """Создает прогресс1 бар мастер-процесса"""
    progress = current_step / total_steps
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress * 100}%"></div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #666; font-size: 0.9rem;">
        <span>📥 Data Input</span>
        <span>🔍 Analysis</span>
        <span>🎯 Results</span>
        <span>📊 Export</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h3>Step {current_step} of {total_steps}</h3>", unsafe_allow_html=True)

def create_metric_card(title: str, value, change: str = "", icon: str = "📊"):
    """Создает карточку с метрикой"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 1.5rem; margin-right: 10px;">{icon}</div>
            <div style="font-size: 0.9rem; color: #666;">{title}</div>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #333;">{value}</div>
        {f'<div style="font-size: 0.85rem; color: #4CAF50;">{change}</div>' if change else ''}
    </div>
    """, unsafe_allow_html=True)

def create_result_card(work: dict, index: int):
    """Создает карточку результата"""
    citation_count = work.get('cited_by_count', 0)
    
    if citation_count == 0:
        citation_badge = '<span class="citation-badge low-citation">0 citations</span>'
    elif citation_count <= 3:
        citation_badge = f'<span class="citation-badge low-citation">{citation_count} citation{"s" if citation_count > 1 else ""}</span>'
    elif citation_count <= 10:
        citation_badge = f'<span class="citation-badge medium-citation">{citation_count} citations</span>'
    else:
        citation_badge = f'<span class="citation-badge high-citation">{citation_count} citations</span>'
    
    oa_badge = '✅ Open Access' if work.get('is_oa') else '🔒 Closed Access'
    relevance_score = work.get('relevance_score', 0)
    
    authors = ', '.join(work.get('authors', [])[:3])
    if len(work.get('authors', [])) > 3:
        authors += f' and {len(work.get('authors', [])) - 3} more'
    
    doi_url = work.get('doi_url', '')
    title = work.get('title', 'No title')
    
    # Формируем HTML для ключевых слов
    keywords_html = ''
    if work.get('matched_keywords'):
        keywords = work.get('matched_keywords', [])[:5]
        keywords_html = '<div style="margin: 10px 0;">'
        for kw in keywords:
            keywords_html += f'<span style="background: #f0f4ff; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.8rem; display: inline-block;">{kw}</span>'
        keywords_html += '</div>'
    
    html_content = f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
            <div style="display: flex; align-items: center;">
                <span style="font-weight: 700; color: #667eea; margin-right: 10px;">#{index}</span>
                {citation_badge}
                <span style="background: #e3f2fd; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem;">
                    Score: {relevance_score}
                </span>
            </div>
            <span style="color: #666; font-size: 0.9rem;">{work.get('publication_year', '')} • {work.get('venue_name', '')[:30]}</span>
        </div>
        
        <h4 style="margin: 10px 0; line-height: 1.4;">{title}</h4>
        
        <div style="color: #555; margin: 10px 0; font-size: 0.95rem;">
            <span>👥 {authors if authors else 'Unknown authors'}</span>
        </div>
        
        {keywords_html}
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
            <div>
                {oa_badge}
            </div>
            <div>
                {'<a href="' + doi_url + '" target="_blank" class="doi-link">🔗 View Article</a>' if doi_url else '<span style="color: #999;">No DOI available</span>'}
            </div>
        </div>
    </div>
    """
    
    # Ключевое: используем st.markdown с unsafe_allow_html=True
    st.markdown(html_content, unsafe_allow_html=True)

def create_filters_ui() -> Dict:
    """Создание интерфейса фильтров"""
    with st.sidebar:
        st.markdown("<h3 style='color: #667eea;'>🎯 Filters</h3>", unsafe_allow_html=True)
        
        # Цитирования - диапазоны вместо ползунка
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>Citation Range</p>", unsafe_allow_html=True)
        
        citation_options = [
            ("Exactly 0", 0, 0),
            ("0-2 citations", 0, 2),
            ("0-5 citations", 0, 5),
            ("2-5 citations", 2, 5),
            ("Exactly 3 citations", 3, 3),
            ("3-5 citations", 3, 5),
            ("5-10 citations", 5, 10),
            ("10-20 citations", 10, 20)
        ]
        
        selected_option = st.selectbox(
            "Select citation range:",
            options=[opt[0] for opt in citation_options],
            index=2,
            label_visibility="collapsed"
        )
        
        # Находим выбранный диапазон
        selected_range = next((opt[1:]) for opt in citation_options if opt[0] == selected_option)
        min_citations, max_citations = selected_range
        
        st.markdown(f"""
        <div class="info-message" style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between;">
                <span>Selected range:</span>
                <span style="font-weight: 600;">{min_citations} - {max_citations} citations</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Год публикации
        current_year = datetime.now().year
        min_year = st.slider(
            "Minimum Publication Year",
            min_value=2000,
            max_value=current_year,
            value=2015,
            help="Filter works published after this year"
        )
        
        # Типы публикаций
        st.markdown("<p style='font-weight: 600; margin: 15px 0 5px 0;'>Publication Types</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            journal = st.checkbox("Journal", value=True)
            conference = st.checkbox("Conference", value=False)
        with col2:
            repository = st.checkbox("Repository", value=True)
            book = st.checkbox("Book", value=False)
        
        venue_types = []
        if journal: venue_types.append('journal')
        if conference: venue_types.append('conference')
        if repository: venue_types.append('repository')
        if book: venue_types.append('book')
        
        # Open Access
        open_access = st.checkbox("🔓 Open Access Only", value=False)
        
        # Минимальная релевантность
        min_relevance = st.slider(
            "Minimum Relevance Score",
            min_value=1,
            max_value=10,
            value=3,
            help="Filter works with at least this relevance score"
        )
    
    return {
        'min_citations': min_citations,
        'max_citations': max_citations,
        'min_year': min_year,
        'venue_types': venue_types,
        'open_access': open_access,
        'min_relevance': min_relevance
    }

def apply_filters(works: List[dict], filters: Dict) -> List[dict]:
    """Применение фильтров"""
    filtered = []
    
    for work in works:
        # Цитирования (диапазон)
        citations = work.get('cited_by_count', 0)
        if not (filters['min_citations'] <= citations <= filters['max_citations']):
            continue
        
        # Год
        if work.get('publication_year', 0) < filters['min_year']:
            continue
        
        # Тип издания
        venue_type = work.get('venue_type', '')
        if filters['venue_types'] and venue_type not in filters['venue_types']:
            continue
        
        # Открытый доступ
        if filters['open_access'] and not work.get('is_oa', False):
            continue
        
        # Релевантность
        if work.get('relevance_score', 0) < filters['min_relevance']:
            continue
        
        filtered.append(work)
    
    return filtered

# ============================================================================
# ШАГИ МАСТЕР-ПРОЦЕССА
# ============================================================================

def step_data_input():
    """Шаг 1: Ввод данных"""
    st.markdown("""
    <div class="step-card">
        <h2>📥 Step 1: Input Research DOIs</h2>
        <p>Enter DOI identifiers of relevant papers to analyze their topics and keywords.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        doi_input = st.text_area(
            "**DOI Input** (one per line or comma-separated):",
            height=200,
            placeholder="Example formats:\n10.1016/j.jpowsour.2020.228660\n10.1038/s41560-020-00734-0\nhttps://doi.org/10.1021/acsenergylett.1c00123\n\nOr paste a list of DOIs from your references...",
            help="You can enter up to 300 DOI identifiers"
        )
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if doi_input:
                dois = parse_doi_input(doi_input)
                if dois:
                    st.session_state.dois = dois
                    st.session_state.current_step = 2
                    st.rerun()
                else:
                    st.error("❌ No valid DOI identifiers found. Please check your input.")
            else:
                st.error("❌ Please enter at least one DOI")
    
    with col2:
        st.markdown("""
        <div class="info-message">
            <h4>📋 Quick Tips:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                <li>Copy DOIs from reference lists</li>
                <li>Works with any DOI format</li>
                <li>Supports up to 300 papers</li>
                <li>Automatic deduplication</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Пример DOI
        st.markdown("""
        <div style="margin-top: 20px;">
            <p style="font-weight: 600; color: #666;">Try this example:</p>
            <code style="background: #f5f5f5; padding: 8px; border-radius: 5px; display: block; font-size: 0.85rem;">
            10.1038/s41560-020-00734-0<br>
            10.1016/j.jpowsour.2020.228660<br>
            10.1021/acsenergylett.1c00123
            </code>
        </div>
        """, unsafe_allow_html=True)

def step_analysis():
    """Шаг 2: Анализ"""
    st.markdown("""
    <div class="step-card">
        <h2>🔍 Step 2: Analysis in Progress</h2>
        <p>Fetching data from OpenAlex and analyzing papers...</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'dois' not in st.session_state:
        st.error("❌ No data to analyze. Please go back to Step 1.")
        if st.button("⬅️ Back to Data Input"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    dois = st.session_state.dois
    
    # Показываем метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card("DOIs to Process", len(dois), "", "🔢")
    with col2:
        create_metric_card("Estimated Time", f"~{len(dois)//10} sec", "", "⏱️")
    with col3:
        create_metric_card("API Queries", "0/8 per sec", "", "⚡")
    
    # Загрузка данных
    with st.spinner("Fetching data from OpenAlex..."):
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
    
    # Показываем результаты
    st.markdown(f"""
    <div class="success-message">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #2E7D32;">✅ Analysis Complete!</h3>
                <p style="margin: 5px 0 0 0;">Successfully processed {successful} papers</p>
            </div>
            <span style="font-size: 2.5rem;">🎯</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Successful", successful, "", "✅")
    with col2:
        create_metric_card("Failed", failed, "", "❌")
    with col3:
        create_metric_card("Topics Found", len(topic_counter), "", "🏷️")
    with col4:
        avg_citations = np.mean([w.get('cited_by_count', 0) for w in works_data]) if works_data else 0
        create_metric_card("Avg Citations", f"{avg_citations:.1f}", "", "📈")
    
    # Топ темы
    if topic_counter:
        st.markdown("<h4>📊 Top Research Topics:</h4>", unsafe_allow_html=True)
        topics = topic_counter.most_common(8)
        
        cols = st.columns(4)
        for idx, (topic, count) in enumerate(topics):
            with cols[idx % 4]:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <div style="font-weight: 600; color: #333; margin-bottom: 5px;">{topic[:25]}{'...' if len(topic) > 25 else ''}</div>
                    <div style="color: #667eea; font-weight: 700;">{count} papers</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Кнопка продолжения
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Continue to Topic Selection", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

def step_topic_selection():
    """Шаг 3: Выбор темы"""
    st.markdown("""
    <div class="step-card">
        <h2>🎯 Step 3: Select Research Topic</h2>
        <p>Choose a topic for deep analysis of under-cited papers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.works_data:
        st.error("❌ No data available. Please start from Step 1.")
        if st.button("⬅️ Back to Data Input"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # Список тем
    topics = st.session_state.topic_counter.most_common()
    
    if not topics:
        st.warning("⚠️ No topics found in the analyzed papers.")
        if st.button("⬅️ Back to Analysis"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    st.markdown(f"<h4>Found {len(topics)} research topics:</h4>", unsafe_allow_html=True)
    
    # Отображение тем в виде карточек
    cols = st.columns(2)
    for idx, (topic, count) in enumerate(topics[:10]):
        with cols[idx % 2]:
            is_selected = st.session_state.get('selected_topic') == topic
            
            st.markdown(f"""
            <div style="
                background: {'#667eea' if is_selected else '#f8f9fa'}; 
                color: {'white' if is_selected else '#333'};
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 10px;
                cursor: pointer;
                border: 2px solid {'#764ba2' if is_selected else '#e0e0e0'};
                transition: all 0.3s ease;
            " onclick="this.style.background='#667eea'; this.style.color='white';">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; font-size: 1.1rem;">{topic}</div>
                    <span style="
                        background: {'white' if is_selected else '#667eea'}; 
                        color: {'#667eea' if is_selected else 'white'};
                        padding: 3px 10px; 
                        border-radius: 15px; 
                        font-size: 0.85rem; 
                        font-weight: 600;
                    ">{count} papers</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.9;">
                    Click to select for deep analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select '{topic[:20]}...'" if len(topic) > 20 else f"Select '{topic}'", 
                        key=f"select_{idx}", 
                        use_container_width=True):
                st.session_state.selected_topic = topic
                st.rerun()
    
    # Если тема выбрана, показываем детали
    if 'selected_topic' in st.session_state:
        st.markdown("---")
        topic_name = st.session_state.selected_topic
        
        # Находим ID темы
        topic_id = None
        for work in st.session_state.works_data:
            if work.get('primary_topic') == topic_name:
                topic_id = work.get('topic_id')
                break
        
        if topic_id:
            st.session_state.selected_topic_id = topic_id
            
            # Получаем статистику темы
            with st.spinner(f"Fetching statistics for '{topic_name}'..."):
                topic_stats = fetch_topic_stats_sync(topic_id)
            
            col1, col2 = st.columns(2)
            with col1:
                if topic_stats:
                    total_works = topic_stats.get('works_count', 0)
                    create_metric_card("Total Papers in Topic", f"{total_works:,}", "", "📚")
            
            with col2:
                # Получаем ключевые слова
                top_keywords = [kw for kw, _ in st.session_state.keyword_counter.most_common(10)]
                st.session_state.top_keywords = [k.lower() for k in top_keywords]
                
                st.markdown("<h4>🔑 Top Keywords:</h4>", unsafe_allow_html=True)
                keywords_html = ' '.join([f'<span class="filter-chip">{kw}</span>' for kw in top_keywords[:8]])
                st.markdown(f"<div style='margin-top: 10px;'>{keywords_html}</div>", unsafe_allow_html=True)
            
            # Кнопка для анализа
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Analyze Under-Cited Papers", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.rerun()
        else:
            st.error("❌ Could not find topic ID. Please select another topic.")

def step_results():
    """Шаг 4: Результаты"""
    st.markdown("""
    <div class="step-card">
        <h2>📊 Step 4: Analysis Results</h2>
        <p>Discover under-cited papers in your selected research area.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Показываем фильтры в сайдбаре
    filters = create_filters_ui()
    
    if 'selected_topic_id' not in st.session_state or 'top_keywords' not in st.session_state:
        st.error("❌ Topic not selected. Please go back to Step 3.")
        if st.button("⬅️ Back to Topic Selection"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    # Анализ работ по теме
    if 'relevant_works' not in st.session_state:
        with st.spinner("Searching for under-cited papers..."):
            relevant_works = analyze_works_for_topic(
                st.session_state.selected_topic_id,
                st.session_state.top_keywords,
                max_citations=filters['max_citations'],
                max_works=2000,
                top_n=100
            )
        st.session_state.relevant_works = relevant_works
    else:
        relevant_works = st.session_state.relevant_works
    
    # Применяем фильтры
    filtered_works = apply_filters(relevant_works, filters)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Papers Found", len(relevant_works), "", "📄")
    with col2:
        create_metric_card("After Filters", len(filtered_works), "", "🎯")
    with col3:
        avg_citations = np.mean([w.get('cited_by_count', 0) for w in filtered_works]) if filtered_works else 0
        create_metric_card("Avg Citations", f"{avg_citations:.1f}", "", "📈")
    with col4:
        oa_count = sum(1 for w in filtered_works if w.get('is_oa'))
        create_metric_card("Open Access", f"{oa_count}", "", "🔓")
    
    # Показываем активные фильтры
    st.markdown("<h4>🔧 Active Filters:</h4>", unsafe_allow_html=True)
    filters_html = f"""
    <span class="filter-chip">Citations: {filters['min_citations']}-{filters['max_citations']}</span>
    <span class="filter-chip">Year ≥ {filters['min_year']}</span>
    """
    if filters['open_access']:
        filters_html += '<span class="filter-chip">Open Access Only</span>'
    if filters['venue_types']:
        filters_html += f'<span class="filter-chip">Types: {", ".join(filters["venue_types"])}</span>'
    
    st.markdown(f"<div style='margin-bottom: 20px;'>{filters_html}</div>", unsafe_allow_html=True)
    
    if not filtered_works:
        st.warning("""
        <div class="warning-message">
            <h4>⚠️ No papers match your filters</h4>
            <p>Try adjusting your filters to find more results:</p>
            <ul>
                <li>Increase the citation range</li>
                <li>Include more publication types</li>
                <li>Lower the minimum relevance score</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Результаты в виде карточек
        st.markdown("<h4>🎯 Recommended Papers:</h4>", unsafe_allow_html=True)
        
        for idx, work in enumerate(filtered_works[:20], 1):
            create_result_card(work, idx)
        
        # Если много результатов, показываем кнопку "Показать еще"
        if len(filtered_works) > 20:
            if st.button("📖 Show More Results", use_container_width=True):
                st.session_state.show_all = True
        
        # Таблица для детального просмотра
        st.markdown("<h4>📋 Detailed View:</h4>", unsafe_allow_html=True)
        
        display_data = []
        for i, work in enumerate(filtered_works, 1):
            doi_url = work.get('doi_url', '')
            title = work.get('title', '')[:80] + '...' if len(work.get('title', '')) > 80 else work.get('title', '')
            
            display_data.append({
                '#': i,
                'Title': title,
                'Citations': work.get('cited_by_count', 0),
                'Relevance': work.get('relevance_score', 0),
                'Year': work.get('publication_year', ''),
                'Journal': work.get('venue_name', '')[:25],
                'DOI': f'[🔗]({doi_url})' if doi_url else 'N/A',
                'OA': '✅' if work.get('is_oa') else '❌',
                'Authors': ', '.join(work.get('authors', [])[:2])
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            column_config={
                "DOI": st.column_config.LinkColumn("DOI", display_text="View"),
                "Relevance": st.column_config.ProgressColumn(
                    "Relevance",
                    help="Relevance score (higher is better)",
                    format="%d",
                    min_value=1,
                    max_value=10
                )
            }
        )
        
        # Экспорт
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = pd.DataFrame(filtered_works).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"under_cited_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                for key in ['relevant_works', 'selected_topic', 'selected_topic_id', 'top_keywords']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.current_step = 1
                st.rerun()
        
        # Визуализации
        with st.expander("📊 Visualizations"):
            col1, col2 = st.columns(2)
            
            with col1:
                citations = [w.get('cited_by_count', 0) for w in filtered_works]
                fig = px.histogram(
                    x=citations, 
                    nbins=20, 
                    title='Citation Distribution',
                    labels={'x': 'Number of Citations', 'y': 'Number of Papers'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                years = [w.get('publication_year', 0) for w in filtered_works if w.get('publication_year', 0) > 1900]
                if years:
                    year_counts = pd.Series(years).value_counts().sort_index()
                    fig = px.line(
                        x=year_counts.index, 
                        y=year_counts.values, 
                        title='Publications by Year',
                        labels={'x': 'Year', 'y': 'Number of Papers'},
                        color_discrete_sequence=['#764ba2']
                    )
                    fig.update_traces(line=dict(width=3))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция приложения"""
    
    # Инициализация состояния
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Заголовок
    st.markdown("""
    <h1 class="main-header">🔬 CTA Research Explorer Pro</h1>
    <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
    Discover under-cited papers in your research area using AI-powered analysis of OpenAlex database.
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
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem;">
            <p>Powered by OpenAlex API • Uses polite pool for better rate limits</p>
            <p>Data is cached locally for faster subsequent queries</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()




