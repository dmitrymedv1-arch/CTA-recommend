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
    page_title="CTA Article Recommendation Pro+",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Конфигурация OpenAlex API
OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = "your-email@example.com"  # Замените на ваш email для polite pool
POLITE_POOL_HEADER = {'User-Agent': f'CTA-App (mailto:{MAILTO})'}

# Настройки rate limit
RATE_LIMIT_PER_SECOND = 8  # 8 запросов в секунду для polite pool
BATCH_SIZE = 50  # Размер batch запросов для работ
CURSOR_PAGE_SIZE = 200  # Размер страницы для cursor pagination
MAX_WORKERS_ASYNC = 3  # Ограничение параллельных запросов
MAX_RETRIES = 3  # Максимум попыток при ошибках
INITIAL_DELAY = 1  # Начальная задержка при retry
MAX_DELAY = 60  # Максимальная задержка

# Настройки кэширования
CACHE_DIR = Path("./cache")
CACHE_DB = CACHE_DIR / "openalex_cache.db"
CACHE_EXPIRY_DAYS = 30  # Дней хранения кэша

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
    """Инициализирует базу данных для кэширования"""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    # Таблица для кэширования работ по DOI
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS works_cache (
            doi TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        )
    ''')
    
    # Таблица для кэширования работ по теме
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
    
    # Таблица для кэширования статистики тем
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics_cache (
            topic_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        )
    ''')
    
    # Создаем индексы для ускорения запросов
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_works_expires ON works_cache(expires_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_works_expires ON topic_works_cache(expires_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topics_expires ON topics_cache(expires_at)')
    
    conn.commit()
    conn.close()

def get_cache_key(prefix: str, key: str) -> str:
    """Создает уникальный ключ кэша"""
    return hashlib.md5(f"{prefix}:{key}".encode()).hexdigest()

@st.cache_resource
def get_db_connection():
    """Возвращает соединение с базой данных кэша"""
    init_cache_db()
    return sqlite3.connect(CACHE_DB, check_same_thread=False)

def cache_work(doi: str, data: dict):
    """Кэширует данные работы"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
    
    cursor.execute('''
        INSERT OR REPLACE INTO works_cache (doi, data, expires_at)
        VALUES (?, ?, ?)
    ''', (doi, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_work(doi: str) -> Optional[dict]:
    """Получает кэшированные данные работы"""
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
    """Кэширует данные работ по теме"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=7)  # Кэш тем на 7 дней
    
    cursor.execute('''
        INSERT OR REPLACE INTO topic_works_cache (topic_id, cursor_key, data, expires_at)
        VALUES (?, ?, ?, ?)
    ''', (topic_id, cursor_key, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_topic_works(topic_id: str, cursor_key: str) -> Optional[dict]:
    """Получает кэшированные данные работ по теме"""
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
    """Кэширует статистику темы"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=30)  # Статистика на 30 дней
    
    cursor.execute('''
        INSERT OR REPLACE INTO topics_cache (topic_id, data, expires_at)
        VALUES (?, ?, ?)
    ''', (topic_id, json.dumps(data), expires_at))
    
    conn.commit()

def get_cached_topic_stats(topic_id: str) -> Optional[dict]:
    """Получает кэшированную статистику темы"""
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
    """Очищает устаревший кэш"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM works_cache WHERE expires_at <= ?', (datetime.now(),))
    cursor.execute('DELETE FROM topic_works_cache WHERE expires_at <= ?', (datetime.now(),))
    cursor.execute('DELETE FROM topics_cache WHERE expires_at <= ?', (datetime.now(),))
    
    conn.commit()

# ============================================================================
# ASYNCIO + AIOHTTP ДЛЯ ПАРАЛЛЕЛЬНЫХ ЗАПРОСОВ
# ============================================================================

class OpenAlexAsyncClient:
    """Асинхронный клиент для OpenAlex API с rate limiting"""
    
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
        """Делает запрос с rate limiting и retry логикой"""
        async with self.semaphore:
            # Rate limiting: 8 запросов в секунду
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
        """Batch запрос работ по нескольким DOI одновременно"""
        if not dois:
            return []
        
        # Проверяем кэш
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
        
        # Делаем batch запрос для некэшированных DOI
        logger.info(f"Fetching {len(uncached_dois)} works via batch API")
        
        # OpenAlex поддерживает filter по нескольким DOI через |
        doi_filter = "|".join(uncached_dois)
        url = f"{OPENALEX_BASE_URL}/works?filter=doi:{doi_filter}&per-page=200"
        
        try:
            data = await self.make_request(url)
            if data and 'results' in data:
                results = data['results']
                
                # Кэшируем результаты
                for work in results:
                    doi = work.get('doi', '').replace('https://doi.org/', '')
                    if doi:
                        cache_work(doi, work)
                
                # Сопоставляем результаты с запрошенными DOI
                doi_to_work = {w.get('doi', '').replace('https://doi.org/', ''): w for w in results}
                batch_results = []
                
                for doi in uncached_dois:
                    if doi in doi_to_work:
                        batch_results.append(doi_to_work[doi])
                    else:
                        # Если работа не найдена, пробуем получить через отдельный запрос
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
        """Получает одну работу по DOI"""
        cached = get_cached_work(doi)
        if cached:
            return cached
        
        url = f"{OPENALEX_BASE_URL}/works/https://doi.org/{doi}"
        data = await self.make_request(url)
        
        if data:
            cache_work(doi, data)
        
        return data
    
    async def fetch_works_by_topic_cursor(self, topic_id: str, max_results: int = 2000) -> List[dict]:
        """Получает работы по теме с использованием cursor pagination"""
        all_works = []
        cursor = "*"
        page_count = 0
        
        # Проверяем, есть ли уже кэшированные данные для этой темы
        cache_key = f"{topic_id}_cursor_{cursor}"
        cached = get_cached_topic_works(topic_id, cache_key)
        
        if cached and len(cached) >= max_results:
            logger.info(f"Using cached data for topic {topic_id}")
            return cached[:max_results]
        
        logger.info(f"Fetching works for topic {topic_id} (max: {max_results})")
        
        try:
            while len(all_works) < max_results and cursor:
                page_count += 1
                
                # Используем cursor pagination вместо обычной
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
                
                # Получаем следующий cursor
                meta = data.get('meta', {})
                cursor = meta.get('next_cursor')
                
                logger.info(f"Page {page_count}: got {len(works)} works, total: {len(all_works)}")
                
                # Сохраняем промежуточные результаты в кэш
                cache_key = f"{topic_id}_cursor_{cursor or 'end'}"
                cache_topic_works(topic_id, cache_key, all_works)
                
                if not cursor or page_count >= 10:  # Ограничиваем количество страниц
                    break
                
                # Небольшая пауза между страницами
                await asyncio.sleep(0.5)
            
            # Обрезаем до нужного количества
            result = all_works[:max_results]
            
            # Сохраняем финальный результат
            cache_topic_works(topic_id, "final", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching topic works: {str(e)}")
            return all_works
    
    async def fetch_topic_stats(self, topic_id: str) -> Optional[dict]:
        """Получает статистику по теме"""
        cached = get_cached_topic_stats(topic_id)
        if cached:
            return cached
        
        url = f"{OPENALEX_BASE_URL}/topics/{topic_id}"
        data = await self.make_request(url)
        
        if data:
            cache_topic_stats(topic_id, data)
        
        return data

# ============================================================================
# СИНХРОННЫЕ ОБЕРТКИ ДЛЯ STREAMLIT
# ============================================================================

def run_async(coro):
    """Запускает асинхронную корутину в синхронном контексте"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def fetch_works_by_dois_sync(dois: List[str]) -> Tuple[List[dict], int, int]:
    """Синхронная обертка для batch запроса работ"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Разбиваем на батчи
    batches = [dois[i:i + BATCH_SIZE] for i in range(0, len(dois), BATCH_SIZE)]
    all_results = []
    successful = 0
    failed = 0
    
    async def process_batches():
        nonlocal all_results, successful, failed
        async with OpenAlexAsyncClient() as client:
            for i, batch in enumerate(batches):
                # Обновляем прогресс
                progress = (i + 1) / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Батч {i+1}/{len(batches)}: обработка {len(batch)} DOI")
                
                # Запрашиваем батч
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
                
                # Пауза между батчами
                if i < len(batches) - 1:
                    await asyncio.sleep(1)
    
    run_async(process_batches())
    
    progress_bar.empty()
    status_text.empty()
    
    return all_results, successful, failed

def fetch_works_by_topic_sync(topic_id: str, max_results: int = 2000) -> List[dict]:
    """Синхронная обертка для запроса работ по теме"""
    async def fetch():
        async with OpenAlexAsyncClient() as client:
            return await client.fetch_works_by_topic_cursor(topic_id, max_results)
    
    return run_async(fetch())

def fetch_topic_stats_sync(topic_id: str) -> Optional[dict]:
    """Синхронная обертка для запроса статистики темы"""
    async def fetch():
        async with OpenAlexAsyncClient() as client:
            return await client.fetch_topic_stats(topic_id)
    
    return run_async(fetch())

# ============================================================================
# ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ (упрощенная)
# ============================================================================

def normalize_word(word: str) -> str:
    """Нормализация слова"""
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
    """Извлечение ключевых слов"""
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
    """Парсинг DOI"""
    if not text or not text.strip():
        return []
    
    # Извлекаем DOI с помощью регулярного выражения
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    dois = re.findall(doi_pattern, text, re.IGNORECASE)
    
    # Очистка
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
    """Анализ ключевых слов"""
    all_keywords = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_keywords_from_title, title) for title in titles]
        for future in as_completed(futures):
            all_keywords.extend(future.result())
    
    return Counter(all_keywords)

def enrich_work_data(work: dict) -> dict:
    """Обогащение данных работы"""
    if not work:
        return {}
    
    enriched = {
        'id': work.get('id', ''),
        'doi': work.get('doi', '').replace('https://doi.org/', ''),
        'title': work.get('title', ''),
        'publication_date': work.get('publication_date', ''),
        'publication_year': work.get('publication_year', 0),
        'cited_by_count': work.get('cited_by_count', 0),
        'type': work.get('type', ''),
        'abstract': work.get('abstract', '')[:500] if work.get('abstract') else '',
    }
    
    # Авторы
    authorships = work.get('authorships', [])
    authors = []
    institutions = set()
    
    for authorship in authorships:
        author = authorship.get('author', {})
        if author:
            author_name = author.get('display_name', '')
            if author_name:
                authors.append(author_name)
        
        for inst in authorship.get('institutions', []):
            inst_name = inst.get('display_name', '')
            if inst_name:
                institutions.add(inst_name)
    
    enriched['authors'] = authors[:5]
    enriched['institutions'] = list(institutions)
    
    # Журнал
    source = work.get('primary_location', {}).get('source', {})
    enriched['venue_name'] = source.get('display_name', '')
    enriched['venue_type'] = source.get('type', '')
    enriched['is_oa'] = work.get('open_access', {}).get('is_oa', False)
    
    # Темы
    topics = work.get('topics', [])
    if topics:
        sorted_topics = sorted(topics, key=lambda x: x.get('score', 0), reverse=True)
        primary_topic = sorted_topics[0]
        enriched['primary_topic'] = primary_topic.get('display_name', '')
        enriched['topic_id'] = primary_topic.get('id', '').split('/')[-1]
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
    """Анализ работ по теме"""
    
    with st.spinner(f"Загрузка до {max_works} работ..."):
        works = fetch_works_by_topic_sync(topic_id, max_works)
    
    if not works:
        return []
    
    with st.spinner(f"Анализ {len(works)} работ..."):
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
        
        # Сортировка и выбор топ-N
        analyzed.sort(key=lambda x: x['relevance_score'], reverse=True)
        return analyzed[:top_n]

def create_filters_ui() -> Dict:
    """Создание интерфейса фильтров"""
    with st.sidebar:
        st.header("🎯 Фильтры")
        
        max_citations = st.slider(
            "Макс. цитирований",
            min_value=0,
            max_value=50,
            value=10,
            help="Включает работы с указанным или меньшим числом цитирований"
        )
        
        min_year = st.number_input(
            "Мин. год",
            min_value=1900,
            max_value=datetime.now().year,
            value=2015
        )
        
        venue_types = st.multiselect(
            "Тип издания",
            options=['journal', 'repository', 'conference', 'book'],
            default=['journal', 'repository']
        )
        
        open_access = st.checkbox("Только открытый доступ", value=False)
        
        min_relevance = st.slider(
            "Мин. релевантность",
            min_value=1,
            max_value=10,
            value=3
        )
    
    return {
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
        # Цитирования
        if work.get('cited_by_count', 0) > filters['max_citations']:
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

def main():
    """Главная функция приложения"""
    
    st.title("🚀 CTA Article Recommendation Pro+")
    st.markdown("""
    ### Высокопроизводительный поиск низкоцитируемых научных статей
    
    **Оптимизировано для:**
    - 🚄 Batch запросы к OpenAlex API
    - 🎯 Cursor pagination для получения тысяч работ
    - 💾 Интеллектуальное кэширование в SQLite
    - ⚡ Асинхронная обработка с rate limiting
    """)
    
    # Инициализация сессии
    if 'works_data' not in st.session_state:
        st.session_state.works_data = []
    if 'topic_counter' not in st.session_state:
        st.session_state.topic_counter = Counter()
    if 'keyword_counter' not in st.session_state:
        st.session_state.keyword_counter = Counter()
    
    # Очистка старого кэша
    clear_old_cache()
    
    # Фильтры
    filters = create_filters_ui()
    
    # Основной интерфейс
    tab1, tab2, tab3 = st.tabs(["📥 Ввод DOI", "📊 Анализ", "🎯 Результаты"])
    
    with tab1:
        st.subheader("Введите DOI для анализа")
        
        doi_input = st.text_area(
            "DOI (по одному на строку или через запятую):",
            height=150,
            placeholder="10.1016/j.jpowsour.2020.228660\n10.1038/s41560-020-00734-0\nhttps://doi.org/10.1021/acsenergylett.1c00123"
        )
        
        if st.button("🚀 Начать анализ", type="primary"):
            if doi_input:
                dois = parse_doi_input(doi_input)
                st.info(f"Найдено {len(dois)} DOI. Начинаю загрузку...")
                
                # Загрузка работ
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
                
                # Сохранение в сессии
                st.session_state.works_data = works_data
                st.session_state.topic_counter = topic_counter
                st.session_state.keyword_counter = keyword_counter
                
                st.success(f"✅ Загружено {successful} работ, найдено {len(topic_counter)} тем")
                
                # Статистика
                col1, col2, col3 = st.columns(3)
                col1.metric("Успешно", successful)
                col2.metric("Темы", len(topic_counter))
                col3.metric("Средние цитирования", 
                          f"{np.mean([w.get('cited_by_count', 0) for w in works_data]):.1f}" if works_data else "0")
    
    with tab2:
        if not st.session_state.works_data:
            st.info("Сначала загрузите DOI на вкладке 'Ввод DOI'")
        else:
            st.subheader("Анализ тем")
            
            # Список тем
            topics = st.session_state.topic_counter.most_common()
            
            if topics:
                st.write(f"Найдено {len(topics)} тем:")
                
                for i, (topic, count) in enumerate(topics[:20], 1):
                    st.write(f"{i}. **{topic}** - {count} работ")
                
                # Выбор темы
                topic_options = [f"{topic} ({count} работ)" for topic, count in topics]
                selected = st.selectbox("Выберите тему для детального анализа:", topic_options)
                
                if selected:
                    topic_name = selected.split(" (")[0]
                    
                    if st.button("🔍 Анализировать тему", type="primary"):
                        # Находим ID темы
                        topic_id = None
                        for work in st.session_state.works_data:
                            if work.get('primary_topic') == topic_name:
                                topic_id = work.get('topic_id')
                                break
                        
                        if topic_id:
                            # Получаем статистику темы
                            with st.spinner("Получение статистики..."):
                                topic_stats = fetch_topic_stats_sync(topic_id)
                            
                            if topic_stats:
                                total_works = topic_stats.get('works_count', 0)
                                st.metric("Всего работ по теме", f"{total_works:,}")
                            
                            # Получаем ключевые слова
                            top_keywords = [kw for kw, _ in st.session_state.keyword_counter.most_common(15)]
                            
                            # Анализируем работы по теме
                            with st.spinner(f"Поиск релевантных работ..."):
                                relevant_works = analyze_works_for_topic(
                                    topic_id,
                                    [k.lower() for k in top_keywords],
                                    max_citations=filters['max_citations'],
                                    max_works=2000,
                                    top_n=100
                                )
                            
                            st.session_state.selected_topic = topic_name
                            st.session_state.selected_topic_id = topic_id
                            st.session_state.relevant_works = relevant_works
                            
                            st.success(f"✅ Найдено {len(relevant_works)} релевантных работ")
                        else:
                            st.error("Не удалось найти ID темы")
            else:
                st.warning("Темы не найдены")
    
    with tab3:
        if 'relevant_works' not in st.session_state:
            st.info("Сначала проанализируйте тему на вкладке 'Анализ'")
        else:
            works = st.session_state.relevant_works
            topic = st.session_state.get('selected_topic', 'Неизвестная тема')
            
            st.subheader(f"Результаты: {topic}")
            
            # Применяем фильтры
            filtered_works = apply_filters(works, filters)
            
            st.write(f"Найдено работ: {len(works)} → После фильтров: {len(filtered_works)}")
            
            if filtered_works:
                # Создаем DataFrame для отображения
                display_data = []
                for i, work in enumerate(filtered_works, 1):
                    display_data.append({
                        '№': i,
                        'Заголовок': work.get('title', '')[:100] + '...' if len(work.get('title', '')) > 100 else work.get('title', ''),
                        'Цитирования': work.get('cited_by_count', 0),
                        'Релевантность': work.get('relevance_score', 0),
                        'Год': work.get('publication_year', ''),
                        'Журнал': work.get('venue_name', '')[:30],
                        'DOI': work.get('doi', ''),
                        'Открытый доступ': '✅' if work.get('is_oa') else '❌'
                    })
                
                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True, height=500)
                
                # Экспорт
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Скачать CSV",
                    data=csv,
                    file_name=f"results_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Визуализации
                st.subheader("📊 Визуализации")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Распределение по цитированиям
                    citations = [w.get('cited_by_count', 0) for w in filtered_works]
                    fig = px.histogram(x=citations, nbins=20, title='Распределение цитирований')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Распределение по годам
                    years = [w.get('publication_year', 0) for w in filtered_works if w.get('publication_year', 0) > 1900]
                    if years:
                        year_counts = pd.Series(years).value_counts().sort_index()
                        fig = px.line(x=year_counts.index, y=year_counts.values, title='Публикации по годам')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Нет работ, соответствующих фильтрам")

if __name__ == "__main__":
    main()