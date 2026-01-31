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
# НОВЫЕ КЛАССЫ ДЛЯ ХИМИЧЕСКОГО АНАЛИЗА
# ============================================================================

class ChemicalFormulaAnalyzer:
    def __init__(self):
        # Паттерны для распознавания химических формул
        self.patterns = {
            'simple_formula': r'\b([A-Z][a-z]?\d*)+',  # BaTiO3, La2O3
            'doped_formula': r'\b([A-Z][a-z]?)\([^)]+\)O\d*',  # La(Fe,Ni)O3
            'layered_formula': r'\b([A-Z][a-z]?)\d+[A-Z][a-z]?\d+O\d*',  # La2NiO4
            'composite': r'\b([A-Z][a-z]?\d*[/-][A-Z][a-z]?\d*)',  # BaTiO3/SrTiO3
            'solid_solution': r'\b([A-Z][a-z]?)_{?\d+\.\d+}[A-Z][a-z]?',  # Ba_0.6Sr_0.4TiO3
        }
        
        # Элементы для поиска (можно расширить)
        self.elements = {
            'La', 'Ni', 'O', 'Co', 'Fe', 'Mn', 'Cu', 'Zn', 'Ti', 'Zr',
            'Ba', 'Sr', 'Ca', 'Pb', 'Bi', 'Na', 'K', 'Li', 'Mg', 'Al',
            'Si', 'Ge', 'Sn', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Y', 'Sc',
            'Hf', 'Ta', 'Nb', 'Mo', 'W', 'Ru', 'Rh', 'Pd', 'Pt', 'Ag',
            'Au', 'Ir', 'Os', 'Re', 'Cr', 'V', 'Mn', 'P', 'S', 'Cl',
            'F', 'Br', 'I', 'As', 'Se', 'Te', 'Sb', 'Cd', 'Hg', 'Ga',
            'In', 'Tl', 'Pb', 'Sn', 'Ge', 'Si', 'B', 'C', 'N'
        }
    
    def extract_chemical_entities(self, text: str) -> List[Dict]:
        """Извлечение химических формул и соединений"""
        entities = []
        
        if not text:
            return entities
        
        text_str = str(text)
        
        # 1. Простые формулы (LaNiO3, BaTiO3)
        for match in re.finditer(r'\b([A-Z][a-z]?\d*)+\b', text_str):
            formula = match.group()
            if self._is_valid_formula(formula):
                elements = self._extract_elements(formula)
                entities.append({
                    'type': 'simple_formula',
                    'formula': formula,
                    'elements': elements,
                    'normalized': self._normalize_formula(formula)
                })
        
        # 2. Составные формулы с запятыми/скобками
        for match in re.finditer(r'\b([A-Z][a-z]?[\(\)\d\.,\s\-]+[A-Za-z\d]+)', text_str):
            formula = match.group()
            # Проверка на наличие химических элементов
            if any(elem in formula for elem in self.elements):
                entities.append({
                    'type': 'complex_formula',
                    'formula': formula,
                    'elements': self._extract_elements(formula),
                    'normalized': self._normalize_complex_formula(formula)
                })
        
        # 3. Замещенные соединения (допированные)
        doped_patterns = [
            r'([A-Z][a-z]?)([A-Z][a-z]?)_{?(\d[\.\d]*)}?([A-Z][a-z]?)_{?(\d[\.\d]*)}?O',  # LaNi_0.5Fe_0.5O3
            r'([A-Z][a-z]?)\(([^)]+)\)O',  # La(Fe,Co)O3
        ]
        
        for pattern in doped_patterns:
            for match in re.finditer(pattern, text_str):
                formula = match.group()
                entities.append({
                    'type': 'doped_formula',
                    'formula': formula,
                    'base': match.group(1) if match.groups() else None,
                    'elements': self._extract_elements(formula),
                    'normalized': self._normalize_doped_formula(formula)
                })
        
        # 4. Формулы с нижними индексами
        for match in re.finditer(r'\b([A-Z][a-z]?)_\{?(\d+(?:\.\d+)?)\}?([A-Z][a-z]?)_\{?(\d+(?:\.\d+)?)\}?O\d*', text_str):
            formula = match.group()
            entities.append({
                'type': 'indexed_formula',
                'formula': formula,
                'elements': self._extract_elements(formula),
                'normalized': self._normalize_formula(formula)
            })
        
        # Удаляем дубликаты по формуле
        unique_entities = []
        seen_formulas = set()
        for entity in entities:
            if entity['formula'] not in seen_formulas:
                seen_formulas.add(entity['formula'])
                unique_entities.append(entity)
        
        return unique_entities
    
    def _is_valid_formula(self, formula: str) -> bool:
        """Проверка, является ли строка химической формулой"""
        # Удаляем цифры и проверяем наличие химических элементов
        letters = re.sub(r'\d+', '', formula)
        return any(elem in formula for elem in self.elements) and len(letters) >= 2
    
    def _extract_elements(self, formula: str) -> List[str]:
        """Извлечение элементов из формулы"""
        elements = []
        i = 0
        formula_clean = re.sub(r'[\(\)\{\}_]', '', formula)  # Удаляем скобки и нижние индексы
        
        while i < len(formula_clean):
            if formula_clean[i].isupper():
                elem = formula_clean[i]
                if i + 1 < len(formula_clean) and formula_clean[i+1].islower():
                    elem += formula_clean[i+1]
                    i += 1
                # Проверяем, что это известный элемент
                if elem in self.elements:
                    elements.append(elem)
            i += 1
        
        return list(set(elements))  # Уникальные элементы
    
    def _normalize_formula(self, formula: str) -> str:
        """Нормализация формулы (приведение к каноническому виду)"""
        # Удаление лишних пробелов, приведение к единому регистру
        normalized = formula.strip()
        
        # Удаление пробелов в формуле
        normalized = normalized.replace(' ', '')
        
        # Приведение O2, O3 к O (только для оксидов)
        if 'O' in normalized:
            normalized = re.sub(r'O(\d+)', 'O', normalized)
        
        # Удаление лишних символов
        normalized = re.sub(r'[\(\)\{\}_]', '', normalized)
        
        return normalized
    
    def _normalize_complex_formula(self, formula: str) -> str:
        """Нормализация комплексных формул"""
        normalized = formula.strip()
        
        # Удаление пробелов вокруг запятых и скобок
        normalized = re.sub(r'\s*,\s*', ',', normalized)
        normalized = re.sub(r'\s*\(\s*', '(', normalized)
        normalized = re.sub(r'\s*\)\s*', ')', normalized)
        
        return normalized
    
    def _normalize_doped_formula(self, formula: str) -> str:
        """Нормализация допированных формул"""
        normalized = formula.strip()
        
        # Замена подчеркиваний и скобок
        normalized = normalized.replace('{', '').replace('}', '')
        
        return normalized
    
    def group_by_chemical_family(self, formulas: List[str]) -> Dict[str, List[str]]:
        """Группировка формул по химическим семействам"""
        families = {}
        
        for formula in formulas:
            # Извлекаем основные элементы
            elements = self._extract_elements(formula)
            if not elements:
                continue
            
            # Создаем ключ семейства (например, "La-Ni-O")
            elements_sorted = sorted(set(elements))
            family_key = '-'.join(elements_sorted)
            
            if family_key not in families:
                families[family_key] = {
                    'formulas': [],
                    'elements': elements_sorted
                }
            
            if formula not in families[family_key]['formulas']:
                families[family_key]['formulas'].append(formula)
        
        return families
    
    def calculate_chemical_similarity(self, formula1: str, formula2: str) -> float:
        """Расчет химической схожести между двумя формулами"""
        elements1 = set(self._extract_elements(formula1))
        elements2 = set(self._extract_elements(formula2))
        
        if not elements1 or not elements2:
            return 0.0
        
        # Коэффициент Жаккара
        intersection = len(elements1.intersection(elements2))
        union = len(elements1.union(elements2))
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Бонус за одинаковые основные элементы (первые в формуле)
        formula1_norm = self._normalize_formula(formula1)
        formula2_norm = self._normalize_formula(formula2)
        
        if formula1_norm[:3] == formula2_norm[:3]:
            similarity += 0.2
        
        return min(similarity, 1.0)

class MaterialsSemanticAnalyzer:
    def __init__(self):
        # Классификация материалов по типам
        self.material_types = {
            'perovskite': ['ab03', 'перовскит', 'perovskite'],
            'spinel': ['ab2o4', 'шпинель', 'spinel'],
            'fluorite': ['ao2', 'флюорит', 'fluorite'],
            'layered': ['слоистый', 'layered', 'ruddlesden', 'popper'],
            'composite': ['композит', 'composite', 'гибрид', 'hybrid'],
            'nanomaterial': ['нан', 'nano', 'квант', 'quantum', 'nanoparticle'],
            'thin_film': ['пленка', 'film', 'тонкий слой', 'thin film'],
            'catalyst': ['катализатор', 'catalyst', 'каталитический'],
            'electrode': ['электрод', 'electrode'],
            'membrane': ['мембрана', 'membrane'],
            'ceramic': ['керамик', 'ceramic'],
            'polymer': ['полимер', 'polymer'],
            'alloy': ['сплав', 'alloy'],
            'semiconductor': ['полупроводник', 'semiconductor'],
        }
        
        # Свойства материалов
        self.properties = {
            'electrical': ['проводимость', 'conductivity', 'сопротивление', 'resistivity', 'electrical'],
            'magnetic': ['магнит', 'magnetic', 'ферромагнит', 'ferromagnetic', 'magnetization'],
            'optical': ['оптический', 'optical', 'люминесцент', 'luminescent', 'photoluminescence'],
            'catalytic': ['каталитический', 'catalytic', 'активность', 'activity', 'catalysis'],
            'structural': ['структур', 'structural', 'кристалл', 'crystal', 'xrd', 'diffraction'],
            'thermal': ['тепловой', 'thermal', 'термо', 'thermo', 'теплопроводность'],
            'mechanical': ['механический', 'mechanical', 'прочность', 'strength', 'hardness'],
            'electrochemical': ['электрохимический', 'electrochemical', 'емкость', 'capacity', 'battery'],
        }
        
        # Применения материалов
        self.applications = {
            'fuel_cell': ['топливный элемент', 'fuel cell', 'sofc', 'soec', 'solid oxide'],
            'battery': ['батарея', 'battery', 'аккумулятор', 'li-ion', 'lithium', 'energy storage'],
            'catalysis': ['катализ', 'catalysis', 'реформинг', 'reforming', 'oxidation'],
            'sensor': ['сенсор', 'sensor', 'датчик', 'detection', 'sensing'],
            'memory': ['память', 'memory', 'memristor', 'ferroelectric'],
            'energy_storage': ['хранение энергии', 'energy storage', 'суперконденсатор', 'supercapacitor'],
            'photocatalysis': ['фотокатализ', 'photocatalysis', 'water splitting'],
            'photovoltaic': ['фотовольтаический', 'photovoltaic', 'solar cell', 'солнечная ячейка'],
        }
    
    def analyze_materials_context(self, text: str) -> Dict:
        """Анализ контекста материалов в тексте"""
        result = {
            'material_types': [],
            'properties_studied': [],
            'applications': [],
            'chemical_context': []
        }
        
        if not text:
            return result
        
        text_lower = text.lower()
        
        # Определение типа материала
        for mat_type, keywords in self.material_types.items():
            if any(keyword in text_lower for keyword in keywords):
                result['material_types'].append(mat_type)
        
        # Определение изучаемых свойств
        for prop, keywords in self.properties.items():
            if any(keyword in text_lower for keyword in keywords):
                result['properties_studied'].append(prop)
        
        # Определение применений
        for app, keywords in self.applications.items():
            if any(keyword in text_lower for keyword in keywords):
                result['applications'].append(app)
        
        # Извлечение химического контекста
        chem_patterns = [
            r'synthesis of',
            r'preparation of',
            r'fabrication of',
            r'growth of',
            r'characterization of',
            r'study of',
            r'properties of',
            r'application of',
        ]
        
        for pattern in chem_patterns:
            if re.search(pattern, text_lower):
                result['chemical_context'].append(pattern.replace(' of', ''))
        
        return result
    
    def get_research_focus_weight(self, focus_areas: List[str]) -> float:
        """Вес исследовательского фокуса"""
        weights = {
            'synthesis': 1.0,
            'preparation': 1.0,
            'fabrication': 1.0,
            'properties': 1.2,  # Свойства часто важнее для рекомендаций
            'characterization': 1.1,
            'application': 1.3,  # Применения - наиболее релевантны
            'study': 1.0,
            'growth': 1.0,
        }
        
        total_weight = 0.0
        for focus in focus_areas:
            total_weight += weights.get(focus, 1.0)
        
        return total_weight

class HybridRelevanceScorer:
    def __init__(self):
        # Веса для разных компонентов релевантности
        self.keyword_weight = 0.40
        self.chemical_weight = 0.35  # Высокий вес для химических формул
        self.semantic_weight = 0.15
        self.recency_weight = 0.10
        
        # Инициализация анализаторов
        self.chemical_analyzer = ChemicalFormulaAnalyzer()
        self.semantic_analyzer = MaterialsSemanticAnalyzer()
        
        # Настройки химического анализа
        self.chemical_score_config = {
            'simple_formula': 2.0,
            'complex_formula': 2.5,
            'doped_formula': 3.0,
            'indexed_formula': 2.8,
            'element_match': 1.5,
            'family_similarity': 2.0,
        }
    
    def score_work(self, work: dict, keywords: Dict[str, float], 
                   keyword_analyzer, reference_chemicals: List[Dict] = None) -> Tuple[float, List[str], Dict]:
        """Гибридный расчет релевантности работы"""
        
        # 1. Базовый score по ключевым словам (существующий подход)
        base_score, matched_keywords = calculate_enhanced_relevance(work, keywords, keyword_analyzer)
        keyword_score = base_score
        
        # 2. Химический score
        chemical_score, chemical_matches = self._calculate_chemical_score(
            work, keywords, reference_chemicals
        )
        
        # 3. Семантический score
        semantic_score, semantic_info = self._calculate_semantic_score(work)
        
        # 4. Recency score
        recency_score = self._calculate_recency_score(work)
        
        # 5. Комбинированный score с весами
        total_score = (
            self.keyword_weight * keyword_score +
            self.chemical_weight * chemical_score +
            self.semantic_weight * semantic_score +
            self.recency_weight * recency_score
        )
        
        # Собираем информацию о матчах
        all_matches = matched_keywords.copy()
        if chemical_matches:
            all_matches.extend([f"⚗️ {cm}" for cm in chemical_matches])
        
        additional_info = {
            'keyword_score': keyword_score,
            'chemical_score': chemical_score,
            'semantic_score': semantic_score,
            'recency_score': recency_score,
            'chemical_matches': chemical_matches,
            'semantic_info': semantic_info,
        }
        
        return total_score, all_matches, additional_info
    
    def _calculate_chemical_score(self, work: dict, keywords: Dict[str, float], 
                                 reference_chemicals: List[Dict] = None) -> Tuple[float, List[str]]:
        """Расчет химического score"""
        title = work.get('title', '')
        abstract = work.get('abstract', '')
        
        if not title:
            return 0.0, []
        
        # Извлекаем химические формулы из работы
        formulas = self.chemical_analyzer.extract_chemical_entities(title)
        if abstract:
            formulas.extend(self.chemical_analyzer.extract_chemical_entities(abstract))
        
        if not formulas:
            return 0.0, []
        
        chemical_score = 0.0
        chemical_matches = []
        
        # Вес за наличие химических формул
        for formula_info in formulas:
            formula_type = formula_info['type']
            formula = formula_info['formula']
            
            # Базовый вес за тип формулы
            type_weight = self.chemical_score_config.get(formula_type, 1.0)
            chemical_score += type_weight
            chemical_matches.append(f"{formula} ({formula_type})")
            
            # Дополнительный вес за совпадение элементов с ключевыми словами
            elements = formula_info.get('elements', [])
            for element in elements:
                element_lower = element.lower()
                # Проверяем, есть ли элемент в ключевых словах
                for keyword in keywords.keys():
                    if element_lower in keyword.lower():
                        chemical_score += self.chemical_score_config['element_match']
                        chemical_matches.append(f"Element: {element}")
                        break
        
        # Сравнение с reference chemicals (если есть)
        if reference_chemicals:
            for ref_chem in reference_chemicals:
                for work_formula_info in formulas:
                    similarity = self.chemical_analyzer.calculate_chemical_similarity(
                        ref_chem.get('formula', ''),
                        work_formula_info.get('formula', '')
                    )
                    if similarity > 0.3:  # Порог схожести
                        chemical_score += similarity * self.chemical_score_config['family_similarity']
                        chemical_matches.append(f"Similar to: {ref_chem.get('formula', '')}")
        
        return min(chemical_score, 10.0), chemical_matches
    
    def _calculate_semantic_score(self, work: dict) -> Tuple[float, Dict]:
        """Расчет семантического score"""
        title = work.get('title', '')
        abstract = work.get('abstract', '')
        
        if not title:
            return 0.0, {}
        
        # Анализ семантического контекста
        semantic_info = self.semantic_analyzer.analyze_materials_context(title)
        if abstract:
            abstract_info = self.semantic_analyzer.analyze_materials_context(abstract)
            # Объединяем информацию
            for key in semantic_info:
                if key in abstract_info:
                    semantic_info[key] = list(set(semantic_info[key] + abstract_info[key]))
        
        # Расчет веса на основе исследовательского фокуса
        focus_weight = self.semantic_analyzer.get_research_focus_weight(
            semantic_info.get('chemical_context', [])
        )
        
        # Базовый score за наличие семантической информации
        semantic_score = 0.0
        
        if semantic_info.get('material_types'):
            semantic_score += len(semantic_info['material_types']) * 0.5
        
        if semantic_info.get('properties_studied'):
            semantic_score += len(semantic_info['properties_studied']) * 0.7
        
        if semantic_info.get('applications'):
            semantic_score += len(semantic_info['applications']) * 0.8
        
        # Умножаем на вес фокуса
        semantic_score *= focus_weight
        
        return min(semantic_score, 5.0), semantic_info
    
    def _calculate_recency_score(self, work: dict) -> float:
        """Расчет score за новизну"""
        publication_year = work.get('publication_year', 0)
        current_year = datetime.now().year
        
        if publication_year <= 0:
            return 0.0
        
        # Нормализованный score: более свежие статьи получают более высокий score
        year_diff = current_year - publication_year
        
        if year_diff <= 1:
            return 1.0
        elif year_diff <= 3:
            return 0.8
        elif year_diff <= 5:
            return 0.5
        else:
            return 0.2

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
            'compound': 1.5,  # Составные слова важнее
            'scientific': 0.7  # Научные стоп-слова менее важны
        }
    
    def extract_weighted_keywords(self, titles: List[str]) -> Dict[str, float]:
        """Извлечение ключевых слов с весами"""
        weighted_counter = Counter()
        
        for title in titles:
            if not title:
                continue
                
            # Извлекаем все типы слов
            content_words = self.title_analyzer.preprocess_content_words(title)
            compound_words = self.title_analyzer.extract_compound_words(title)
            
            # Учитываем веса
            for word_info in content_words:
                lemma = word_info['lemma']
                if lemma:
                    weighted_counter[lemma] += self.weights['content']
            
            for word_info in compound_words:
                lemma = word_info['lemma']
                if lemma:
                    weighted_counter[lemma] += self.weights['compound']
        
        return weighted_counter

# ============================================================================
# ОБНОВЛЕННЫЕ ФУНКЦИИ ДЛЯ РАСЧЕТА РЕЛЕВАНТНОСТИ С ХИМИЧЕСКИМ АНАЛИЗОМ
# ============================================================================

def calculate_enhanced_relevance(work: dict, keywords: Dict[str, float], 
                                 analyzer: TitleKeywordsAnalyzer) -> Tuple[float, List[str]]:
    """Расчет релевантности с учетом семантической близости"""
    
    title = work.get('title', '').lower()
    abstract = work.get('abstract', '').lower()
    
    if not title:
        return 0.0, []
    
    score = 0.0
    matched_keywords = []
    
    # Извлекаем слова из заголовка анализируемой работы
    title_words = analyzer.preprocess_content_words(title)
    compound_words = analyzer.extract_compound_words(title)
    
    title_lemmas = {w['lemma'] for w in title_words}
    compound_lemmas = {w['lemma'] for w in compound_words}
    all_title_lemmas = title_lemmas.union(compound_lemmas)
    
    # Проверяем каждое ключевое слово
    for keyword, weight in keywords.items():
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
    
    # Дополнительные бонусы
    compound_words_list = analyzer.extract_compound_words(title)
    if compound_words_list:
        score += len(compound_words_list) * 0.5
    
    return score, matched_keywords

def calculate_enhanced_relevance_with_chemistry(work: dict, keywords: Dict[str, float], 
                                                analyzer: TitleKeywordsAnalyzer, 
                                                chemical_analyzer: ChemicalFormulaAnalyzer,
                                                reference_chemicals: List[Dict] = None) -> Tuple[float, List[str], Dict]:
    """Расчет релевантности с химическим анализом"""
    
    # Базовый score по ключевым словам
    score, matched_keywords = calculate_enhanced_relevance(work, keywords, analyzer)
    
    # ДОПОЛНИТЕЛЬНЫЙ ВЕС за химические совпадения
    title = work.get('title', '')
    abstract = work.get('abstract', '')
    
    chemical_info = {
        'formulas': [],
        'elements': set(),
        'chemical_score': 0.0,
        'chemical_matches': []
    }
    
    if title:
        formulas = chemical_analyzer.extract_chemical_entities(title)
        chemical_info['formulas'] = [f['formula'] for f in formulas]
        
        for formula_info in formulas:
            # Вес зависит от типа формулы
            if formula_info['type'] == 'simple_formula':
                score += 2.0
                chemical_info['chemical_score'] += 2.0
            elif formula_info['type'] == 'doped_formula':
                score += 3.0  # Более специфичные формулы важнее
                chemical_info['chemical_score'] += 3.0
            elif formula_info['type'] == 'complex_formula':
                score += 2.5
                chemical_info['chemical_score'] += 2.5
            elif formula_info['type'] == 'indexed_formula':
                score += 2.8
                chemical_info['chemical_score'] += 2.8
            
            chemical_info['chemical_matches'].append(f"{formula_info['formula']} ({formula_info['type']})")
            
            # Дополнительный вес за совпадение элементов
            elements = formula_info.get('elements', [])
            chemical_info['elements'].update(elements)
            
            for element in elements:
                element_lower = element.lower()
                # Проверяем, есть ли элемент в ключевых словах
                for keyword in keywords.keys():
                    if element_lower in keyword.lower():
                        score += 1.5
                        chemical_info['chemical_score'] += 1.5
                        chemical_info['chemical_matches'].append(f"Element: {element}")
                        break
    
    # Сравнение с reference chemicals (если есть)
    if reference_chemicals and chemical_info['formulas']:
        for ref_chem in reference_chemicals:
            ref_formula = ref_chem.get('formula', '')
            for work_formula in chemical_info['formulas']:
                similarity = chemical_analyzer.calculate_chemical_similarity(ref_formula, work_formula)
                if similarity > 0.3:  # Порог схожести
                    bonus = similarity * 2.0
                    score += bonus
                    chemical_info['chemical_score'] += bonus
                    chemical_info['chemical_matches'].append(f"Similar to: {ref_formula}")
    
    return score, matched_keywords, chemical_info

# ============================================================================
# ФУНКЦИИ ФИЛЬТРАЦИИ И АНАЛИЗА
# ============================================================================

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
    max_citations: int = 10,
    max_works: int = 2000,
    top_n: int = 100,
    year_filter: List[int] = None,
    citation_ranges: List[Tuple[int, int]] = None
) -> List[dict]:
    """
    Analyze works for a specific topic with filtering of input DOIs and duplicate titles.
    ИСПОЛЬЗУЕТ УЛУЧШЕННЫЙ АЛГОРИТМ С СЕМАНТИЧЕСКОЙ БЛИЗОСТЬЮ И ХИМИЧЕСКИМ АНАЛИЗОМ.
    
    Args:
        topic_id: OpenAlex topic ID
        keywords: List of keywords for relevance scoring
        max_citations: Maximum citations threshold (deprecated, use citation_ranges instead)
        max_works: Maximum number of works to fetch from OpenAlex
        top_n: Number of top results to return
        year_filter: List of years to filter by
        citation_ranges: List of citation ranges as tuples (min, max)
        
    Returns:
        List of enriched work dictionaries with duplicates removed
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
    chemical_analyzer = ChemicalFormulaAnalyzer()
    hybrid_scorer = HybridRelevanceScorer()
    
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
    
    # Извлекаем химические формулы из исходных работ для сравнения
    reference_chemicals = []
    if 'works_data' in st.session_state:
        for work in st.session_state.works_data:
            title = work.get('title', '')
            if title:
                formulas = chemical_analyzer.extract_chemical_entities(title)
                reference_chemicals.extend(formulas)
    
    # Track duplicate titles to keep only one version (with highest DOI number)
    title_to_work_map = {}
    
    with st.spinner(f"Analyzing {len(works)} works with enhanced hybrid algorithm..."):
        analyzed = []
        
        for work in works:
            cited_by_count = work.get('cited_by_count', 0)
            publication_year = work.get('publication_year', 0)
            
            # Filter by years
            if publication_year not in year_filter:
                continue
            
            # Filter by citations (ranges)
            in_range = False
            for min_cit, max_cit in citation_ranges:
                if min_cit <= cited_by_count <= max_cit:
                    in_range = True
                    break
            
            if not in_range:
                continue
            
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
            
            # Calculate enhanced relevance score with chemistry
            relevance_score, matched_keywords, chemical_info = calculate_enhanced_relevance_with_chemistry(
                work, normalized_keywords, title_analyzer, chemical_analyzer, reference_chemicals
            )
            
            # Альтернативно используем гибридный скорер
            hybrid_score, hybrid_matches, additional_info = hybrid_scorer.score_work(
                work, normalized_keywords, title_analyzer, reference_chemicals
            )
            
            # Используем максимальный score из двух методов
            final_score = max(relevance_score, hybrid_score)
            
            if final_score > 0:
                enriched = enrich_work_data(work)
                enriched.update({
                    'relevance_score': final_score,
                    'matched_keywords': matched_keywords,
                    'hybrid_score': hybrid_score,
                    'chemical_score': additional_info.get('chemical_score', 0),
                    'semantic_score': additional_info.get('semantic_score', 0),
                    'chemical_formulas': chemical_info.get('formulas', []),
                    'chemical_elements': list(chemical_info.get('elements', set())),
                    'analysis_time': datetime.now().isoformat()
                })
                
                # Объединяем матчи из обоих методов
                all_matches = list(set(matched_keywords + hybrid_matches))
                enriched['all_matches'] = all_matches
                
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
        
        # Многокритериальная сортировка с учетом химического score
        analyzed.sort(key=lambda x: (
            -x['relevance_score'],          # 1. Общая релевантность
            -x.get('chemical_score', 0),    # 2. Химический score
            -x.get('publication_year', 0),  # 3. Новизна
            -x.get('cited_by_count', 0)     # 4. Цитирования (в пределах диапазона)
        ))
        
        # Apply top_n limit
        result = analyzed[:top_n]
        
        # Log summary statistics
        logger.info(f"Found {len(result)} unique works after filtering")
        logger.info(f"Removed {len(works) - len(analyzed)} works due to filters")
        if len(analyzed) > len(result):
            logger.info(f"Limited from {len(analyzed)} to {len(result)} works by top_n parameter")
        
        # Сохраняем химическую статистику в сессии
        if result:
            chemical_formulas_count = sum(len(w.get('chemical_formulas', [])) for w in result)
            chemical_elements_count = sum(len(w.get('chemical_elements', [])) for w in result)
            st.session_state['chemical_stats'] = {
                'total_formulas': chemical_formulas_count,
                'unique_elements': len(set().union(*[set(w.get('chemical_elements', [])) for w in result])),
                'papers_with_formulas': sum(1 for w in result if w.get('chemical_formulas')),
                'avg_chemical_score': np.mean([w.get('chemical_score', 0) for w in result])
            }
        
        return result

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
    
    # Стиль для химической информации
    chemical_style = ParagraphStyle(
        'CustomChemical',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#8E44AD'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
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
    story.append(Paragraph("Advanced Chemistry-Aware Paper Analysis", subtitle_style))
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
        
        # Химическая статистика
        chemical_formulas_count = sum(len(w.get('chemical_formulas', [])) for w in data)
        papers_with_chemistry = sum(1 for w in data if w.get('chemical_formulas'))
        
        stats_text = f"""
        Average citations: {avg_citations:.1f} | 
        Open Access papers: {oa_count} | 
        Recent papers (≤2 years): {recent_count} |
        Papers with chemical formulas: {papers_with_chemistry}
        """
        story.append(Paragraph(stats_text, meta_style))
    
    story.append(Spacer(1, 1.5*cm))
    
    # Копирайт информация
    story.append(Paragraph("© CTA - Chimica Techno Acta", footer_style))
    story.append(Paragraph("https://chimicatechnoacta.ru", footer_style))
    story.append(Paragraph("Developed by daM© with Advanced Chemistry Analysis", footer_style))
    
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
        ["Papers Found", len(data)],
        ["Chemistry-Aware Analysis", "Enabled"]
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
        chemical_score = work.get('chemical_score', 0)
        
        metrics_text = f"""
        <b>Citations:</b> {citations} | 
        <b>Year:</b> {year} | 
        <b>Relevance Score:</b> {relevance}/10 | 
        <b>Chemistry Score:</b> {chemical_score:.1f} | 
        <b>Journal:</b> {journal} | 
        <b>Open Access:</b> {'Yes' if work.get('is_oa') else 'No'}
        """
        story.append(Paragraph(metrics_text, metrics_style))
        
        # Химическая информация (если есть)
        chemical_formulas = work.get('chemical_formulas', [])
        chemical_elements = work.get('chemical_elements', [])
        
        if chemical_formulas:
            formulas_text = ', '.join(chemical_formulas[:3])
            if len(chemical_formulas) > 3:
                formulas_text += f' +{len(chemical_formulas)-3} more'
            
            elements_text = ', '.join(chemical_elements[:5]) if chemical_elements else 'None'
            
            chem_text = f"""
            <b>Chemical Formulas:</b> {clean_text(formulas_text)} | 
            <b>Elements:</b> {clean_text(elements_text)}
            """
            story.append(Paragraph(chem_text, chemical_style))
        
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
        chemical_scores = [w.get('chemical_score', 0) for w in data]
        relevance_scores = [w.get('relevance_score', 0) for w in data]
        
        if citations_list and years_list:
            # Химическая статистика
            papers_with_chemistry = sum(1 for w in data if w.get('chemical_formulas'))
            total_formulas = sum(len(w.get('chemical_formulas', [])) for w in data)
            all_elements = set()
            for w in data:
                all_elements.update(w.get('chemical_elements', []))
            
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
                ["Average Relevance", f"{np.mean(relevance_scores):.2f}/10"],
                ["Papers with Chemistry", f"{papers_with_chemistry} ({papers_with_chemistry/len(data)*100:.1f}%)"],
                ["Total Chemical Formulas", total_formulas],
                ["Unique Elements", len(all_elements)],
                ["Average Chemistry Score", f"{np.mean(chemical_scores):.2f}"]
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
        f"Chemistry-aware analysis identified chemical formulas in papers, providing enhanced relevance scoring.",
        "Papers with low citation counts often represent emerging ideas or niche research areas.",
        "Consider these papers for:",
        "• Literature reviews of emerging topics",
        "• Identifying research gaps",
        "• Finding novel methodologies",
        "• Cross-disciplinary connections",
        "• Chemical compound analysis and similarity"
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
        "This report was generated automatically by CTA Article Recommender Pro with Advanced Chemistry Analysis.",
        "All data is sourced from OpenAlex API and is subject to their terms of use.",
        "For the most current data, please visit the original sources via the provided DOIs.",
        "Citation counts are as of the report generation date and may change over time.",
        "Chemical formula detection enhances relevance scoring for materials science and chemistry papers."
    ]
    
    for note in final_notes:
        story.append(Paragraph(f"• {clean_text(note)}", details_style))
    
    # Нижний колонтитул на последней странице
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("© CTA Article Recommender Pro - https://chimicatechnoacta.ru", footer_style))
    story.append(Paragraph("Advanced Chemistry-Aware Analysis Engine", footer_style))
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
    output.append("Advanced Chemistry-Aware Paper Analysis Report")
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
        
        # Химическая статистика
        papers_with_chemistry = sum(1 for w in data if w.get('chemical_formulas'))
        total_formulas = sum(len(w.get('chemical_formulas', [])) for w in data)
        avg_chemical_score = np.mean([w.get('chemical_score', 0) for w in data])
        
        output.append(f"  Average citations: {avg_citations:.2f}")
        output.append(f"  Open Access papers: {oa_count}")
        output.append(f"  Recent papers (≤2 years): {recent_count}")
        output.append(f"  Papers with chemical formulas: {papers_with_chemistry}")
        output.append(f"  Total chemical formulas: {total_formulas}")
        output.append(f"  Average chemistry score: {avg_chemical_score:.2f}")
    
    output.append("")
    output.append("© CTA - Chemical Technology Acta")
    output.append("https://chimicatechnoacta.ru")
    output.append("Developed by daM© with Advanced Chemistry Analysis")
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
    output.append(f"  Chemistry-Aware Analysis: Enabled")
    
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
    
    # Группируем статьи по релевантности и химическому score
    high_relevance = [w for w in data if w.get('relevance_score', 0) >= 8]
    medium_relevance = [w for w in data if 5 <= w.get('relevance_score', 0) < 8]
    low_relevance = [w for w in data if w.get('relevance_score', 0) < 5]
    
    high_chemistry = [w for w in data if w.get('chemical_score', 0) >= 3]
    medium_chemistry = [w for w in data if 1 <= w.get('chemical_score', 0) < 3]
    low_chemistry = [w for w in data if w.get('chemical_score', 0) < 1]
    
    output.append(f"  High Relevance (Score ≥ 8): {len(high_relevance)} papers")
    output.append(f"  Medium Relevance (5-7): {len(medium_relevance)} papers")
    output.append(f"  Low Relevance (Score < 5): {len(low_relevance)} papers")
    output.append("")
    output.append(f"  High Chemistry Score (≥3): {len(high_chemistry)} papers")
    output.append(f"  Medium Chemistry Score (1-3): {len(medium_chemistry)} papers")
    output.append(f"  Low Chemistry Score (<1): {len(low_chemistry)} papers")
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
        chemical_score = work.get('chemical_score', 0)
        relevance_stars = "★" * min(int(relevance_score), 5) + "☆" * max(5 - int(relevance_score), 0)
        chemistry_stars = "⚗️" * min(int(chemical_score), 5) + " " * max(5 - int(chemical_score), 0)
        
        output.append(f"PAPER #{i:03d}")
        output.append(f"Relevance: {relevance_score}/10 {relevance_stars}")
        output.append(f"Chemistry Score: {chemical_score:.1f} {chemistry_stars}")
        output.append("-" * 40)
        
        # Заголовок
        title = work.get('title', 'No title available')
        output.append(f"TITLE: {title}")
        
        # Химическая информация
        chemical_formulas = work.get('chemical_formulas', [])
        chemical_elements = work.get('chemical_elements', [])
        
        if chemical_formulas:
            formulas_text = ', '.join(chemical_formulas[:3])
            if len(chemical_formulas) > 3:
                formulas_text += f' (+{len(chemical_formulas)-3} more)'
            output.append(f"CHEMICAL FORMULAS: {formulas_text}")
        
        if chemical_elements:
            elements_text = ', '.join(chemical_elements[:5])
            if len(chemical_elements) > 5:
                elements_text += f' (+{len(chemical_elements)-5} more)'
            output.append(f"CHEMICAL ELEMENTS: {elements_text}")
        
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
        output.append(f"  • Hybrid Score: {work.get('hybrid_score', 0):.1f}")
        
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
        chemical_list = [w.get('chemical_score', 0) for w in data]
        
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
        
        if chemical_list:
            output.append("CHEMISTRY SCORE ANALYSIS:")
            output.append(f"  Average: {np.mean(chemical_list):.2f}")
            output.append(f"  Median: {np.median(chemical_list):.2f}")
            output.append(f"  Papers with formulas: {sum(1 for w in data if w.get('chemical_formulas'))}")
            output.append("")
            
            # Химическая статистика
            all_formulas = []
            all_elements = set()
            for w in data:
                all_formulas.extend(w.get('chemical_formulas', []))
                all_elements.update(w.get('chemical_elements', []))
            
            output.append(f"  Total chemical formulas: {len(all_formulas)}")
            output.append(f"  Unique chemical formulas: {len(set(all_formulas))}")
            output.append(f"  Unique elements: {len(all_elements)}")
            
            if all_formulas:
                formula_counts = Counter(all_formulas)
                output.append("  Most common formulas:")
                for formula, count in formula_counts.most_common(10):
                    output.append(f"    {formula:20}: {count:3d} papers")
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
        
        # Сортируем по релевантности, затем по химическому score
        sorted_data = sorted(data, key=lambda x: (-x.get('relevance_score', 0), 
                                                  -x.get('chemical_score', 0)))
        
        output.append("Highest Relevance & Chemistry Score:")
        for i, work in enumerate(sorted_data[:5], 1):
            title = work.get('title', '')[:70] + "..." if len(work.get('title', '')) > 70 else work.get('title', '')
            output.append(f"  {i}. {title}")
            output.append(f"     Year: {work.get('publication_year', 'N/A')}, "
                         f"Citations: {work.get('cited_by_count', 0)}, "
                         f"Score: {work.get('relevance_score', 0)}/10, "
                         f"Chemistry: {work.get('chemical_score', 0):.1f}")
        
        output.append("")
        output.append("Best Chemistry Matches:")
        # Берем статьи с ненулевыми химическими формулами
        chemistry_papers = [w for w in data if w.get('chemical_formulas')]
        if chemistry_papers:
            best_chemistry = sorted(chemistry_papers, key=lambda x: -x.get('chemical_score', 0))
            for i, work in enumerate(best_chemistry[:3], 1):
                title = work.get('title', '')[:70] + "..." if len(work.get('title', '')) > 70 else work.get('title', '')
                formulas = ', '.join(work.get('chemical_formulas', [])[:2])
                output.append(f"  {i}. {title}")
                output.append(f"     Formulas: {formulas}, "
                             f"Chemistry Score: {work.get('chemical_score', 0):.1f}")
        
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
        f"Chemistry-aware analysis enhanced relevance scoring for {sum(1 for w in data if w.get('chemical_formulas'))} papers.",
        "",
        "KEY INSIGHTS:",
        "• Papers with chemical formulas receive enhanced relevance scores",
        "• Low citation counts don't necessarily indicate low quality",
        "• Chemical similarity enhances cross-disciplinary discovery",
        "• Consider these for literature reviews and gap analysis",
        "• They may contain novel methodologies or cross-disciplinary insights",
        "",
        "RECOMMENDED ACTIONS:",
        "1. Review high-chemistry papers for material similarity",
        "2. Use as starting points for systematic reviews",
        "3. Identify research gaps and opportunities",
        "4. Track emerging chemical compounds in this field",
        "5. Leverage chemical formula matching for precision recommendations",
        "",
        "REPORT METADATA:",
        f"• Generated by: CTA Article Recommender Pro with Chemistry Analysis",
        f"• Report ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12].upper()}",
        f"• Data source: OpenAlex API",
        f"• Analysis date: {current_date}",
        f"• Chemistry analysis: Enabled with hybrid scoring",
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
        <span class="{'active' if current_step >= 4 else ''}">⚗️ Chemistry Results</span>
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
                if 'chemical_stats' in st.session_state:
                    del st.session_state['chemical_stats']
            
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
    """Создает компактную карточку результата с химической информацией"""
    citation_count = work.get('cited_by_count', 0)
    chemical_score = work.get('chemical_score', 0)
    
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
    
    # Определяем цвет баджа химического score
    if chemical_score >= 3:
        chem_color = "#8E44AD"
        chem_text = f"⚗️ {chemical_score:.1f}"
    elif chemical_score >= 1:
        chem_color = "#3498DB"
        chem_text = f"⚗️ {chemical_score:.1f}"
    else:
        chem_color = "#95A5A6"
        chem_text = f"⚗️ {chemical_score:.1f}"
    
    oa_badge = '🔓' if work.get('is_oa') else '🔒'
    doi_url = work.get('doi_url', '')
    title_text = work.get('title', 'No title')
    authors = ', '.join(work.get('authors', [])[:2])
    if len(work.get('authors', [])) > 2:
        authors += ' et al.'
    
    # Химические формулы (первые 2)
    chemical_formulas = work.get('chemical_formulas', [])
    chem_formulas_text = ''
    if chemical_formulas:
        formulas_display = ', '.join(chemical_formulas[:2])
        if len(chemical_formulas) > 2:
            formulas_display += f' +{len(chemical_formulas)-2}'
        chem_formulas_text = f'<div style="color: #8E44AD; font-size: 0.8rem; margin: 3px 0;">🧪 {formulas_display}</div>'
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <span style="font-weight: 600; color: #667eea; margin-right: 8px;">#{index}</span>
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                    {badge_text}
                </span>
                <span style="background: {chem_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 5px;">
                    {chem_text}
                </span>
                <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 5px;">
                    Score: {work.get('relevance_score', 0)}
                </span>
            </div>
            <span style="color: #666; font-size: 0.8rem;">{work.get('publication_year', '')}</span>
        </div>
        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 5px; line-height: 1.3;">{title_text}</div>
        {chem_formulas_text}
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
        
        # Информация о химическом анализе
        st.markdown("---")
        st.markdown("""
        <div class="info-message">
            <strong>⚗️ Chemistry-Aware Analysis</strong><br>
            The analysis now includes chemical formula detection and similarity scoring.
            Papers with chemical formulas will receive enhanced relevance scores.
        </div>
        """, unsafe_allow_html=True)
        
        # Кнопка запуска анализа
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚗️ Start Advanced Analysis", type="primary", use_container_width=True, key="start_analysis"):
                # Сбрасываем кэш предыдущих результатов
                if 'relevant_works' in st.session_state:
                    del st.session_state['relevant_works']
                if 'top_keywords' in st.session_state:
                    del st.session_state['top_keywords']
                if 'chemical_stats' in st.session_state:
                    del st.session_state['chemical_stats']
                
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
    
    # Анализ химических формул
    chemical_analyzer = ChemicalFormulaAnalyzer()
    chemical_stats = {
        'total_formulas': 0,
        'unique_elements': set(),
        'papers_with_formulas': 0
    }
    
    for work in works_data:
        title = work.get('title', '')
        if title:
            formulas = chemical_analyzer.extract_chemical_entities(title)
            if formulas:
                chemical_stats['total_formulas'] += len(formulas)
                chemical_stats['papers_with_formulas'] += 1
                for formula in formulas:
                    elements = formula.get('elements', [])
                    chemical_stats['unique_elements'].update(elements)
    
    # Сохранение результатов
    st.session_state.works_data = works_data
    st.session_state.topic_counter = topic_counter
    st.session_state.keyword_counter = keyword_counter
    st.session_state.successful = successful
    st.session_state.failed = failed
    st.session_state.chemical_stats = chemical_stats
    
    # Результаты анализа
    st.markdown(f"""
    <div class="info-message">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>✅ Analysis Complete!</strong><br>
                Successfully processed {successful} papers<br>
                Found chemical formulas in {chemical_stats['papers_with_formulas']} papers
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card_compact("Successful", successful, "✅")
    with col2:
        create_metric_card_compact("Failed", failed, "❌")
    with col3:
        create_metric_card_compact("Topics", len(topic_counter), "🏷️")
    with col4:
        create_metric_card_compact("Chemical", f"{chemical_stats['papers_with_formulas']}", "⚗️")
    
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
        <p style="margin: 5px 0; font-size: 0.9rem;">Choose a topic for advanced chemistry-aware analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.works_data:
        st.error("❌ No data available. Please start from Step 1.")
        return
    
    # Показываем химическую статистику
    if 'chemical_stats' in st.session_state:
        chem_stats = st.session_state.chemical_stats
        st.markdown(f"""
        <div class="info-message">
            <strong>⚗️ Chemical Analysis Summary</strong><br>
            Found {chem_stats['total_formulas']} chemical formulas in {chem_stats['papers_with_formulas']} papers<br>
            Unique elements: {len(chem_stats['unique_elements'])}
        </div>
        """, unsafe_allow_html=True)
    
    create_topic_selection_ui()

def step_results():
    """Шаг 4: Результаты (компактный)"""
    create_back_button()
    
    st.markdown("""
    <div class="step-card">
        <h3 style="margin: 0; font-size: 1.3rem;">⚗️ Step 4: Chemistry-Aware Results</h3>
        <p style="margin: 5px 0; font-size: 0.9rem;">Fresh papers with enhanced chemical analysis.</p>
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
        with st.spinner("Searching for fresh papers with advanced chemistry-aware algorithm..."):
            # Получаем топ-10 ключевых слов
            top_keywords = [kw for kw, _ in st.session_state.keyword_counter.most_common(10)]
            
            # Сохраняем ключевые слова в сессии
            st.session_state.top_keywords = top_keywords
            
            # Выполняем улучшенный анализ с химией
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
        chem_papers = sum(1 for w in relevant_works if w.get('chemical_formulas'))
        create_metric_card_compact("With Chemistry", chem_papers, "⚗️")
    
    # Химическая статистика (если есть)
    if 'chemical_stats' in st.session_state and relevant_works:
        chem_stats = st.session_state.chemical_stats
        st.markdown(f"""
        <div class="info-message">
            <strong>⚗️ Chemical Analysis Results</strong><br>
            Papers with chemical formulas: {chem_papers} ({chem_papers/len(relevant_works)*100:.1f}%)<br>
            Average chemistry score: {np.mean([w.get('chemical_score', 0) for w in relevant_works]):.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Показываем активные фильтры
    st.markdown(f"""
    <div style="margin: 10px 0; font-size: 0.85rem; color: #666;">
        <strong>Active filters:</strong> Years: {', '.join(map(str, selected_years))} | 
        Citation ranges: {format_citation_ranges(selected_ranges)} |
        Chemistry analysis: <span style="color: #8E44AD; font-weight: bold;">Enabled</span>
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
        st.markdown("<h4>⚗️ Recommended Papers (Chemistry-Enhanced):</h4>", unsafe_allow_html=True)
        
        for idx, work in enumerate(relevant_works[:10], 1):
            create_result_card_compact(work, idx)
        
        # Таблица для детального просмотра
        st.markdown("<h4>📋 Detailed View with Chemistry Scores:</h4>", unsafe_allow_html=True)
        
        display_data = []
        for i, work in enumerate(relevant_works, 1):
            doi_url = work.get('doi_url', '')
            title = work.get('title', '')
            chemical_formulas = work.get('chemical_formulas', [])
            
            formulas_display = ', '.join(chemical_formulas[:2]) if chemical_formulas else 'None'
            if len(chemical_formulas) > 2:
                formulas_display += f' +{len(chemical_formulas)-2}'
            
            display_data.append({
                '#': i,
                'Title': title[:60] + '...' if len(title) > 60 else title,
                'Citations': work.get('cited_by_count', 0),
                'Relevance': work.get('relevance_score', 0),
                'Chemistry': work.get('chemical_score', 0),
                'Formulas': formulas_display,
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
                ),
                "Chemistry": st.column_config.ProgressColumn(
                    "Chemistry",
                    help="Chemistry score (higher = more chemical formulas)",
                    format="%.1f",
                    min_value=0,
                    max_value=5
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
                file_name=f"chemistry_aware_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            excel_data = generate_excel(relevant_works)
            st.download_button(
                label="📈 Excel",
                data=excel_data,
                file_name=f"chemistry_aware_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            txt_data = generate_txt(relevant_works, st.session_state.get('selected_topic', 'Results'))
            st.download_button(
                label="📝 TXT",
                data=txt_data,
                file_name=f"chemistry_aware_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col4:
            pdf_data = generate_pdf(relevant_works[:50], st.session_state.get('selected_topic', 'Results'))
            st.download_button(
                label="📄 PDF",
                data=pdf_data,
                file_name=f"chemistry_aware_papers_{st.session_state.get('selected_topic', 'results').replace(' ', '_')}.pdf",
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
                          'successful', 'failed', 'dois', 'chemical_stats']:
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
    <h1 class="main-header">⚗️ CTA Article Recommender Pro</h1>
    <p style="font-size: 1rem; color: #666; margin-bottom: 1.5rem;">
    Advanced chemistry-aware paper discovery with AI-powered analysis
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
        <p>Advanced Chemistry-Aware Analysis Engine v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

