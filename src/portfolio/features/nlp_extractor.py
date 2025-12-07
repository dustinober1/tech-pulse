"""
Advanced NLP feature extraction for Tech-Pulse analysis.

This module provides comprehensive natural language processing feature extraction
capabilities for Hacker News posts and comments, including semantic analysis,
sentiment processing, topic modeling, and advanced linguistic features.
"""

import re
import string
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional NLP libraries with graceful fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    from wordfreq import zipf_frequency, top_n_list
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False


@dataclass
class NLPFeatures:
    """Container for extracted NLP features."""

    # Basic text features
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float

    # Lexical diversity
    unique_words: int
    lexical_diversity: float
    hapax_legomena: int  # Words appearing only once

    # Readability scores
    flesch_reading_ease: Optional[float] = None
    flesch_kincaid_grade: Optional[float] = None
    gunning_fog: Optional[float] = None
    coleman_liau: Optional[float] = None

    # Sentiment features
    sentiment_compound: Optional[float] = None
    sentiment_positive: Optional[float] = None
    sentiment_negative: Optional[float] = None
    sentiment_neutral: Optional[float] = None

    # POS features
    noun_ratio: Optional[float] = None
    verb_ratio: Optional[float] = None
    adj_ratio: Optional[float] = None
    adv_ratio: Optional[float] = None

    # Named entities
    entity_count: Optional[int] = None
    entity_types: Optional[Dict[str, int]] = None

    # Topic features
    topic_distribution: Optional[np.ndarray] = None
    dominant_topic: Optional[int] = None

    # Advanced features
    readability_index: Optional[float] = None
    formality_score: Optional[float] = None
    technical_density: Optional[float] = None


@dataclass
class ExtractionConfig:
    """Configuration for NLP feature extraction."""

    # Text preprocessing
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    min_word_length: int = 2
    max_word_length: int = 20

    # Feature selection
    include_sentiment: bool = True
    include_pos: bool = True
    include_entities: bool = True
    include_readability: bool = True
    include_topics: bool = True

    # Vectorization
    vectorizer_type: str = "tfidf"  # "tfidf", "count", "hash"
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 0.95

    # Topic modeling
    n_topics: int = 10
    topic_algorithm: str = "lda"  # "lda", "nmf", "svd"

    # Advanced options
    use_wordfreq: bool = True
    use_textstat: bool = True
    cache_features: bool = True


class AdvancedNLPExtractor:
    """
    Advanced NLP feature extractor for text analysis.

    Provides comprehensive feature extraction including:
    - Basic text statistics
    - Lexical diversity metrics
    - Readability scores
    - Sentiment analysis
    - Part-of-speech tagging
    - Named entity recognition
    - Topic modeling
    - Advanced linguistic features
    """

    def __init__(self,
                 config: Optional[ExtractionConfig] = None,
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize NLP extractor.

        Args:
            config: Extraction configuration
            spacy_model: spaCy model name to use
        """
        self.config = config or ExtractionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize NLP components
        self.spacy_nlp = None
        self.sentiment_analyzer = None
        self.lemmatizer = None
        self.stemmer = None
        self.vectorizer = None
        self.topic_model = None

        # Feature cache
        self.feature_cache = {} if self.config.cache_features else None

        # Initialize models
        self._initialize_models(spacy_model)

    def _initialize_models(self, spacy_model: str):
        """Initialize NLP models and components."""

        # Initialize spaCy
        if SPACY_AVAILABLE and self.config.include_entities:
            try:
                self.spacy_nlp = spacy.load(spacy_model, disable=["parser"])
                self.logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                self.logger.warning(f"spaCy model {spacy_model} not found, skipping entity extraction")
                self.config.include_entities = False

        # Initialize NLTK components
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('sentiment/vader_lexicon')
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('corpora/wordnet')
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                self.logger.info("Downloading required NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)

            # Initialize components
            if self.config.include_sentiment:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()

            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()

            # Initialize stopwords set
            self.stop_words = set(stopwords.words('english'))

        # Initialize vectorizer
        self._initialize_vectorizer()

        # Initialize topic model
        if self.config.include_topics:
            self._initialize_topic_model()

    def _initialize_vectorizer(self):
        """Initialize text vectorizer."""
        if self.config.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words='english' if self.config.remove_stopwords else None,
                lowercase=self.config.lowercase
            )
        elif self.config.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words='english' if self.config.remove_stopwords else None,
                lowercase=self.config.lowercase
            )
        elif self.config.vectorizer_type == "hash":
            self.vectorizer = HashingVectorizer(
                n_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english' if self.config.remove_stopwords else None,
                lowercase=self.config.lowercase
            )

    def _initialize_topic_model(self):
        """Initialize topic modeling algorithm."""
        if self.config.topic_algorithm == "lda":
            self.topic_model = LatentDirichletAllocation(
                n_components=self.config.n_topics,
                random_state=42,
                max_iter=10
            )
        elif self.config.topic_algorithm == "nmf":
            self.topic_model = NMF(
                n_components=self.config.n_topics,
                random_state=42,
                max_iter=200
            )
        elif self.config.topic_algorithm == "svd":
            self.topic_model = TruncatedSVD(
                n_components=self.config.n_topics,
                random_state=42
            )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove punctuation if specified
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers if specified
        if self.config.remove_numbers:
            text = re.sub(r'\d+', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_basic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text statistics."""
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0
            }

        # Basic counts
        char_count = len(text)
        words = text.split() if text else []
        word_count = len(words)

        # Sentence counting
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                sentence_count = len(sentences)
            except (LookupError, AttributeError, ImportError):
                # Fallback to simple sentence splitting
                sentences = re.split(r'[.!?]+', text)
                sentence_count = len([s for s in sentences if s.strip()])
        else:
            # Simple sentence splitting as fallback
            sentences = re.split(r'[.!?]+', text)
            sentence_count = len([s for s in sentences if s.strip()])

        # Average lengths
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length
        }

    def extract_lexical_features(self, text: str) -> Dict[str, Any]:
        """Extract lexical diversity features."""
        if not text:
            return {
                'unique_words': 0,
                'lexical_diversity': 0,
                'hapax_legomena': 0
            }

        words = text.lower().split()
        word_count = len(words)

        if word_count == 0:
            return {
                'unique_words': 0,
                'lexical_diversity': 0,
                'hapax_legomena': 0
            }

        # Unique words
        unique_words = len(set(words))

        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = unique_words / word_count

        # Hapax legomena (words appearing only once)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        hapax_legomena = len([word for word, freq in word_freq.items() if freq == 1])

        return {
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity,
            'hapax_legomena': hapax_legomena
        }

    def extract_readability_features(self, text: str) -> Dict[str, Any]:
        """Extract readability scores."""
        features = {}

        if not text:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'coleman_liau': 0,
                'readability_index': 0
            }

        if TEXTSTAT_AVAILABLE and self.config.use_textstat:
            try:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
                features['gunning_fog'] = textstat.gunning_fog(text)
                features['coleman_liau'] = textstat.coleman_liau_index(text)

                # Custom readability index
                avg_sentence_length = len(text.split()) / max(1, text.count('.') + text.count('!') + text.count('?'))
                syllable_count = textstat.syllable_count(text)
                word_count = len(text.split())

                if word_count > 0:
                    readability_index = 206.835 - 1.015 * (word_count / max(1, text.count('.') + text.count('!') + text.count('?'))) - 84.6 * (syllable_count / word_count)
                    features['readability_index'] = readability_index
            except Exception as e:
                self.logger.warning(f"Error calculating readability scores: {e}")

        return features

    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment analysis features."""
        features = {}

        if not text or not self.sentiment_analyzer:
            return {
                'sentiment_compound': 0,
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 1
            }

        try:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            features.update({
                'sentiment_compound': sentiment['compound'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu']
            })
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {e}")
            # Fallback to neutral sentiment
            features.update({
                'sentiment_compound': 0,
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 1
            })

        return features

    def extract_pos_features(self, text: str) -> Dict[str, Any]:
        """Extract part-of-speech features."""
        features = {}

        if not text or not NLTK_AVAILABLE:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0
            }

        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Count POS categories
            pos_counts = {
                'nouns': 0,
                'verbs': 0,
                'adjectives': 0,
                'adverbs': 0,
                'total': len(pos_tags)
            }

            for _, pos in pos_tags:
                if pos.startswith('NN'):  # Nouns
                    pos_counts['nouns'] += 1
                elif pos.startswith('VB'):  # Verbs
                    pos_counts['verbs'] += 1
                elif pos.startswith('JJ'):  # Adjectives
                    pos_counts['adjectives'] += 1
                elif pos.startswith('RB'):  # Adverbs
                    pos_counts['adverbs'] += 1

            # Calculate ratios
            if pos_counts['total'] > 0:
                features['noun_ratio'] = pos_counts['nouns'] / pos_counts['total']
                features['verb_ratio'] = pos_counts['verbs'] / pos_counts['total']
                features['adj_ratio'] = pos_counts['adjectives'] / pos_counts['total']
                features['adv_ratio'] = pos_counts['adverbs'] / pos_counts['total']
            else:
                features.update({
                    'noun_ratio': 0,
                    'verb_ratio': 0,
                    'adj_ratio': 0,
                    'adv_ratio': 0
                })
        except (LookupError, AttributeError, ImportError) as e:
            # Fallback if NLTK data not available
            self.logger.warning(f"NLTK not available for POS tagging: {e}")
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0
            }
        except Exception as e:
            self.logger.warning(f"Error in POS tagging: {e}")
            features.update({
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0
            })

        return features

    def extract_entity_features(self, text: str) -> Dict[str, Any]:
        """Extract named entity features."""
        features = {}

        if not text or not self.spacy_nlp:
            return {
                'entity_count': 0,
                'entity_types': {}
            }

        try:
            doc = self.spacy_nlp(text)

            # Count entities
            entities = list(doc.ents)
            features['entity_count'] = len(entities)

            # Count entity types
            entity_types = {}
            for ent in entities:
                entity_type = ent.label_
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            features['entity_types'] = entity_types

        except Exception as e:
            self.logger.warning(f"Error in entity extraction: {e}")
            features.update({
                'entity_count': 0,
                'entity_types': {}
            })

        return features

    def extract_technical_density(self, text: str) -> float:
        """Calculate technical density indicator."""
        if not text:
            return 0.0

        # Technical terms commonly found in tech discussions
        technical_terms = {
            'api', 'algorithm', 'architecture', 'backend', 'bug', 'code', 'database',
            'debug', 'deploy', 'development', 'framework', 'frontend', 'function',
            'git', 'github', 'interface', 'javascript', 'library', 'method', 'node',
            'python', 'react', 'repository', 'server', 'software', 'system', 'testing',
            'ui', 'ux', 'variable', 'web', 'programming', 'developer', 'stack',
            'scalability', 'performance', 'optimization', 'security', 'authentication'
        }

        words = text.lower().split()
        if not words:
            return 0.0

        technical_count = sum(1 for word in words if word in technical_terms)
        return technical_count / len(words)

    def extract_formality_score(self, text: str) -> float:
        """Calculate text formality score."""
        if not text:
            return 0.0

        # Informal indicators
        informal_indicators = {
            'lol', 'omg', 'btw', 'idk', 'imo', 'imho', 'tbh', 'smh', 'fyi', 'tldr',
            'gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'nah', 'yep', 'nope'
        }

        # Formal indicators
        formal_indicators = {
            'therefore', 'furthermore', 'consequently', 'nevertheless', 'however',
            'additionally', 'moreover', 'specifically', 'particular', 'significant',
            'substantial', 'considerable', 'appropriate', 'sufficient', 'necessary'
        }

        words = text.lower().split()
        if not words:
            return 0.0

        informal_count = sum(1 for word in words if word in informal_indicators)
        formal_count = sum(1 for word in words if word in formal_indicators)

        # Calculate formality score (0 = very informal, 1 = very formal)
        total_indicators = informal_count + formal_count
        if total_indicators == 0:
            return 0.5  # Neutral if no indicators found

        return formal_count / total_indicators

    def fit_topic_model(self, texts: List[str]) -> None:
        """
        Fit topic model on corpus of texts.

        Args:
            texts: List of texts for topic modeling
        """
        if not texts or not self.vectorizer or not self.topic_model:
            self.logger.warning("Cannot fit topic model: no texts or models available")
            return

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]  # Remove empty texts

        if not processed_texts:
            self.logger.warning("No valid texts after preprocessing")
            return

        # Fit vectorizer
        try:
            doc_term_matrix = self.vectorizer.fit_transform(processed_texts)

            # Fit topic model
            self.topic_model.fit(doc_term_matrix)
            self.logger.info(f"Topic model fitted with {len(processed_texts)} documents")

        except Exception as e:
            self.logger.error(f"Error fitting topic model: {e}")

    def extract_topic_features(self, text: str) -> Dict[str, Any]:
        """Extract topic modeling features."""
        features = {}

        if not text or not self.vectorizer or not self.topic_model:
            return {
                'topic_distribution': np.zeros(self.config.n_topics),
                'dominant_topic': 0
            }

        try:
            # Vectorize text
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {
                    'topic_distribution': np.zeros(self.config.n_topics),
                    'dominant_topic': 0
                }

            text_vector = self.vectorizer.transform([processed_text])

            # Get topic distribution
            topic_dist = self.topic_model.transform(text_vector)[0]
            dominant_topic = np.argmax(topic_dist)

            features['topic_distribution'] = topic_dist
            features['dominant_topic'] = int(dominant_topic)

        except Exception as e:
            self.logger.warning(f"Error in topic extraction: {e}")
            features.update({
                'topic_distribution': np.zeros(self.config.n_topics),
                'dominant_topic': 0
            })

        return features

    def extract_all_features(self, text: str, use_cache: bool = True) -> NLPFeatures:
        """
        Extract all available NLP features from text.

        Args:
            text: Input text
            use_cache: Whether to use cached features

        Returns:
            NLPFeatures object with all extracted features
        """
        # Check cache
        if use_cache and self.feature_cache is not None and text in self.feature_cache:
            return self.feature_cache[text]

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Extract feature groups
        basic_features = self.extract_basic_features(processed_text)
        lexical_features = self.extract_lexical_features(processed_text)

        readability_features = {}
        if self.config.include_readability:
            readability_features = self.extract_readability_features(text)

        sentiment_features = {}
        if self.config.include_sentiment:
            sentiment_features = self.extract_sentiment_features(text)

        pos_features = {}
        if self.config.include_pos:
            pos_features = self.extract_pos_features(text)

        entity_features = {}
        if self.config.include_entities:
            entity_features = self.extract_entity_features(text)

        topic_features = {}
        if self.config.include_topics:
            topic_features = self.extract_topic_features(text)

        # Advanced features
        technical_density = self.extract_technical_density(processed_text)
        formality_score = self.extract_formality_score(processed_text)

        # Create NLPFeatures object
        nlp_features = NLPFeatures(
            **basic_features,
            **lexical_features,
            **readability_features,
            **sentiment_features,
            **pos_features,
            entity_count=entity_features.get('entity_count'),
            entity_types=entity_features.get('entity_types'),
            **topic_features,
            technical_density=technical_density,
            formality_score=formality_score
        )

        # Cache features
        if use_cache and self.feature_cache is not None:
            self.feature_cache[text] = nlp_features

        return nlp_features

    def extract_features_batch(self,
                              texts: List[str],
                              fit_topics: bool = False) -> List[NLPFeatures]:
        """
        Extract features from multiple texts.

        Args:
            texts: List of input texts
            fit_topics: Whether to fit topic model on this batch

        Returns:
            List of NLPFeatures objects
        """
        # Fit topic model if requested
        if fit_topics and self.config.include_topics:
            self.fit_topic_model(texts)

        # Extract features for each text
        features = []
        for text in texts:
            try:
                feature = self.extract_all_features(text)
                features.append(feature)
            except Exception as e:
                self.logger.error(f"Error extracting features from text: {e}")
                # Create empty feature object as fallback
                features.append(NLPFeatures(
                    char_count=0, word_count=0, sentence_count=0,
                    avg_word_length=0, avg_sentence_length=0,
                    unique_words=0, lexical_diversity=0, hapax_legomena=0
                ))

        return features

    def get_feature_vector(self, features: NLPFeatures) -> np.ndarray:
        """
        Convert NLPFeatures to numerical vector.

        Args:
            features: NLPFeatures object

        Returns:
            Numerical feature vector
        """
        # Basic features
        vector = [
            features.char_count,
            features.word_count,
            features.sentence_count,
            features.avg_word_length,
            features.avg_sentence_length,
            features.unique_words,
            features.lexical_diversity,
            features.hapax_legomena
        ]

        # Readability features
        if features.flesch_reading_ease is not None:
            vector.extend([
                features.flesch_reading_ease,
                features.flesch_kincaid_grade or 0,
                features.gunning_fog or 0,
                features.coleman_liau or 0
            ])

        # Sentiment features
        if features.sentiment_compound is not None:
            vector.extend([
                features.sentiment_compound,
                features.sentiment_positive,
                features.sentiment_negative,
                features.sentiment_neutral
            ])

        # POS features
        if features.noun_ratio is not None:
            vector.extend([
                features.noun_ratio,
                features.verb_ratio,
                features.adj_ratio,
                features.adv_ratio
            ])

        # Entity features
        if features.entity_count is not None:
            vector.append(features.entity_count)

        # Topic features
        if features.topic_distribution is not None:
            vector.extend(features.topic_distribution.tolist())

        # Advanced features
        if features.technical_density is not None:
            vector.append(features.technical_density)

        if features.formality_score is not None:
            vector.append(features.formality_score)

        return np.array(vector)

    def get_feature_names(self) -> List[str]:
        """
        Get names of all features in the vector.

        Returns:
            List of feature names
        """
        names = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'unique_words', 'lexical_diversity',
            'hapax_legomena'
        ]

        # Readability features
        if self.config.include_readability:
            names.extend([
                'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                'coleman_liau'
            ])

        # Sentiment features
        if self.config.include_sentiment:
            names.extend([
                'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
                'sentiment_neutral'
            ])

        # POS features
        if self.config.include_pos:
            names.extend(['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio'])

        # Entity features
        if self.config.include_entities:
            names.append('entity_count')

        # Topic features
        if self.config.include_topics:
            for i in range(self.config.n_topics):
                names.append(f'topic_{i}')

        # Advanced features
        names.extend(['technical_density', 'formality_score'])

        return names

    def analyze_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Analyze similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary of similarity scores
        """
        if not text1 or not text2 or not self.vectorizer:
            return {
                'cosine_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'topic_similarity': 0.0
            }

        try:
            # Preprocess texts
            proc_text1 = self.preprocess_text(text1)
            proc_text2 = self.preprocess_text(text2)

            if not proc_text1 or not proc_text2:
                return {
                    'cosine_similarity': 0.0,
                    'jaccard_similarity': 0.0,
                    'topic_similarity': 0.0
                }

            # Vectorize texts - handle the case with only 2 documents
            try:
                vectors = self.vectorizer.fit_transform([proc_text1, proc_text2])
            except ValueError as e:
                # Handle the min_df/max_df issue with few documents
                if "max_df corresponds to < documents than min_df" in str(e):
                    # Create a temporary vectorizer with different settings
                    temp_vectorizer = TfidfVectorizer(
                        max_features=1000,
                        ngram_range=(1, 2),
                        min_df=1,  # Allow single occurrence
                        max_df=1.0,  # Allow all documents
                        stop_words='english'
                    )
                    vectors = temp_vectorizer.fit_transform([proc_text1, proc_text2])
                else:
                    raise

            # Cosine similarity
            cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            # Jaccard similarity
            words1 = set(proc_text1.split())
            words2 = set(proc_text2.split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_sim = intersection / union if union > 0 else 0.0

            # Topic similarity
            topic_sim = 0.0
            if self.topic_model and hasattr(self.topic_model, 'components_'):
                # Only use topic model if it's fitted
                try:
                    topic_dists = self.topic_model.transform(vectors)
                    topic_sim = cosine_similarity(topic_dists[0:1], topic_dists[1:2])[0][0]
                except Exception:
                    # If topic model fails to transform, skip it
                    topic_sim = 0.0

            return {
                'cosine_similarity': float(cos_sim),
                'jaccard_similarity': float(jaccard_sim),
                'topic_similarity': float(topic_sim)
            }

        except Exception as e:
            self.logger.error(f"Error in text similarity analysis: {e}")
            return {
                'cosine_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'topic_similarity': 0.0
            }

    def get_topic_keywords(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top keywords for each topic.

        Args:
            n_words: Number of top words to return per topic

        Returns:
            Dictionary mapping topic ID to list of (word, score) tuples
        """
        if not self.vectorizer or not self.topic_model:
            return {}

        try:
            feature_names = self.vectorizer.get_feature_names_out()

            topic_keywords = {}
            for topic_idx, topic in enumerate(self.topic_model.components_):
                # Get top words for this topic
                top_indices = topic.argsort()[-n_words:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_indices]
                topic_keywords[topic_idx] = top_words

            return topic_keywords

        except Exception as e:
            self.logger.error(f"Error getting topic keywords: {e}")
            return {}

    def export_features_to_dataframe(self,
                                   features_list: List[NLPFeatures],
                                   include_all: bool = True) -> pd.DataFrame:
        """
        Export features to pandas DataFrame.

        Args:
            features_list: List of NLPFeatures objects
            include_all: Whether to include all features including complex ones

        Returns:
            DataFrame with features
        """
        data = []

        for i, features in enumerate(features_list):
            row = {
                'index': i,
                'char_count': features.char_count,
                'word_count': features.word_count,
                'sentence_count': features.sentence_count,
                'avg_word_length': features.avg_word_length,
                'avg_sentence_length': features.avg_sentence_length,
                'unique_words': features.unique_words,
                'lexical_diversity': features.lexical_diversity,
                'hapax_legomena': features.hapax_legomena
            }

            if include_all:
                # Add optional features if available
                if features.flesch_reading_ease is not None:
                    row.update({
                        'flesch_reading_ease': features.flesch_reading_ease,
                        'flesch_kincaid_grade': features.flesch_kincaid_grade,
                        'gunning_fog': features.gunning_fog,
                        'coleman_liau': features.coleman_liau
                    })

                if features.sentiment_compound is not None:
                    row.update({
                        'sentiment_compound': features.sentiment_compound,
                        'sentiment_positive': features.sentiment_positive,
                        'sentiment_negative': features.sentiment_negative,
                        'sentiment_neutral': features.sentiment_neutral
                    })

                if features.noun_ratio is not None:
                    row.update({
                        'noun_ratio': features.noun_ratio,
                        'verb_ratio': features.verb_ratio,
                        'adj_ratio': features.adj_ratio,
                        'adv_ratio': features.adv_ratio
                    })

                if features.entity_count is not None:
                    row['entity_count'] = features.entity_count

                if features.dominant_topic is not None:
                    row['dominant_topic'] = features.dominant_topic

                if features.technical_density is not None:
                    row['technical_density'] = features.technical_density

                if features.formality_score is not None:
                    row['formality_score'] = features.formality_score

            data.append(row)

        return pd.DataFrame(data)