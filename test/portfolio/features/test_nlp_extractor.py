"""
Tests for advanced NLP feature extractor.

This module tests the comprehensive NLP feature extraction capabilities
including text preprocessing, feature extraction, and topic modeling.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from src.portfolio.features.nlp_extractor import (
    AdvancedNLPExtractor,
    NLPFeatures,
    ExtractionConfig
)


class TestExtractionConfig:
    """Test cases for ExtractionConfig"""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.lowercase == True
        assert config.remove_punctuation == True
        assert config.remove_numbers == False
        assert config.remove_stopwords == True
        assert config.min_word_length == 2
        assert config.max_word_length == 20

        assert config.include_sentiment == True
        assert config.include_pos == True
        assert config.include_entities == True
        assert config.include_readability == True
        assert config.include_topics == True

        assert config.vectorizer_type == "tfidf"
        assert config.max_features == 10000
        assert config.ngram_range == (1, 2)
        assert config.min_df == 2
        assert config.max_df == 0.95

        assert config.n_topics == 10
        assert config.topic_algorithm == "lda"

        assert config.use_wordfreq == True
        assert config.use_textstat == True
        assert config.cache_features == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExtractionConfig(
            lowercase=False,
            vectorizer_type="count",
            n_topics=5,
            include_sentiment=False
        )

        assert config.lowercase == False
        assert config.vectorizer_type == "count"
        assert config.n_topics == 5
        assert config.include_sentiment == False


class TestNLPFeatures:
    """Test cases for NLPFeatures dataclass"""

    def test_nlp_features_creation(self):
        """Test NLPFeatures object creation."""
        features = NLPFeatures(
            char_count=100,
            word_count=20,
            sentence_count=5,
            avg_word_length=5.0,
            avg_sentence_length=4.0,
            unique_words=15,
            lexical_diversity=0.75,
            hapax_legomena=8,
            sentiment_compound=0.5,
            noun_ratio=0.3
        )

        assert features.char_count == 100
        assert features.word_count == 20
        assert features.sentiment_compound == 0.5
        assert features.noun_ratio == 0.3

    def test_nlp_features_defaults(self):
        """Test NLPFeatures default values."""
        features = NLPFeatures(
            char_count=50,
            word_count=10,
            sentence_count=2,
            avg_word_length=5.0,
            avg_sentence_length=5.0,
            unique_words=8,
            lexical_diversity=0.8,
            hapax_legomena=3
        )

        # Optional features should default to None
        assert features.sentiment_compound is None
        assert features.flesch_reading_ease is None
        assert features.entity_count is None
        assert features.topic_distribution is None


class TestAdvancedNLPExtractor:
    """Test cases for AdvancedNLPExtractor"""

    @pytest.fixture
    def basic_config(self):
        """Create a basic extraction configuration."""
        return ExtractionConfig(
            include_sentiment=True,
            include_pos=True,
            include_entities=False,  # Disable to avoid spaCy dependency in tests
            include_readability=True,
            include_topics=True,
            vectorizer_type="tfidf",
            n_topics=5
        )

    @pytest.fixture
    def extractor(self, basic_config):
        """Create an NLP extractor instance."""
        with patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', True), \
             patch('src.portfolio.features.nlp_extractor.SPACY_AVAILABLE', False), \
             patch('src.portfolio.features.nlp_extractor.TEXTSTAT_AVAILABLE', False), \
             patch('src.portfolio.features.nlp_extractor.SentimentIntensityAnalyzer') as mock_sia, \
             patch('src.portfolio.features.nlp_extractor.word_tokenize') as mock_tokenize, \
             patch('src.portfolio.features.nlp_extractor.pos_tag') as mock_pos_tag:

            # Mock sentiment analyzer
            mock_sia_instance = Mock()
            mock_sia_instance.polarity_scores.return_value = {
                'neg': 0.1,
                'neu': 0.6,
                'pos': 0.3,
                'compound': 0.5
            }
            mock_sia.return_value = mock_sia_instance

            # Mock tokenization
            mock_tokenize.return_value = ['test', 'sentence']

            # Mock POS tagging
            mock_pos_tag.return_value = [('test', 'NN'), ('sentence', 'NN')]

            return AdvancedNLPExtractor(config=basic_config)

    def test_extractor_initialization(self, basic_config):
        """Test extractor initialization."""
        with patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', True), \
             patch('src.portfolio.features.nlp_extractor.SPACY_AVAILABLE', False):
            extractor = AdvancedNLPExtractor(config=basic_config)

            assert extractor.config == basic_config
            assert extractor.vectorizer is not None
            assert extractor.topic_model is not None
            assert extractor.spacy_nlp is None  # Disabled in config

    def test_preprocess_text(self, extractor):
        """Test text preprocessing."""
        # Test basic preprocessing
        text = "Hello WORLD! This is a test. https://example.com"
        processed = extractor.preprocess_text(text)

        assert "hello world" in processed.lower()
        assert "https://example.com" not in processed

        # Test empty text
        assert extractor.preprocess_text("") == ""
        assert extractor.preprocess_text(None) == ""

    def test_extract_basic_features(self, extractor):
        """Test basic feature extraction."""
        text = "This is a test sentence. This is another sentence."
        features = extractor.extract_basic_features(text)

        assert features['char_count'] > 0
        assert features['word_count'] > 0
        assert features['sentence_count'] == 2
        assert features['avg_word_length'] > 0
        assert features['avg_sentence_length'] > 0

        # Test empty text
        empty_features = extractor.extract_basic_features("")
        assert empty_features['char_count'] == 0
        assert empty_features['word_count'] == 0

    def test_extract_lexical_features(self, extractor):
        """Test lexical diversity features."""
        text = "The quick brown fox jumps over the lazy dog. The fox is quick."
        features = extractor.extract_lexical_features(text)

        assert features['unique_words'] > 0
        assert features['lexical_diversity'] > 0
        assert 0 <= features['lexical_diversity'] <= 1
        assert features['hapax_legomena'] >= 0

    def test_extract_sentiment_features(self, extractor):
        """Test sentiment feature extraction."""
        # Mock NLTK sentiment analyzer
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            'compound': 0.5,
            'pos': 0.3,
            'neg': 0.1,
            'neu': 0.6
        }

        extractor.sentiment_analyzer = mock_analyzer

        text = "I love this product! It's amazing."
        features = extractor.extract_sentiment_features(text)

        assert features['sentiment_compound'] == 0.5
        assert features['sentiment_positive'] == 0.3
        assert features['sentiment_negative'] == 0.1
        assert features['sentiment_neutral'] == 0.6

    def test_extract_pos_features(self, extractor):
        """Test part-of-speech feature extraction."""
        # Test with mocked NLTK functions at the extractor level
        config = ExtractionConfig(
            include_sentiment=False,
            include_pos=True,
            include_entities=False,
            include_readability=False,
            include_topics=False
        )

        with patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', True), \
             patch('src.portfolio.features.nlp_extractor.word_tokenize') as mock_tokenize, \
             patch('src.portfolio.features.nlp_extractor.pos_tag') as mock_pos_tag:

            mock_tokenize.return_value = ['The', 'quick', 'brown', 'fox', 'jumps']
            mock_pos_tag.return_value = [
                ('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'),
                ('fox', 'NN'), ('jumps', 'VBZ')
            ]

            extractor = AdvancedNLPExtractor(config=config)
            features = extractor.extract_pos_features("The quick brown fox jumps")

            # Check that all expected keys are present
            expected_keys = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']
            for key in expected_keys:
                assert key in features
                assert isinstance(features[key], (int, float))
                assert 0 <= features[key] <= 1

            # Specific assertions based on mocked POS tags
            assert features['noun_ratio'] == 0.2  # 1 noun / 5 total
            assert features['verb_ratio'] == 0.2   # 1 verb / 5 total
            assert features['adj_ratio'] == 0.4    # 2 adj / 5 total
            assert features['adv_ratio'] == 0.0    # 0 adverbs / 5 total

    def test_extract_pos_features_no_nltk(self, extractor):
        """Test POS features extraction without NLTK."""
        # Temporarily disable NLTK
        with patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', False):
            features = extractor.extract_pos_features("The quick brown fox jumps")

            # Should return zero values when NLTK is not available
            expected = {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0
            }
            assert features == expected

    def test_extract_entity_features(self, extractor):
        """Test named entity feature extraction."""
        # Test with spacy disabled
        features = extractor.extract_entity_features("Apple is a company")

        assert features['entity_count'] == 0
        assert features['entity_types'] == {}

    def test_extract_technical_density(self, extractor):
        """Test technical density calculation."""
        text = "This is about programming with Python and JavaScript APIs"
        density = extractor.extract_technical_density(text)

        assert 0 <= density <= 1
        assert density > 0  # Should detect some technical terms

        # Test empty text
        assert extractor.extract_technical_density("") == 0.0

    def test_extract_formality_score(self, extractor):
        """Test formality score calculation."""
        # Formal text
        formal_text = "Therefore, we must consider the implications carefully."
        formal_score = extractor.extract_formality_score(formal_text)

        # Informal text
        informal_text = "lol omg btw this is awesome"
        informal_score = extractor.extract_formality_score(informal_text)

        assert 0 <= formal_score <= 1
        assert 0 <= informal_score <= 1
        # Formal text should have higher score than informal
        assert formal_score > informal_score

    @patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', True)
    @patch('src.portfolio.features.nlp_extractor.SPACY_AVAILABLE', False)
    def test_extract_all_features(self):
        """Test comprehensive feature extraction."""
        config = ExtractionConfig(
            include_entities=False,  # Disable to avoid spaCy
            include_readability=False  # Disable to avoid textstat
        )
        extractor = AdvancedNLPExtractor(config=config)

        text = "This is a simple test sentence for feature extraction."
        features = extractor.extract_all_features(text)

        assert isinstance(features, NLPFeatures)
        assert features.char_count > 0
        assert features.word_count > 0
        assert features.unique_words > 0
        assert 0 <= features.lexical_diversity <= 1

    def test_extract_features_batch(self, extractor):
        """Test batch feature extraction."""
        texts = [
            "This is the first text.",
            "This is the second text.",
            "This is the third text."
        ]

        features_list = extractor.extract_features_batch(texts)

        assert len(features_list) == 3
        for features in features_list:
            assert isinstance(features, NLPFeatures)
            assert features.word_count > 0

    def test_get_feature_vector(self, extractor):
        """Test feature vector conversion."""
        # Create mock features
        features = NLPFeatures(
            char_count=100,
            word_count=20,
            sentence_count=2,
            avg_word_length=5.0,
            avg_sentence_length=10.0,
            unique_words=15,
            lexical_diversity=0.75,
            hapax_legomena=5,
            sentiment_compound=0.5,
            technical_density=0.3,
            formality_score=0.7
        )

        vector = extractor.get_feature_vector(features)

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 8  # Basic features
        assert vector[0] == 100  # char_count
        assert vector[1] == 20   # word_count

    def test_get_feature_names(self, extractor):
        """Test feature name generation."""
        names = extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 8  # Basic features
        assert 'char_count' in names
        assert 'word_count' in names
        assert 'lexical_diversity' in names

    def test_analyze_text_similarity(self, extractor):
        """Test text similarity analysis."""
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "AI and machine learning are related fields."
        text3 = "Cooking recipes can be found in many books."

        # Similar texts
        sim1_2 = extractor.analyze_text_similarity(text1, text2)

        # Different texts
        sim1_3 = extractor.analyze_text_similarity(text1, text3)

        assert sim1_2['cosine_similarity'] >= 0
        assert sim1_3['cosine_similarity'] >= 0
        assert sim1_2['cosine_similarity'] > sim1_3['cosine_similarity']

    def test_get_topic_keywords(self, extractor):
        """Test topic keyword extraction."""
        # Mock the vectorizer and topic model
        mock_vectorizer = Mock()
        mock_vectorizer.get_feature_names_out.return_value = [
            'machine', 'learning', 'data', 'science', 'python',
            'algorithm', 'model', 'neural', 'network', 'deep'
        ]

        extractor.vectorizer = mock_vectorizer

        # Mock topic model components
        mock_topic_model = Mock()
        mock_topic_model.components_ = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.1, 0.2, 0.4, 0.5, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4]
        ])
        extractor.topic_model = mock_topic_model

        keywords = extractor.get_topic_keywords(n_words=5)

        assert isinstance(keywords, dict)
        assert len(keywords) == 2  # 2 topics
        assert len(keywords[0]) == 5  # 5 keywords per topic

    def test_export_features_to_dataframe(self, extractor):
        """Test exporting features to DataFrame."""
        features_list = [
            NLPFeatures(
                char_count=100, word_count=20, sentence_count=2,
                avg_word_length=5.0, avg_sentence_length=10.0,
                unique_words=15, lexical_diversity=0.75, hapax_legomena=5
            ),
            NLPFeatures(
                char_count=50, word_count=10, sentence_count=1,
                avg_word_length=5.0, avg_sentence_length=10.0,
                unique_words=8, lexical_diversity=0.8, hapax_legomena=2
            )
        ]

        df = extractor.export_features_to_dataframe(features_list, include_all=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'char_count' in df.columns
        assert 'word_count' in df.columns
        assert df['char_count'].iloc[0] == 100
        assert df['char_count'].iloc[1] == 50

    @patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', False)
    @patch('src.portfolio.features.nlp_extractor.SPACY_AVAILABLE', False)
    def test_extractor_without_dependencies(self):
        """Test extractor behavior without optional dependencies."""
        config = ExtractionConfig(
            include_sentiment=False,  # Disable due to no NLTK
            include_pos=False,        # Disable due to no NLTK
            include_entities=False,   # Disable due to no spaCy
            include_readability=False  # Disable due to no textstat
        )

        extractor = AdvancedNLPExtractor(config=config)

        # Should still be able to extract basic features
        text = "This is a test text."
        basic_features = extractor.extract_basic_features(text)
        assert basic_features['word_count'] > 0

        # Should still extract lexical features
        lexical_features = extractor.extract_lexical_features(text)
        assert lexical_features['unique_words'] > 0

    def test_fit_topic_model(self, extractor):
        """Test topic model fitting."""
        texts = [
            "Machine learning algorithms and data science",
            "Python programming for data analysis",
            "Neural networks and deep learning models",
            "Software engineering best practices",
            "Database design and optimization"
        ]

        # Should not raise errors
        extractor.fit_topic_model(texts)

        # Verify model was fitted
        assert extractor.topic_model is not None

    def test_extract_topic_features(self, extractor):
        """Test topic feature extraction."""
        # Set up a simple topic model
        extractor.fit_topic_model([
            "Machine learning is about data and algorithms",
            "Python programming involves code and functions"
        ])

        text = "This text is about machine learning and data analysis"
        features = extractor.extract_topic_features(text)

        assert 'topic_distribution' in features
        assert 'dominant_topic' in features
        assert isinstance(features['topic_distribution'], np.ndarray)
        assert isinstance(features['dominant_topic'], int)

    def test_text_with_special_characters(self, extractor):
        """Test handling of text with special characters."""
        text = "Check out this link: https://example.com! Email me at test@example.com."

        # Should not crash
        processed = extractor.preprocess_text(text)
        assert "https://example.com" not in processed
        assert "test@example.com" not in processed

        # Should still extract basic features
        features = extractor.extract_basic_features(text)
        assert features['char_count'] > 0

    def test_very_long_text(self, extractor):
        """Test handling of very long text."""
        # Create a long text by repetition
        base_text = "This is a sentence for testing purposes."
        long_text = " ".join([base_text] * 100)

        # Should handle gracefully
        features = extractor.extract_all_features(long_text)
        assert features.char_count > 1000  # Reasonable expectation after preprocessing
        assert features.word_count > 100

    def test_unicode_text(self, extractor):
        """Test handling of Unicode characters."""
        text = "Testing with émojis 😀 and accénted characters like café and naïve."

        # Should handle Unicode properly
        features = extractor.extract_basic_features(text)
        assert features['char_count'] > 0
        assert features['word_count'] > 0

    def test_cache_functionality(self, extractor):
        """Test feature caching functionality."""
        text = "This is a test for caching functionality."

        # Extract features first time
        features1 = extractor.extract_all_features(text, use_cache=True)

        # Extract features second time (should use cache)
        features2 = extractor.extract_all_features(text, use_cache=True)

        # Should be identical
        assert features1.char_count == features2.char_count
        assert features1.word_count == features2.word_count

        # Test cache miss with different text
        features3 = extractor.extract_all_features("Different text", use_cache=True)
        assert features3.char_count != features1.char_count

    def test_configuration_variations(self, basic_config):
        """Test extractor with different configurations."""
        # Test with minimal configuration
        minimal_config = ExtractionConfig(
            include_sentiment=False,
            include_pos=False,
            include_entities=False,
            include_readability=False,
            include_topics=False,
            remove_stopwords=False,
            lowercase=False
        )

        with patch('src.portfolio.features.nlp_extractor.NLTK_AVAILABLE', True), \
             patch('src.portfolio.features.nlp_extractor.SPACY_AVAILABLE', False):
            extractor = AdvancedNLPExtractor(config=minimal_config)

            text = "The Quick Brown Fox Jumps Over The Lazy Dog"
            features = extractor.extract_all_features(text)

            # Should have basic features
            assert features.word_count > 0
            assert features.lexical_diversity > 0

            # Should not have sentiment features (disabled)
            assert features.sentiment_compound is None

            # Should preserve case (lowercase=False)
            assert "Quick" in text  # Text wasn't lowercased

    def test_error_handling(self, extractor):
        """Test error handling in extraction methods."""
        # Test with None input
        features = extractor.extract_all_features(None)
        assert features.char_count == 0
        assert features.word_count == 0

        # Test similarity with empty texts
        similarity = extractor.analyze_text_similarity("", "")
        assert similarity['cosine_similarity'] == 0.0

        # Test very short text
        short_text = "Hi"
        features = extractor.extract_all_features(short_text)
        assert features.word_count == 1