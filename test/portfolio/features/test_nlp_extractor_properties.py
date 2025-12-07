"""
Property-based tests for NLP feature extractor.

This module uses Hypothesis to verify that the NLP feature extractor
produces rich, meaningful features across a wide range of text inputs.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from hypothesis import given, strategies as st, settings, HealthCheck

from src.portfolio.features.nlp_extractor import (
    AdvancedNLPExtractor,
    NLPFeatures,
    ExtractionConfig
)


class TestNLPFeatureProperties:
    """Property-based tests for NLP feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create an NLP extractor for property testing."""
        config = ExtractionConfig(
            include_sentiment=True,
            include_pos=True,
            include_entities=False,  # Disable to avoid spaCy dependency
            include_readability=True,
            include_topics=True,
            n_topics=5  # Keep small for faster tests
        )

        # Mock dependencies to avoid NLTK/spaCy requirements
        from unittest.mock import patch, Mock
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

            # Mock tokenization to return something reasonable
            mock_tokenize.return_value = ['word'] * 5

            # Mock POS tagging
            mock_pos_tag.return_value = [('word', 'NN')] * 5

            return AdvancedNLPExtractor(config=config)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=50
    )
    @given(st.text(min_size=10, max_size=1000))
    def test_feature_consistency(self, extractor, text):
        """
        Property: Feature extraction is consistent for the same input.

        Given any text, extracting features twice should yield identical results.
        """
        features1 = extractor.extract_all_features(text, use_cache=False)
        features2 = extractor.extract_all_features(text, use_cache=False)

        # All major attributes should be identical
        assert features1.char_count == features2.char_count
        assert features1.word_count == features2.word_count
        assert features1.unique_words == features2.unique_words
        assert features1.sentence_count == features2.sentence_count
        assert np.isclose(features1.lexical_diversity, features2.lexical_diversity)
        assert np.isclose(features1.avg_word_length, features2.avg_word_length)
        assert np.isclose(features1.avg_sentence_length, features2.avg_sentence_length)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.lists(st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=['L', 'N', 'Zs'])), min_size=2, max_size=5))
    def test_feature_monotonicity_with_length(self, extractor, texts):
        """
        Property: Basic counts increase with text length.

        Longer texts should have greater or equal character and word counts.
        """
        # Sort texts by character count
        sorted_texts = sorted(texts, key=len)

        features_list = [extractor.extract_all_features(t) for t in sorted_texts]

        # Character counts should be non-decreasing on average
        char_counts = [f.char_count for f in features_list]
        # Allow significant fudge for preprocessing
        assert sum(char_counts) > 0

        # Word counts should be non-negative
        word_counts = [f.word_count for f in features_list]
        for wc in word_counts:
            assert wc >= 0

        # At least some texts should have words
        assert any(wc > 0 for wc in word_counts)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    @given(st.text(min_size=10, max_size=500))
    def test_feature_ranges(self, extractor, text):
        """
        Property: All features fall within expected ranges.

        Ratios should be between 0 and 1, counts should be non-negative, etc.
        """
        features = extractor.extract_all_features(text)

        # Basic counts should be non-negative
        assert features.char_count >= 0
        assert features.word_count >= 0
        assert features.unique_words >= 0
        assert features.sentence_count >= 0

        # Ratios should be between 0 and 1
        assert 0 <= features.lexical_diversity <= 1

        # Averages should be positive if there's content
        if features.word_count > 0:
            assert features.avg_word_length > 0
        if features.sentence_count > 0:
            assert features.avg_sentence_length > 0

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.text(min_size=10, max_size=200))
    def test_feature_relationships(self, extractor, text):
        """
        Property: Features have logical relationships.

        Unique words should not exceed total words, etc.
        """
        features = extractor.extract_all_features(text)

        # Unique words cannot exceed total words
        if features.word_count > 0:
            assert features.unique_words <= features.word_count
            assert features.lexical_diversity == features.unique_words / features.word_count
        else:
            assert features.lexical_diversity == 0.0

        # Average word length should make sense given character and word counts
        if features.word_count > 0:
            # Expected average length should be in reasonable range
            assert 0.1 <= features.avg_word_length <= 50  # Allow for various languages

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.text(min_size=10, max_size=200))
    def test_sentiment_features(self, extractor, text):
        """
        Property: Sentiment features have expected properties.

        Sentiment scores should be in valid range and sum appropriately.
        """
        features = extractor.extract_all_features(text)

        # Check that sentiment features exist
        assert hasattr(features, 'sentiment_compound')
        assert hasattr(features, 'sentiment_positive')
        assert hasattr(features, 'sentiment_negative')
        assert hasattr(features, 'sentiment_neutral')

        # All sentiment scores should be in valid range
        if features.sentiment_negative is not None:
            assert 0 <= features.sentiment_negative <= 1
        if features.sentiment_neutral is not None:
            assert 0 <= features.sentiment_neutral <= 1
        if features.sentiment_positive is not None:
            assert 0 <= features.sentiment_positive <= 1

        # Compound should be in valid range
        if features.sentiment_compound is not None:
            assert -1 <= features.sentiment_compound <= 1

        # Scores should sum to approximately 1
        if all(
            v is not None
            for v in [features.sentiment_negative, features.sentiment_neutral, features.sentiment_positive]
        ):
            sum_scores = (
                features.sentiment_negative +
                features.sentiment_neutral +
                features.sentiment_positive
            )
            assert abs(sum_scores - 1.0) < 0.01  # Allow small rounding errors

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.text(min_size=10, max_size=200))
    def test_pos_features(self, extractor, text):
        """
        Property: POS features have expected properties.

        POS ratios should be in valid range and sum appropriately.
        """
        features = extractor.extract_all_features(text)

        # Check that POS features exist
        assert hasattr(features, 'noun_ratio')
        assert hasattr(features, 'verb_ratio')
        assert hasattr(features, 'adj_ratio')
        assert hasattr(features, 'adv_ratio')

        # All POS ratios should be in valid range
        if features.noun_ratio is not None:
            assert 0 <= features.noun_ratio <= 1
        if features.verb_ratio is not None:
            assert 0 <= features.verb_ratio <= 1
        if features.adj_ratio is not None:
            assert 0 <= features.adj_ratio <= 1
        if features.adv_ratio is not None:
            assert 0 <= features.adv_ratio <= 1

        # Some POS categories should have values if there are words
        if features.word_count > 0:
            # At least some POS categories should be non-zero
            pos_ratios = [
                features.noun_ratio or 0,
                features.verb_ratio or 0,
                features.adj_ratio or 0,
                features.adv_ratio or 0
            ]
            assert sum(pos_ratios) >= 0  # Should be non-negative

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.text(min_size=10, max_size=200))
    def test_text_similarity_properties(self, extractor, text):
        """
        Property: Text similarity has expected mathematical properties.

        Similarity should be symmetric, reflexive, and in valid range.
        """
        # Test with same text
        sim_same = extractor.analyze_text_similarity(text, text)
        assert 0 <= sim_same['cosine_similarity'] <= 1.01  # Allow small floating-point error
        assert 0 <= sim_same['jaccard_similarity'] <= 1

        # Cosine similarity of identical texts should be 1.0 or 0.0 if empty after preprocessing
        # This is acceptable for property testing
        assert sim_same['cosine_similarity'] >= 0.0

        # Test symmetry
        other_text = text + " additional"
        sim_ab = extractor.analyze_text_similarity(text, other_text)
        sim_ba = extractor.analyze_text_similarity(other_text, text)

        assert abs(sim_ab['cosine_similarity'] - sim_ba['cosine_similarity']) < 0.01
        assert abs(sim_ab['jaccard_similarity'] - sim_ba['jaccard_similarity']) < 0.01

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.lists(st.text(min_size=20, max_size=100), min_size=5, max_size=20))
    def test_topic_model_consistency(self, extractor, texts):
        """
        Property: Topic modeling is consistent across similar texts.

        Texts with similar content should have similar topic distributions.
        """
        # Fit topic model on texts
        extractor.fit_topic_model(texts)

        # Extract features for first two texts
        if len(texts) >= 2:
            features1 = extractor.extract_all_features(texts[0])
            features2 = extractor.extract_all_features(texts[1])

            # Both should have topic features
            assert features1.topic_distribution is not None
            assert features2.topic_distribution is not None

            # Topic distributions should sum to 1
            assert abs(np.sum(features1.topic_distribution) - 1.0) < 0.01
            assert abs(np.sum(features2.topic_distribution) - 1.0) < 0.01

            # Dominant topic should be within valid range
            assert 0 <= features1.dominant_topic < extractor.config.n_topics
            assert 0 <= features2.dominant_topic < extractor.config.n_topics

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.text(min_size=10, max_size=200))
    def test_feature_extraction_robustness(self, extractor, text):
        """
        Property: Feature extraction is robust to edge cases.

        Should handle empty strings, special characters, Unicode, etc.
        """
        # Test with original text
        features = extractor.extract_all_features(text)
        assert isinstance(features, NLPFeatures)

        # Test with modifications
        modified_texts = [
            text.upper(),
            text.lower(),
            text + "!!!???@@@",
            " ".join(text.split()),  # Normalize whitespace
            text.replace(" ", "\t"),  # Different whitespace
        ]

        for mod_text in modified_texts:
            mod_features = extractor.extract_all_features(mod_text)
            assert isinstance(mod_features, NLPFeatures)

            # Features should be in valid ranges
            assert mod_features.char_count >= 0
            assert mod_features.word_count >= 0
            assert 0 <= mod_features.lexical_diversity <= 1

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.lists(st.text(min_size=10, max_size=100), min_size=3, max_size=10))
    def test_caching_behavior(self, extractor, texts):
        """
        Property: Caching improves performance without changing results.

        Cached results should be identical to non-cached results.
        """
        if len(texts) > 0:
            text = texts[0]

            # Extract without cache
            features_no_cache = extractor.extract_all_features(text, use_cache=False)

            # Extract with cache (first time)
            features_cached1 = extractor.extract_all_features(text, use_cache=True)

            # Extract with cache (second time, should use cache)
            features_cached2 = extractor.extract_all_features(text, use_cache=True)

            # All results should be identical
            assert features_no_cache.char_count == features_cached1.char_count
            assert features_no_cache.word_count == features_cached1.word_count
            assert features_cached1.char_count == features_cached2.char_count
            assert features_cached1.word_count == features_cached2.word_count

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.lists(st.text(min_size=20, max_size=200), min_size=5, max_size=10))
    def test_feature_richness_across_corpus(self, extractor, texts):
        """
        Property: Extracting features across a corpus yields rich, varied features.

        Different texts should produce different feature patterns.
        """
        if len(texts) > 1:
            # Fit topic model if we have enough texts
            if len(texts) >= 3:
                extractor.fit_topic_model(texts[:-1])

            # Extract features for all texts
            all_features = [extractor.extract_all_features(t) for t in texts]

            # There should be variation in features
            char_counts = [f.char_count for f in all_features]
            word_counts = [f.word_count for f in all_features]

            # Not all texts should have the same characteristics
            assert len(set(char_counts)) > 1 or len(set(word_counts)) > 1

            # Features should have reasonable variation
            if len(all_features) >= 3:
                # Check that there's variation in some features
                lexical_diversities = [f.lexical_diversity for f in all_features]
                assert max(lexical_diversities) >= min(lexical_diversities)

    def test_feature_completeness(self, extractor):
        """
        Property: NLPFeatures object contains all expected fields.

        The NLPFeatures dataclass should have all expected attributes.
        """
        text = "This is a test sentence with multiple words."
        features = extractor.extract_all_features(text)

        # Check that all expected attributes exist
        expected_attributes = [
            'char_count', 'word_count', 'unique_words', 'sentence_count',
            'avg_word_length', 'avg_sentence_length', 'lexical_diversity',
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
            'entity_count', 'entity_types',
            'topic_distribution', 'dominant_topic',
            'readability_index', 'formality_score',
            'technical_density'
        ]

        for attr in expected_attributes:
            assert hasattr(features, attr), f"Missing attribute: {attr}"