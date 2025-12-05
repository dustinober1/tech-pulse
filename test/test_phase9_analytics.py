"""
Test suite for Phase 9 Advanced Analytics Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import Phase 9 modules with graceful fallback
PHASE9_AVAILABLE = False
try:
    from src.phase9.market_analyzer import MarketIntelligenceAnalyzer
    from src.phase9.competitor_tracker import CompetitorTracker
    from src.phase9.trend_predictor import AdvancedTrendPredictor
    from src.phase9.intelligence_reporter import IntelligenceReporter
    PHASE9_AVAILABLE = True
except ImportError as e:
    print(f"Phase 9 modules not available: {e}")
    PHASE9_AVAILABLE = False

class TestMarketIntelligence(unittest.TestCase):
    """Test suite for Market Intelligence functionality"""

    def setUp(self):
        """Set up test fixtures"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        self.market_analyzer = MarketIntelligenceAnalyzer()
        self.sample_data = self._create_sample_market_data()

    def _create_sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        data = {
            'date': dates,
            'stock_symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] * 25,
            'price': np.random.uniform(100, 500, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'sentiment': np.random.uniform(-0.5, 0.5, 100),
            'topic_keywords': ['ai_ml', 'cloud', 'security', 'funding'] * 25,
            'source': ['techcrunch', 'venturebeat', 'arxiv', 'github'] * 25
        }
        return pd.DataFrame(data)

    def test_market_volatility_analysis(self):
        """Test market volatility analysis"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            results = self.market_analyzer.analyze_volatility(self.sample_data)
            self.assertIn('volatility_index', results)
            self.assertIn('trend_strength', results)
        except Exception as e:
            self.fail(f"Volatility analysis failed: {e}")

    def test_competitor_tracking(self):
        """Test competitor tracking functionality"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            tracker = CompetitorTracker()
            results = tracker.track_competitors(self.sample_data)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Competitor tracking failed: {e}")

    def test_trend_prediction(self):
        """Test advanced trend prediction"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            predictor = AdvancedTrendPredictor()
            predictions = predictor.predict_trends(self.sample_data, periods=30)
            self.assertGreater(len(predictions), 0)
        except Exception as e:
            self.fail(f"Trend prediction failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)