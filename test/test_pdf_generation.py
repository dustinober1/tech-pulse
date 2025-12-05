"""
Test infrastructure for PDF Generation functionality
Tests basic PDF builder initialization and style creation
"""

import unittest
import os
import tempfile
import io
from datetime import datetime

# Import PDF modules with graceful fallback
PDF_AVAILABLE = False
try:
    # First try to import the dependencies
    import reportlab
    from reportlab.lib.pagesizes import A4
    import fpdf2
    import openai

    # Then try to import our modules
    from src.pdf_generator.pdf_builder import ExecutiveBriefingPDF, create_sample_pdf
    PDF_AVAILABLE = True
except ImportError as e:
    print(f"PDF modules not available: {e}")
    PDF_AVAILABLE = False


class TestPDFGenerationInfrastructure(unittest.TestCase):
    """
    Test suite for PDF generation infrastructure
    """

    def setUp(self):
        """Set up test fixtures"""
        if not PDF_AVAILABLE:
            self.skipTest("PDF generation modules not available")

        self.pdf_builder = ExecutiveBriefingPDF()
        self.test_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up any test files
        import shutil
        if hasattr(self, 'test_output_dir') and os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir, ignore_errors=True)

    def test_pdf_builder_initialization(self):
        """Test PDF builder initializes correctly"""
        self.assertIsNotNone(self.pdf_builder)
        self.assertEqual(self.pdf_builder.page_size, A4)
        self.assertEqual(self.pdf_builder.margin, 0.75 * 72)  # 0.75 inches in points

    def test_custom_styles_creation(self):
        """Test custom styles are created properly"""
        # Check that all required styles exist
        self.assertTrue(hasattr(self.pdf_builder, 'title_style'))
        self.assertTrue(hasattr(self.pdf_builder, 'heading_style'))
        self.assertTrue(hasattr(self.pdf_builder, 'subheading_style'))
        self.assertTrue(hasattr(self.pdf_builder, 'body_style'))
        self.assertTrue(hasattr(self.pdf_builder, 'metric_style'))
        self.assertTrue(hasattr(self.pdf_builder, 'footer_style'))

        # Check style properties
        self.assertEqual(self.pdf_builder.title_style.fontSize, 24)
        self.assertEqual(self.pdf_builder.heading_style.fontSize, 16)
        self.assertEqual(self.pdf_builder.body_style.fontSize, 11)

    def test_basic_page_layout(self):
        """Test basic page layout functionality"""
        # Create minimal sample data
        sample_data = {
            "title": "Test Report",
            "date": datetime.now(),
            "summary": "This is a test summary for the PDF generation test.",
            "metrics": {
                "total_stories": 10,
                "avg_sentiment": 0.5,
                "sentiment_distribution": {"Positive": 5, "Negative": 3, "Neutral": 2},
                "avg_score": 100.0,
                "total_comments": 500,
                "most_active_hour": 12,
                "top_source": "example.com"
            },
            "top_stories": [
                {
                    "rank": 1,
                    "title": "Test Story",
                    "score": 100,
                    "sentiment": "Positive",
                    "url": "https://example.com/story"
                }
            ],
            "topics": {
                "test_topic": 50.0
            }
        }

        try:
            # Generate PDF
            pdf_bytes = self.pdf_builder.build_report(**sample_data)

            # Verify PDF was generated
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 1000)  # Should be substantial

            # Check PDF header
            self.assertTrue(pdf_bytes.startswith(b'%PDF'))

        except Exception as e:
            self.fail(f"PDF generation failed: {e}")

    def test_font_rendering(self):
        """Test font rendering capabilities"""
        # Test various text samples
        test_texts = [
            "Simple text",
            "Text with numbers: 123",
            "Special chars: @#$%^&*()",
            "Unicode: café résumé naïve",
            "Long text that should wrap properly across multiple lines in the PDF document"
        ]

        sample_data = {
            "title": "Font Test Report",
            "date": datetime.now(),
            "summary": " ".join(test_texts),
            "metrics": {
                "total_stories": 1,
                "avg_sentiment": 0.0,
                "sentiment_distribution": {"Neutral": 1},
                "avg_score": 50.0,
                "total_comments": 10,
                "most_active_hour": 9,
                "top_source": "test.com"
            },
            "top_stories": [
                {
                    "rank": 1,
                    "title": test_texts[0],
                    "score": 50,
                    "sentiment": "Neutral",
                    "url": "https://test.com"
                }
            ],
            "topics": {}
        }

        try:
            pdf_bytes = self.pdf_builder.build_report(**sample_data)
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 1000)
        except Exception as e:
            self.fail(f"Font rendering test failed: {e}")

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        # Test with minimal data
        minimal_data = {
            "title": "Minimal Test",
            "date": datetime.now(),
            "summary": "Minimal summary",
            "metrics": {},
            "top_stories": [],
            "topics": {}
        }

        try:
            pdf_bytes = self.pdf_builder.build_report(**minimal_data)
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 500)
        except Exception as e:
            self.fail(f"Empty data handling failed: {e}")

    def test_large_content_handling(self):
        """Test handling of larger content"""
        # Generate large summary
        large_summary = "This is a test. " * 100  # 100 repetitions

        # Generate many top stories
        many_stories = [
            {
                "rank": i + 1,
                "title": f"Test Story {i + 1}",
                "score": 100 - i,
                "sentiment": "Positive" if i % 2 == 0 else "Negative",
                "url": f"https://example.com/story{i + 1}"
            }
            for i in range(20)
        ]

        sample_data = {
            "title": "Large Content Test",
            "date": datetime.now(),
            "summary": large_summary,
            "metrics": {
                "total_stories": 20,
                "avg_sentiment": 0.1,
                "sentiment_distribution": {"Positive": 10, "Negative": 10},
                "avg_score": 55.0,
                "total_comments": 1000,
                "most_active_hour": 15,
                "top_source": "large-content-test.com"
            },
            "top_stories": many_stories,
            "topics": {f"topic_{i}": 5.0 for i in range(10)}
        }

        try:
            pdf_bytes = self.pdf_builder.build_report(**sample_data)
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 5000)  # Should be larger
        except Exception as e:
            self.fail(f"Large content handling failed: {e}")

    def test_sample_pdf_creation(self):
        """Test the sample PDF creation function"""
        try:
            output_path = os.path.join(self.test_output_dir, "test_sample.pdf")
            created_path = create_sample_pdf(output_path)

            # Verify file was created
            self.assertEqual(created_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify file content
            with open(output_path, 'rb') as f:
                content = f.read()
                self.assertTrue(content.startswith(b'%PDF'))
                self.assertGreater(len(content), 1000)

        except Exception as e:
            self.fail(f"Sample PDF creation failed: {e}")

    def test_different_page_sizes(self):
        """Test PDF generation with different page sizes"""
        from reportlab.lib.pagesizes import letter, A4, legal

        for page_size in [letter, A4, legal]:
            with self.subTest(page_size=page_size):
                builder = ExecutiveBriefingPDF(page_size=page_size)
                self.assertEqual(builder.page_size, page_size)

                sample_data = {
                    "title": f"Page Size Test - {page_size[0]}x{page_size[1]}",
                    "date": datetime.now(),
                    "summary": f"Testing page size {page_size}",
                    "metrics": {"total_stories": 1},
                    "top_stories": [],
                    "topics": {}
                }

                try:
                    pdf_bytes = builder.build_report(**sample_data)
                    self.assertIsInstance(pdf_bytes, bytes)
                    self.assertGreater(len(pdf_bytes), 500)
                except Exception as e:
                    self.fail(f"Page size {page_size} test failed: {e}")

    def test_margin_settings(self):
        """Test different margin settings"""
        for margin in [0.5, 0.75, 1.0, 1.5]:
            with self.subTest(margin=margin):
                builder = ExecutiveBriefingPDF(margin=margin)
                expected_margin = margin * 72  # Convert inches to points
                self.assertEqual(builder.margin, expected_margin)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid date
        with self.assertRaises(Exception):
            self.pdf_builder.build_report(
                title="Test",
                date="invalid_date",  # Should be datetime object
                summary="Test",
                metrics={},
                top_stories=[],
                topics={}
            )

        # Test with None values
        with self.assertRaises(Exception):
            self.pdf_builder.build_report(
                title=None,
                date=datetime.now(),
                summary="Test",
                metrics={},
                top_stories=[],
                topics={}
            )


if __name__ == '__main__':
    # Configure test output
    import sys
    import logging

    # Set logging level
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2, exit=False)