# Phase 7: The Intelligence Matrix - Implementation Guide

**Created:** 2025-12-05
**Status:** Ready for Implementation
**Estimated Duration:** 9 weeks
**Target Audience:** Junior Engineers

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Implementation Phases](#implementation-phases)
5. [Detailed Tasks](#detailed-tasks)
6. [Code Templates](#code-templates)
7. [Testing Instructions](#testing-instructions)
8. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

Phase 7 transforms Tech-Pulse from a reactive analytics dashboard into a proactive intelligence platform. We'll add three major features:
1. **Predictive Analytics** - Predict which stories will go viral
2. **User Personalization** - Remember user preferences and customize content
3. **Multi-Source Data** - Pull from Reddit and RSS feeds, not just Hacker News

## Prerequisites

Before starting Phase 7, ensure you have:
- Python 3.13+ installed
- Git access to the repository
- Basic understanding of:
  - Python classes and functions
  - SQL databases
  - REST APIs
  - Machine learning concepts (don't worry, we'll guide you!)
- Access to the Tech-Pulse codebase

## Environment Setup

### 1. Create Phase 7 Branch

```bash
git checkout main
git pull origin main
git checkout -b phase-7-intelligence-matrix
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv_phase7
source venv_phase7/bin/activate  # On Windows: venv_phase7\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Additional Dependencies

```bash
pip install praw feedparser schedule joblib scipy aiohttp
```

### 4. Create New Directory Structure

```bash
mkdir -p src/phase7/predictive_analytics
mkdir -p src/phase7/user_management
mkdir -p src/phase7/source_connectors
mkdir -p test/phase7
```

## Implementation Phases

### Phase 7.1: Predictive Analytics (Weeks 1-3)
**Goal:** Add a "Crystal Ball" that predicts if a story will go viral

### Phase 7.2: User Personalization (Weeks 4-5)
**Goal:** Let users save preferences and get personalized recommendations

### Phase 7.3: Multi-Source Integration (Weeks 6-9)
**Goal:** Add Reddit and RSS feeds as data sources

## Detailed Tasks

### Phase 7.1: Predictive Analytics - Week 1

#### Task 1.1: Create the Predictive Analytics Module
**File:** `src/phase7/predictive_analytics/predictor.py`

```python
"""
This module will predict which Hacker News stories will go viral.
We'll use machine learning to learn from historical data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class ViralityPredictor:
    """Predicts if a story will go viral based on features"""

    def __init__(self):
        self.model = None
        self.feature_columns = [
            'title_length',
            'sentiment_score',
            'hour_of_day',
            'day_of_week',
            'has_question_mark',
            'word_count'
        ]

    def train_model(self, training_data):
        """
        Train the model with historical data
        Args:
            training_data: DataFrame with features and 'went_viral' column
        """
        # TODO: Implement model training
        pass

    def predict_virality(self, story_data):
        """
        Predict if a story will go viral
        Args:
            story_data: Dict with story features
        Returns:
            float: Probability of going viral (0-1)
        """
        # TODO: Implement prediction
        pass

    def extract_features(self, story):
        """
        Extract features from a story
        Args:
            story: Dict with story data
        Returns:
            Dict: Extracted features
        """
        features = {}

        # Title length
        title = story.get('title', '')
        features['title_length'] = len(title)
        features['word_count'] = len(title.split())

        # Has question mark
        features['has_question_mark'] = 1 if '?' in title else 0

        # Time features
        # TODO: Add time-based features

        # Sentiment score
        # TODO: Get sentiment from existing analysis

        return features
```

**Step-by-step instructions:**
1. Create the directory: `mkdir -p src/phase7/predictive_analytics`
2. Create the file: `touch src/phase7/predictive_analytics/predictor.py`
3. Copy the code above into the file
4. Read through the comments to understand the structure

#### Task 1.2: Create Feature Extractor
**File:** `src/phase7/predictive_analytics/features.py`

```python
"""
Extract features from stories for prediction
"""

import re
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class FeatureExtractor:
    """Extracts features from stories"""

    def __init__(self):
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')

        self.sia = SentimentIntensityAnalyzer()

    def extract_text_features(self, title):
        """Extract features from story title"""
        features = {
            'title_length': len(title),
            'word_count': len(title.split()),
            'has_question_mark': 1 if '?' in title else 0,
            'has_exclamation': 1 if '!' in title else 0,
            'has_numbers': 1 if re.search(r'\d', title) else 0,
            'capital_ratio': sum(1 for c in title if c.isupper()) / len(title) if title else 0,
            'sentiment_score': self.get_sentiment_score(title)
        }

        return features

    def extract_time_features(self, timestamp):
        """Extract time-based features"""
        dt = datetime.fromtimestamp(timestamp)
        features = {
            'hour_of_day': dt.hour,
            'day_of_week': dt.weekday(),  # 0 = Monday
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_business_hours': 1 if 9 <= dt.hour <= 17 else 0
        }

        return features

    def get_sentiment_score(self, text):
        """Get sentiment score from text"""
        scores = self.sia.polarity_scores(text)
        return scores['compound']  # Compound score between -1 and 1
```

**Step-by-step instructions:**
1. Create the file: `touch src/phase7/predictive_analytics/features.py`
2. Copy the code above
3. This file handles extracting numerical features from text
4. The NLTK library helps analyze sentiment

#### Task 1.3: Create Training Data Generator
**File:** `src/phase7/predictive_analytics/training_data.py`

```python
"""
Generate training data from historical stories
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TrainingDataGenerator:
    """Generate training data for the virality predictor"""

    def __init__(self):
        self.feature_extractor = None  # Will be imported when needed

    def load_historical_data(self):
        """
        Load historical story data
        Returns:
            DataFrame: Historical stories with features
        """
        # For now, generate synthetic data
        # In production, this would load from your database

        data = []
        for i in range(1000):  # Generate 1000 sample stories
            story = {
                'id': i,
                'title': self.generate_title(),
                'score': random.randint(0, 5000),
                'timestamp': self.generate_timestamp()
            }
            data.append(story)

        df = pd.DataFrame(data)
        return df

    def generate_title(self):
        """Generate a realistic story title"""
        templates = [
            "New {tech} release announced by {company}",
            "Why {concept} is changing {industry}",
            "{company} raises ${amount} for {product}",
            "Breaking: {event} affects {community}",
            "How to {action} with {tool}"
        ]

        tech_words = ["AI", "blockchain", "cloud", "mobile", "web3", "ML", "API"]
        companies = ["Google", "Microsoft", "Apple", "Amazon", "Meta", "StartupXYZ"]
        concepts = ["remote work", "open source", "agile", "DevOps", "microservices"]

        template = random.choice(templates)
        title = template.format(
            tech=random.choice(tech_words),
            company=random.choice(companies),
            concept=random.choice(concepts),
            industry="tech",
            amount=f"{random.randint(1, 100)}M",
            product="platform",
            event="policy change",
            community="developers",
            action="optimize",
            tool="Python"
        )

        return title

    def generate_timestamp(self):
        """Generate a timestamp from the last 6 months"""
        days_ago = random.randint(0, 180)
        hours_ago = random.randint(0, 23)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
        return timestamp.timestamp()

    def label_viral_stories(self, df, viral_threshold=1000):
        """
        Label stories as viral or not
        Args:
            df: DataFrame with stories
            viral_threshold: Score above which story is considered viral
        """
        df['went_viral'] = (df['score'] > viral_threshold).astype(int)
        return df

    def prepare_training_data(self):
        """
        Prepare complete training dataset
        Returns:
            DataFrame: Training data with features and labels
        """
        # Load historical data
        df = self.load_historical_data()

        # Label viral stories
        df = self.label_viral_stories(df)

        # TODO: Extract features using FeatureExtractor
        # For now, add simple features
        df['title_length'] = df['title'].str.len()
        df['word_count'] = df['title'].str.split().str.len()
        df['has_question_mark'] = df['title'].str.contains('\?').astype(int)

        # Add time features
        df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek

        return df
```

**Step-by-step instructions:**
1. Create the file: `touch src/phase7/predictive_analytics/training_data.py`
2. This generates fake training data to test our predictor
3. Later, we'll replace this with real historical data

#### Task 1.4: Create Test for Predictor
**File:** `test/phase7/test_predictor.py`

```python
"""
Tests for the virality predictor
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from phase7.predictive_analytics.predictor import ViralityPredictor
from phase7.predictive_analytics.features import FeatureExtractor

class TestViralityPredictor(unittest.TestCase):
    """Test the virality predictor"""

    def setUp(self):
        self.predictor = ViralityPredictor()
        self.feature_extractor = FeatureExtractor()

    def test_extract_features(self):
        """Test feature extraction"""
        story = {
            'title': 'Python 3.13 Released with Amazing New Features!',
            'timestamp': 1701388800  # Dec 1, 2023
        }

        features = self.predictor.extract_features(story)

        # Check that all expected features are present
        self.assertIn('title_length', features)
        self.assertIn('word_count', features)
        self.assertIn('has_question_mark', features)

        # Check specific values
        self.assertEqual(features['title_length'], len(story['title']))
        self.assertEqual(features['word_count'], 6)
        self.assertEqual(features['has_question_mark'], 0)

    def test_feature_extractor_text_features(self):
        """Test text feature extraction"""
        title = "Amazing AI Breakthrough! What do you think?"
        features = self.feature_extractor.extract_text_features(title)

        self.assertEqual(features['has_question_mark'], 1)
        self.assertEqual(features['has_exclamation'], 1)
        self.assertEqual(features['title_length'], len(title))
        self.assertGreater(features['sentiment_score'], 0)  # Should be positive

    def test_feature_extractor_time_features(self):
        """Test time feature extraction"""
        # Test a known timestamp (Monday at 2 PM)
        timestamp = datetime(2023, 12, 4, 14, 0, 0).timestamp()
        features = self.feature_extractor.extract_time_features(timestamp)

        self.assertEqual(features['hour_of_day'], 14)
        self.assertEqual(features['day_of_week'], 0)  # Monday
        self.assertEqual(features['is_weekend'], 0)
        self.assertEqual(features['is_business_hours'], 1)

if __name__ == '__main__':
    unittest.main()
```

**Step-by-step instructions:**
1. Create the test file
2. Run the test: `python -m pytest test/phase7/test_predictor.py -v`
3. Fix any import errors until tests pass

#### Task 1.5: Integrate Predictor with Main App
**File:** `app.py` (add to existing file)

Find the section in app.py where it displays story data and add:

```python
# Near line 200, after displaying the metrics
try:
    from src.phase7.predictive_analytics.predictor import ViralityPredictor
    from src.phase7.predictive_analytics.features import FeatureExtractor

    # Initialize predictor if not in session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ViralityPredictor()
        st.session_state.feature_extractor = FeatureExtractor()

    # Add prediction section to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Crystal Ball")

    # Train model button (for demo)
    if st.sidebar.button("Train Virality Predictor", help="Train with historical data"):
        with st.spinner("Training model..."):
            # TODO: Implement actual training
            st.sidebar.success("Model trained!")

    # Show predictions for top stories
    if not df.empty and st.sidebar.checkbox("Show Virality Predictions"):
        st.sidebar.markdown("#### Predictions:")

        predictor = st.session_state.predictor
        feature_extractor = st.session_state.feature_extractor

        for _, row in df.head(5).iterrows():
            # Extract features
            text_features = feature_extractor.extract_text_features(row['title'])
            time_features = feature_extractor.extract_time_features(row['timestamp'])

            # Combine features (simplified for demo)
            all_features = {**text_features, **time_features}

            # Make prediction (random for demo)
            prediction = np.random.random()  # TODO: Use actual predictor

            # Display
            color = "green" if prediction > 0.7 else "orange" if prediction > 0.4 else "red"
            st.sidebar.markdown(f"- <span style='color:{color}'>●</span> {prediction:.1%}", unsafe_allow_html=True)
            st.sidebar.caption(row['title'][:50] + "...")

except ImportError as e:
    st.sidebar.warning("Predictive analytics not available")
    st.sidebar.caption(f"Error: {str(e)}")
```

**Step-by-step instructions:**
1. Open app.py
2. Find where the main content is displayed (after the metrics row)
3. Add the code above
4. Run the app to test: `streamlit run app.py`
5. You should see a "Crystal Ball" section in the sidebar

### Phase 7.1: Predictive Analytics - Week 2

#### Task 2.1: Complete Model Training Implementation

Update `src/phase7/predictive_analytics/predictor.py`:

```python
    def train_model(self, training_data):
        """
        Train the model with historical data
        """
        # Prepare features
        X = training_data[self.feature_columns]
        y = training_data['went_viral']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"Training accuracy: {train_score:.2%}")
        print(f"Test accuracy: {test_score:.2%}")

        # Save model
        self.save_model()

        return test_score

    def predict_virality(self, story_data):
        """
        Predict if a story will go viral
        """
        if self.model is None:
            return 0.5  # Default prediction

        # Convert to DataFrame
        df = pd.DataFrame([story_data])

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Predict probability
        prediction = self.model.predict_proba(df[self.feature_columns])[0]

        # Return probability of going viral (class 1)
        return prediction[1]

    def save_model(self):
        """Save the trained model"""
        model_path = 'models/virality_predictor.joblib'
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)

    def load_model(self):
        """Load a trained model"""
        model_path = 'models/virality_predictor.joblib'
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return True
        return False
```

#### Task 2.2: Create Model Training Script
**File:** `src/phase7/predictive_analytics/train_model.py`

```python
"""
Script to train the virality prediction model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phase7.predictive_analytics.predictor import ViralityPredictor
from phase7.predictive_analytics.training_data import TrainingDataGenerator

def main():
    """Train the virality predictor"""
    print("Starting model training...")

    # Initialize components
    predictor = ViralityPredictor()
    data_generator = TrainingDataGenerator()

    # Generate training data
    print("Generating training data...")
    training_data = data_generator.prepare_training_data()

    # Train model
    print("Training model...")
    accuracy = predictor.train_model(training_data)

    print(f"\nTraining complete! Accuracy: {accuracy:.2%}")

    # Test with a sample
    test_story = {
        'title': ' Revolutionary AI Technology Transforms Industry',
        'timestamp': 1701388800
    }

    prediction = predictor.predict_virality(test_story)
    print(f"Sample prediction: {prediction:.1%}")

if __name__ == "__main__":
    main()
```

**Step-by-step instructions:**
1. Run the training script: `python src/phase7/predictive_analytics/train_model.py`
2. You should see the model training and accuracy metrics
3. Check that a model file was created in `models/virality_predictor.joblib`

### Phase 7.1: Predictive Analytics - Week 3

#### Task 3.1: Create Prediction Dashboard Widget
**File:** `src/phase7/predictive_analytics/dashboard.py`

```python
"""
Dashboard components for predictive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def display_crystal_ball(predictions_df):
    """
    Display the crystal ball widget in the sidebar
    """
    st.sidebar.markdown("### 🔮 Crystal Ball")

    if predictions_df.empty:
        st.sidebar.info("No predictions available")
        return

    # Overall virality indicator
    avg_prediction = predictions_df['virality_score'].mean()

    # Color based on average
    if avg_prediction > 0.7:
        color = "🟢"
        status = "High Virality Potential"
    elif avg_prediction > 0.4:
        color = "🟡"
        status = "Moderate Virality Potential"
    else:
        color = "🔴"
        status = "Low Virality Potential"

    st.sidebar.markdown(f"### {color} {status}")
    st.sidebar.markdown(f"**Average Score:** {avg_prediction:.1%}")

    # Top predictions
    st.sidebar.markdown("#### Top Predictions:")

    top_stories = predictions_df.nlargest(3, 'virality_score')
    for _, row in top_stories.iterrows():
        score = row['virality_score']
        title = row['title'][:40] + "..." if len(row['title']) > 40 else row['title']

        # Progress bar
        st.sidebar.markdown(f"**{title}**")
        st.sidebar.progress(score)
        st.sidebar.caption(f"Virality: {score:.1%}")
        st.sidebar.markdown("---")

    # Confidence meter
    st.sidebar.markdown("#### Model Confidence")
    confidence = 0.85  # TODO: Calculate actual confidence
    st.sidebar.progress(confidence)
    st.sidebar.caption(f"{confidence:.1%} confidence in predictions")

def display_prediction_chart(df, predictions_df):
    """
    Display predictions in the main dashboard
    """
    if predictions_df.empty:
        return

    st.markdown("### 📊 Virality Predictions")

    # Create a copy with predictions
    display_df = df.copy()
    display_df['virality_score'] = predictions_df['virality_score']
    display_df['virality_category'] = pd.cut(
        display_df['virality_score'],
        bins=[0, 0.3, 0.7, 1],
        labels=['Low', 'Medium', 'High']
    )

    # Color mapping
    color_map = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}

    # Create bar chart
    fig = px.bar(
        display_df.head(10),
        x='virality_score',
        y='title',
        orientation='h',
        color='virality_category',
        color_discrete_map=color_map,
        title="Top Stories - Virality Prediction",
        labels={'virality_score': 'Virality Probability', 'title': ''},
        height=400
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Virality Probability",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("#### What Drives Virality?")

    # Mock feature importance data
    feature_importance = pd.DataFrame({
        'feature': ['Sentiment Score', 'Time Posted', 'Title Length', 'Word Count', 'Has Question'],
        'importance': [0.35, 0.25, 0.15, 0.15, 0.10]
    })

    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Predictions",
        labels={'importance': 'Importance', 'feature': ''}
    )

    st.plotly_chart(fig, use_container_width=True)
```

#### Task 3.2: Update Main App with Prediction Dashboard

In `app.py`, find where stories are displayed and add:

```python
# After displaying the main data table
if 'df' in locals() and not df.empty:
    # Get predictions
    predictions = []
    for _, row in df.iterrows():
        story_data = {
            'title': row['title'],
            'timestamp': row.get('timestamp', time.time())
        }

        # Extract features
        features = st.session_state.feature_extractor.extract_text_features(story_data['title'])
        time_features = st.session_state.feature_extractor.extract_time_features(story_data['timestamp'])
        all_features = {**features, **time_features}

        # Get prediction
        prediction = st.session_state.predictor.predict_virality(all_features)
        predictions.append({
            'title': row['title'],
            'virality_score': prediction
        })

    predictions_df = pd.DataFrame(predictions)

    # Display prediction dashboard
    from src.phase7.predictive_analytics.dashboard import display_crystal_ball, display_prediction_chart

    # Add crystal ball to sidebar
    display_crystal_ball(predictions_df)

    # Add prediction chart to main content
    display_prediction_chart(df, predictions_df)
```

### Phase 7.2: User Personalization - Week 4

#### Task 4.1: Create User Database Schema
**File:** `src/phase7/user_management/database.py`

```python
"""
SQLite database for user management and preferences
"""

import sqlite3
import os
from datetime import datetime
import json

class UserDatabase:
    """Manages SQLite database for user data"""

    def __init__(self, db_path='data/users.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT PRIMARY KEY,
                topics TEXT,  -- JSON array of preferred topics
                min_score INTEGER DEFAULT 0,
                sentiment_filter TEXT DEFAULT 'all',  -- positive, negative, all
                notification_freq TEXT DEFAULT 'daily',  -- real-time, hourly, daily
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Watchlist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                keyword TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Reading history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reading_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                story_id TEXT,
                story_title TEXT,
                viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_time INTEGER DEFAULT 0,  -- seconds spent on story
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()
        conn.close()

    def get_or_create_user(self, user_id=None):
        """Get existing user or create new one"""
        if user_id is None:
            user_id = self.generate_user_id()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if cursor.fetchone() is None:
            # Create new user
            cursor.execute(
                "INSERT INTO users (id) VALUES (?)",
                (user_id,)
            )
            # Create default preferences
            cursor.execute(
                """INSERT INTO preferences (user_id, topics, min_score, sentiment_filter, notification_freq)
                   VALUES (?, '[]', 0, 'all', 'daily')""",
                (user_id,)
            )

        conn.commit()
        conn.close()

        return user_id

    def generate_user_id(self):
        """Generate a unique user ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def save_preferences(self, user_id, preferences):
        """Save user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE preferences
            SET topics = ?, min_score = ?, sentiment_filter = ?, notification_freq = ?
            WHERE user_id = ?
        ''', (
            json.dumps(preferences.get('topics', [])),
            preferences.get('min_score', 0),
            preferences.get('sentiment_filter', 'all'),
            preferences.get('notification_freq', 'daily'),
            user_id
        ))

        conn.commit()
        conn.close()

    def get_preferences(self, user_id):
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT topics, min_score, sentiment_filter, notification_freq
            FROM preferences WHERE user_id = ?
        ''', (user_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'topics': json.loads(result[0]) if result[0] else [],
                'min_score': result[1],
                'sentiment_filter': result[2],
                'notification_freq': result[3]
            }

        return None

    def add_watchlist_item(self, user_id, keyword):
        """Add keyword to watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO watchlist (user_id, keyword) VALUES (?, ?)",
            (user_id, keyword)
        )

        conn.commit()
        conn.close()

    def get_watchlist(self, user_id):
        """Get user's watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT keyword FROM watchlist WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )

        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def track_reading(self, user_id, story_id, story_title, interaction_time=0):
        """Track story reading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO reading_history (user_id, story_id, story_title, interaction_time)
            VALUES (?, ?, ?, ?)
        ''', (user_id, story_id, story_title, interaction_time))

        # Update last active
        cursor.execute(
            "UPDATE users SET last_active = ? WHERE id = ?",
            (datetime.now(), user_id)
        )

        conn.commit()
        conn.close()
```

**Step-by-step instructions:**
1. Create the user management directory: `mkdir -p src/phase7/user_management`
2. Create the database file with the code above
3. This sets up the database structure for storing user data

#### Task 4.2: Create User Profile Manager
**File:** `src/phase7/user_manager/user_profile.py`

```python
"""
Manage user profiles and preferences
"""

import streamlit as st
from .database import UserDatabase
import hashlib

class UserProfileManager:
    """Manages user profiles in the dashboard"""

    def __init__(self):
        self.db = UserDatabase()

    def get_current_user(self):
        """Get or create current user"""
        # Try to get user from session state
        if 'user_id' not in st.session_state:
            # Generate simple user ID from browser fingerprint
            user_agent = st.get_option('browser.userAgent', '')
            ip = st.get_option('browser.gatheredStats', {}).get('ip', '')

            # Create hash
            user_id = hashlib.md5(f"{user_agent}_{ip}".encode()).hexdigest()[:16]
            st.session_state.user_id = self.db.get_or_create_user(user_id)

        return st.session_state.user_id

    def display_profile_settings(self):
        """Display user profile settings in sidebar"""
        user_id = self.get_current_user()
        preferences = self.db.get_preferences(user_id)

        if preferences is None:
            preferences = {
                'topics': [],
                'min_score': 0,
                'sentiment_filter': 'all',
                'notification_freq': 'daily'
            }

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Profile Settings")

        # Topics of interest
        st.sidebar.markdown("**Topics of Interest:**")

        # Common tech topics
        all_topics = [
            'Artificial Intelligence', 'Machine Learning', 'Blockchain',
            'Cloud Computing', 'DevOps', 'Security', 'Mobile',
            'Web Development', 'Data Science', 'Startups'
        ]

        selected_topics = st.sidebar.multiselect(
            "Select topics you're interested in:",
            all_topics,
            default=preferences.get('topics', []),
            help="Stories about these topics will be highlighted"
        )

        # Minimum score filter
        min_score = st.sidebar.slider(
            "Minimum story score:",
            min_value=0,
            max_value=500,
            value=preferences.get('min_score', 0),
            help="Hide stories below this score"
        )

        # Sentiment filter
        sentiment_filter = st.sidebar.selectbox(
            "Sentiment filter:",
            ['all', 'positive', 'negative'],
            index=0 if preferences.get('sentiment_filter') == 'all' else
                  1 if preferences.get('sentiment_filter') == 'positive' else 2,
            help="Filter stories by sentiment"
        )

        # Notification frequency
        notification_freq = st.sidebar.selectbox(
            "Notification frequency:",
            ['real-time', 'hourly', 'daily', 'weekly'],
            index=0 if preferences.get('notification_freq') == 'real-time' else
                  1 if preferences.get('notification_freq') == 'hourly' else
                  2 if preferences.get('notification_freq') == 'daily' else 3
        )

        # Save button
        if st.sidebar.button("Save Preferences"):
            new_preferences = {
                'topics': selected_topics,
                'min_score': min_score,
                'sentiment_filter': sentiment_filter,
                'notification_freq': notification_freq
            }

            self.db.save_preferences(user_id, new_preferences)
            st.sidebar.success("Preferences saved!")

    def display_watchlist(self):
        """Display watchlist management"""
        user_id = self.get_current_user()
        watchlist = self.db.get_watchlist(user_id)

        st.sidebar.markdown("**Watchlist:**")

        # Add new keyword
        new_keyword = st.sidebar.text_input(
            "Add keyword to watch:",
            help="Get notified when stories contain this keyword"
        )

        if st.sidebar.button("Add to Watchlist") and new_keyword:
            self.db.add_watchlist_item(user_id, new_keyword)
            st.sidebar.success(f"'{new_keyword}' added to watchlist!")
            st.rerun()

        # Display current watchlist
        if watchlist:
            st.sidebar.markdown("**Current watchlist:**")
            for keyword in watchlist:
                st.sidebar.write(f"• {keyword}")
        else:
            st.sidebar.caption("No keywords in watchlist")

    def get_user_filters(self):
        """Get user's current filters for stories"""
        user_id = self.get_current_user()
        preferences = self.db.get_preferences(user_id)

        if not preferences:
            return {}

        return {
            'topics': preferences.get('topics', []),
            'min_score': preferences.get('min_score', 0),
            'sentiment_filter': preferences.get('sentiment_filter', 'all')
        }

    def track_story_view(self, story_id, story_title):
        """Track when user views a story"""
        user_id = self.get_current_user()
        self.db.track_reading(user_id, story_id, story_title)
```

#### Task 4.3: Integrate User Profile with Main App

In `app.py`, add near the beginning (after imports):

```python
# Import user management
from src.phase7.user_manager.user_profile import UserProfileManager
```

Then in the sidebar creation section (around line 80):

```python
# Add user profile management
profile_manager = UserProfileManager()

# Display profile settings
profile_manager.display_profile_settings()

# Display watchlist
profile_manager.display_watchlist()

# Get user filters
user_filters = profile_manager.get_user_filters()
```

When displaying stories (around line 200), apply filters:

```python
# Apply user filters to DataFrame
if user_filters:
    filtered_df = df.copy()

    # Filter by minimum score
    if user_filters.get('min_score', 0) > 0:
        filtered_df = filtered_df[filtered_df['score'] >= user_filters['min_score']]

    # Filter by sentiment
    if user_filters.get('sentiment_filter') != 'all':
        if user_filters['sentiment_filter'] == 'positive':
            filtered_df = filtered_df[filtered_df['sentiment'] > 0.1]
        elif user_filters['sentiment_filter'] == 'negative':
            filtered_df = filtered_df[filtered_df['sentiment'] < -0.1]

    # Highlight preferred topics
    if user_filters.get('topics'):
        # This is simplified - you'd implement actual topic matching
        pass

    # Use filtered DataFrame
    display_df = filtered_df
else:
    display_df = df
```

### Phase 7.2: User Personalization - Week 5

#### Task 5.1: Create Recommendation Engine
**File:** `src/phase7/user_management/recommendations.py`

```python
"""
Generate personalized story recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    """Generate personalized story recommendations"""

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.story_vectors = None
        self.story_titles = []

    def fit(self, stories_df):
        """Fit the recommendation engine with story data"""
        if stories_df.empty:
            return

        self.story_titles = stories_df['title'].tolist()

        # Create TF-IDF vectors
        self.story_vectors = self.tfidf.fit_transform(self.story_titles)

    def get_recommendations(self, user_history, current_stories, top_n=5):
        """
        Get personalized recommendations
        Args:
            user_history: List of story titles user has read
            current_stories: DataFrame of current stories
            top_n: Number of recommendations to return
        Returns:
            List of recommended story indices
        """
        if self.story_vectors is None or current_stories.empty:
            return []

        # Create user profile from reading history
        if user_history:
            user_vector = self.tfidf.transform(user_history)
            user_profile = np.mean(user_vector.toarray(), axis=0)
        else:
            # Default profile (average of all stories)
            user_profile = np.mean(self.story_vectors.toarray(), axis=0)

        # Get similarity scores
        current_titles = current_stories['title'].tolist()
        current_vectors = self.tfidf.transform(current_titles)

        similarities = cosine_similarity([user_profile], current_vectors)[0]

        # Get top recommendations
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return top_indices

    def get_similar_stories(self, story_title, current_stories, top_n=3):
        """
        Find stories similar to a given story
        Args:
            story_title: Title of reference story
            current_stories: DataFrame of current stories
            top_n: Number of similar stories to return
        Returns:
            List of similar story indices
        """
        if self.story_vectors is None or current_stories.empty:
            return []

        # Find the story vector
        try:
            story_idx = self.story_titles.index(story_title)
            story_vector = self.story_vectors[story_idx]
        except ValueError:
            # Story not in training data, create vector on the fly
            story_vector = self.tfidf.transform([story_title])

        # Get similarities
        current_titles = current_stories['title'].tolist()
        current_vectors = self.tfidf.transform(current_titles)

        similarities = cosine_similarity(story_vector, current_vectors)[0]

        # Get top similar (excluding itself)
        top_indices = np.argsort(similarities)[-top_n-1:-1][::-1]

        return top_indices
```

#### Task 5.2: Create Personalization UI Components
**File:** `src/phase7/user_management/ui_components.py`

```python
"""
UI components for user personalization
"""

import streamlit as st
import pandas as pd
from .recommendations import RecommendationEngine

def display_personalized_stories(stories_df, user_history):
    """Display personalized story recommendations"""
    if stories_df.empty:
        st.info("No stories available")
        return

    st.markdown("### 🎯 Personalized For You")

    # Initialize recommendation engine
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
        st.session_state.recommendation_engine.fit(stories_df)

    # Get recommendations
    engine = st.session_state.recommendation_engine
    rec_indices = engine.get_recommendations(
        user_history=user_history,
        current_stories=stories_df,
        top_n=5
    )

    if rec_indices.size > 0:
        # Display recommended stories
        for idx in rec_indices:
            if idx < len(stories_df):
                row = stories_df.iloc[idx]

                # Story card
                with st.container():
                    st.markdown(f"**{row['title']}**")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"Score: {row['score']} | Sentiment: {row['sentiment']:.2f}")

                    with col2:
                        if st.button("Read", key=f"read_{idx}"):
                            # Track reading
                            user_history.append(row['title'])
                            st.success("Added to reading history!")

                    st.markdown("---")
    else:
        st.info("No personalized recommendations available. Read more stories to improve recommendations.")

def display_similar_stories(stories_df, selected_story):
    """Display stories similar to the selected one"""
    if selected_story is None or stories_df.empty:
        return

    st.markdown("### 📖 Similar Stories")

    engine = st.session_state.get('recommendation_engine')
    if not engine:
        return

    # Get similar stories
    similar_indices = engine.get_similar_stories(
        story_title=selected_story,
        current_stories=stories_df,
        top_n=3
    )

    if similar_indices.size > 0:
        for idx in similar_indices:
            if idx < len(stories_df):
                row = stories_df.iloc[idx]
                st.write(f"• {row['title']} (Score: {row['score']})")
    else:
        st.info("No similar stories found")

def display_reading_summary(user_history):
    """Display summary of user's reading history"""
    if not user_history:
        st.info("You haven't read any stories yet")
        return

    st.markdown("### 📚 Your Reading Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Stories Read", len(user_history))

    with col2:
        # Average score of read stories
        avg_score = np.random.randint(100, 1000)  # TODO: Calculate from actual data
        st.metric("Avg Score", avg_score)

    with col3:
        # Reading time
        total_time = len(user_history) * 2  # Assume 2 minutes per story
        st.metric("Reading Time", f"{total_time} min")

    # Recent reads
    st.markdown("**Recently Read:**")
    for title in user_history[-5:]:
        st.write(f"• {title[:50]}...")
```

### Phase 7.3: Multi-Source Integration - Week 6

#### Task 6.1: Create Reddit Connector
**File:** `src/phase7/source_connectors/reddit_connector.py`

```python
"""
Connect to Reddit API to fetch tech stories
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
import os
import time

class RedditConnector:
    """Fetch stories from Reddit"""

    def __init__(self):
        # Note: You'll need to set up Reddit API credentials
        # For now, we'll use read-only mode
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID', ''),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
            user_agent='TechPulse/1.0'
        )

        # Tech subreddits to monitor
        self.subreddits = [
            'programming',
            'technology',
            'MachineLearning',
            'Python',
            'javascript',
            'webdev',
            'datascience',
            'coding'
        ]

    def fetch_stories(self, limit=50, time_filter='day'):
        """
        Fetch stories from tech subreddits
        Args:
            limit: Maximum number of stories per subreddit
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        Returns:
            DataFrame: Stories from Reddit
        """
        stories = []

        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    # Convert post to our format
                    story = {
                        'id': f"reddit_{post.id}",
                        'title': post.title,
                        'url': post.url,
                        'score': post.score,
                        'source': 'Reddit',
                        'subreddit': subreddit_name,
                        'author': str(post.author),
                        'created_at': datetime.fromtimestamp(post.created_utc),
                        'num_comments': post.num_comments,
                        'selftext': post.selftext[:500] if post.selftext else '',
                        'permalink': f"https://reddit.com{post.permalink}"
                    }

                    # Only include posts with minimum score
                    if story['score'] > 10:
                        stories.append(story)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue

        # Convert to DataFrame
        if stories:
            df = pd.DataFrame(stories)

            # Add timestamp column for consistency
            df['timestamp'] = df['created_at'].astype(int)

            # Sort by score
            df = df.sort_values('score', ascending=False)

            return df

        return pd.DataFrame()

    def get_story_details(self, story_id):
        """
        Get detailed information about a story
        Args:
            story_id: Reddit story ID (without 'reddit_' prefix)
        Returns:
            Dict: Story details
        """
        try:
            # Remove 'reddit_' prefix if present
            if story_id.startswith('reddit_'):
                story_id = story_id[7:]

            submission = self.reddit.submission(id=story_id)

            return {
                'title': submission.title,
                'url': submission.url,
                'score': submission.score,
                'selftext': submission.selftext,
                'author': str(submission.author),
                'created_at': datetime.fromtimestamp(submission.created_utc),
                'num_comments': submission.num_comments,
                'upvote_ratio': submission.upvote_ratio,
                'permalink': f"https://reddit.com{submission.permalink}"
            }
        except Exception as e:
            print(f"Error fetching Reddit story {story_id}: {str(e)}")
            return None

# For testing without API credentials
def fetch_mock_reddit_stories():
    """Fetch mock Reddit stories for testing"""
    mock_stories = [
        {
            'id': 'reddit_mock1',
            'title': 'Python 3.13 Released: Performance Improvements and New Syntax',
            'url': 'https://docs.python.org/3.13/whatsnew.html',
            'score': 2847,
            'source': 'Reddit',
            'subreddit': 'Python',
            'author': 'python_dev',
            'created_at': datetime.now() - timedelta(hours=3),
            'num_comments': 342,
            'selftext': 'Python 3.13 brings exciting new features...',
            'permalink': 'https://reddit.com/r/Python/comments/abc123',
            'timestamp': int((datetime.now() - timedelta(hours=3)).timestamp())
        },
        {
            'id': 'reddit_mock2',
            'title': 'Why I Switched from JavaScript to TypeScript in 2024',
            'url': 'https://dev.to/blogger/ts-switch',
            'score': 1523,
            'source': 'Reddit',
            'subreddit': 'webdev',
            'author': 'typescript_fan',
            'created_at': datetime.now() - timedelta(hours=6),
            'num_comments': 189,
            'selftext': 'After years of JavaScript...',
            'permalink': 'https://reddit.com/r/webdev/comments/def456',
            'timestamp': int((datetime.now() - timedelta(hours=6)).timestamp())
        },
        {
            'id': 'reddit_mock3',
            'title': 'New AI Model Achieves Human-Level Performance on Coding Tasks',
            'url': 'https://arxiv.org/abs/2024.12345',
            'score': 3521,
            'source': 'Reddit',
            'subreddit': 'MachineLearning',
            'author': 'ml_researcher',
            'created_at': datetime.now() - timedelta(hours=12),
            'num_comments': 567,
            'selftext': 'Our latest model...',
            'permalink': 'https://reddit.com/r/MachineLearning/comments/ghi789',
            'timestamp': int((datetime.now() - timedelta(hours=12)).timestamp())
        }
    ]

    return pd.DataFrame(mock_stories)
```

#### Task 6.2: Create RSS Feed Connector
**File:** `src/phase7/source_connectors/rss_connector.py`

```python
"""
Connect to RSS feeds to fetch tech news
"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time

class RSSConnector:
    """Fetch stories from RSS feeds"""

    def __init__(self):
        # List of tech RSS feeds
        self.feeds = [
            {
                'name': 'Hacker News',
                'url': 'https://hnrss.org/frontpage',
                'source': 'Hacker News RSS'
            },
            {
                'name': 'TechCrunch',
                'url': 'https://techcrunch.com/feed/',
                'source': 'TechCrunch'
            },
            {
                'name': 'Ars Technica',
                'url': 'https://feeds.arstechnica.com/arstechnica/index',
                'source': 'Ars Technica'
            },
            {
                'name': 'The Verge - Tech',
                'url': 'https://www.theverge.com/rss/tech/index.xml',
                'source': 'The Verge'
            },
            {
                'name': 'MIT Technology Review',
                'url': 'https://www.technologyreview.com/feed/',
                'source': 'MIT Tech Review'
            }
        ]

    def fetch_stories(self, max_age_days=1):
        """
        Fetch stories from all RSS feeds
        Args:
            max_age_days: Only include stories from last N days
        Returns:
            DataFrame: Stories from RSS feeds
        """
        all_stories = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        for feed_info in self.feeds:
            try:
                print(f"Fetching from {feed_info['name']}...")

                # Parse feed
                feed = feedparser.parse(feed_info['url'])

                # Process entries
                for entry in feed.entries:
                    # Parse published date
                    if hasattr(entry, 'published_parsed'):
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed'):
                        published = datetime(*entry.updated_parsed[:6])
                    else:
                        published = datetime.now()

                    # Skip old stories
                    if published < cutoff_date:
                        continue

                    # Create story object
                    story = {
                        'id': f"rss_{hash(entry.link)}",
                        'title': entry.title,
                        'url': entry.link,
                        'score': 0,  # RSS feeds don't have scores
                        'source': feed_info['source'],
                        'feed_name': feed_info['name'],
                        'author': getattr(entry, 'author', 'Unknown'),
                        'created_at': published,
                        'summary': getattr(entry, 'summary', '')[:500],
                        'tags': [tag.term for tag in getattr(entry, 'tags', [])]
                    }

                    all_stories.append(story)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching from {feed_info['name']}: {str(e)}")
                continue

        # Convert to DataFrame
        if all_stories:
            df = pd.DataFrame(all_stories)

            # Add timestamp column
            df['timestamp'] = df['created_at'].astype(int)

            # Sort by publication date
            df = df.sort_values('created_at', ascending=False)

            return df

        return pd.DataFrame()

    def fetch_single_feed(self, feed_url, feed_name):
        """
        Fetch stories from a single RSS feed
        Args:
            feed_url: URL of the RSS feed
            feed_name: Name of the feed
        Returns:
            DataFrame: Stories from the feed
        """
        try:
            feed = feedparser.parse(feed_url)
            stories = []

            for entry in feed.entries:
                # Parse date
                if hasattr(entry, 'published_parsed'):
                    published = datetime(*entry.published_parsed[:6])
                else:
                    published = datetime.now()

                story = {
                    'id': f"rss_{hash(entry.link)}",
                    'title': entry.title,
                    'url': entry.link,
                    'score': 0,
                    'source': feed_name,
                    'created_at': published,
                    'summary': getattr(entry, 'summary', '')[:500]
                }

                stories.append(story)

            return pd.DataFrame(stories)

        except Exception as e:
            print(f"Error fetching feed {feed_name}: {str(e)}")
            return pd.DataFrame()

# Mock RSS data for testing
def fetch_mock_rss_stories():
    """Fetch mock RSS stories for testing"""
    mock_stories = [
        {
            'id': 'rss_mock1',
            'title': 'Microsoft Announces Major AI Integration in Windows 12',
            'url': 'https://techcrunch.com/2024/12/windows12-ai',
            'score': 0,
            'source': 'TechCrunch',
            'feed_name': 'TechCrunch',
            'author': 'TC Reporter',
            'created_at': datetime.now() - timedelta(hours=2),
            'summary': 'Microsoft reveals plans to deeply integrate AI into...',
            'tags': ['AI', 'Microsoft', 'Windows'],
            'timestamp': int((datetime.now() - timedelta(hours=2)).timestamp())
        },
        {
            'id': 'rss_mock2',
            'title': 'OpenAI Releases GPT-5: The Next Generation of Language Models',
            'url': 'https://openai.com/blog/gpt5',
            'score': 0,
            'source': 'The Verge',
            'feed_name': 'The Verge',
            'author': 'AI Editor',
            'created_at': datetime.now() - timedelta(hours=5),
            'summary': 'OpenAI today announced GPT-5, featuring...',
            'tags': ['OpenAI', 'GPT', 'AI'],
            'timestamp': int((datetime.now() - timedelta(hours=5)).timestamp())
        },
        {
            'id': 'rss_mock3',
            'title': 'Cybersecurity Alert: New Vulnerability Affects Millions of IoT Devices',
            'url': 'https://arstechnica.com/security/2024/12/iot-vulnerability',
            'score': 0,
            'source': 'Ars Technica',
            'feed_name': 'Ars Technica',
            'author': 'Security Team',
            'created_at': datetime.now() - timedelta(hours=8),
            'summary': 'A critical vulnerability has been discovered...',
            'tags': ['Security', 'IoT', 'Vulnerability'],
            'timestamp': int((datetime.now() - timedelta(hours=8)).timestamp())
        }
    ]

    return pd.DataFrame(mock_stories)
```

### Phase 7.3: Multi-Source Integration - Weeks 7-9

#### Task 7.1: Create Data Aggregator
**File:** `src/phase7/source_connectors/aggregator.py`

```python
"""
Aggregate data from multiple sources
"""

import pandas as pd
from datetime import datetime, timedelta
from .reddit_connector import RedditConnector, fetch_mock_reddit_stories
from .rss_connector import RSSConnector, fetch_mock_rss_stories
from ..data_loader import fetch_hn_data  # Import existing HN fetcher

class DataAggregator:
    """Aggregate stories from multiple sources"""

    def __init__(self):
        self.reddit_connector = RedditConnector()
        self.rss_connector = RSSConnector()

        # Source priorities for ranking
        self.source_priorities = {
            'Hacker News': 1.0,
            'Reddit': 0.8,
            'TechCrunch': 0.7,
            'Ars Technica': 0.7,
            'The Verge': 0.6,
            'MIT Tech Review': 0.6
        }

    def fetch_all_stories(self, use_mock=False):
        """
        Fetch stories from all configured sources
        Args:
            use_mock: Use mock data for testing
        Returns:
            DataFrame: Aggregated stories from all sources
        """
        all_stories = []

        # 1. Fetch Hacker News stories
        print("Fetching Hacker News stories...")
        try:
            hn_stories = fetch_hn_data(limit=30)
            if not hn_stories.empty:
                hn_stories['source'] = 'Hacker News'
                all_stories.append(hn_stories)
        except Exception as e:
            print(f"Error fetching HN stories: {str(e)}")

        # 2. Fetch Reddit stories
        print("Fetching Reddit stories...")
        if use_mock:
            reddit_stories = fetch_mock_reddit_stories()
        else:
            reddit_stories = self.reddit_connector.fetch_stories(limit=20)

        if not reddit_stories.empty:
            all_stories.append(reddit_stories)

        # 3. Fetch RSS stories
        print("Fetching RSS stories...")
        if use_mock:
            rss_stories = fetch_mock_rss_stories()
        else:
            rss_stories = self.rss_connector.fetch_stories(max_age_days=1)

        if not rss_stories.empty:
            all_stories.append(rss_stories)

        # Combine all stories
        if all_stories:
            combined_df = pd.concat(all_stories, ignore_index=True)

            # Deduplicate based on URL similarity
            combined_df = self.deduplicate_stories(combined_df)

            # Normalize scores across sources
            combined_df = self.normalize_scores(combined_df)

            # Sort by normalized score
            combined_df = combined_df.sort_values('normalized_score', ascending=False)

            # Add combined metadata
            combined_df['fetched_at'] = datetime.now()

            return combined_df

        return pd.DataFrame()

    def deduplicate_stories(self, df):
        """
        Remove duplicate stories based on URL and title similarity
        Args:
            df: DataFrame of stories
        Returns:
            DataFrame: Deduplicated stories
        """
        if df.empty:
            return df

        # Simple deduplication based on exact URL matches
        deduped = df.drop_duplicates(subset=['url'], keep='first')

        # TODO: Implement fuzzy matching for similar titles

        return deduped

    def normalize_scores(self, df):
        """
        Normalize scores across different sources
        Args:
            df: DataFrame of stories
        Returns:
            DataFrame: Stories with normalized scores
        """
        if df.empty:
            return df

        # Create normalized_score column
        df['normalized_score'] = df.apply(self.calculate_normalized_score, axis=1)

        return df

    def calculate_normalized_score(self, row):
        """
        Calculate normalized score for a story
        Args:
            row: Story row
        Returns:
            float: Normalized score
        """
        source = row.get('source', '')
        original_score = row.get('score', 0)

        # Get source priority
        priority = self.source_priorities.get(source, 0.5)

        # Special handling for sources without scores
        if original_score == 0 and source in ['TechCrunch', 'Ars Technica', 'The Verge']:
            # Estimate score based on recency and other factors
            hours_ago = (datetime.now() - row['created_at']).total_seconds() / 3600
            estimated_score = max(100 - hours_ago * 2, 10)
            original_score = estimated_score

        # Apply source priority
        normalized = original_score * priority

        # Add bonus for recent stories
        hours_ago = (datetime.now() - row['created_at']).total_seconds() / 3600
        if hours_ago < 6:
            normalized *= 1.2
        elif hours_ago < 24:
            normalized *= 1.1

        return int(normalized)

    def get_source_stats(self, df):
        """
        Get statistics about story sources
        Args:
            df: DataFrame of stories
        Returns:
            Dict: Source statistics
        """
        if df.empty:
            return {}

        stats = {}

        # Count stories per source
        source_counts = df['source'].value_counts()
        stats['story_counts'] = source_counts.to_dict()

        # Average score per source
        avg_scores = df.groupby('source')['normalized_score'].mean()
        stats['avg_scores'] = avg_scores.to_dict()

        # Top sources
        stats['top_sources'] = source_counts.head(3).index.tolist()

        return stats

# For testing the aggregator
def test_aggregator():
    """Test the data aggregator with mock data"""
    aggregator = DataAggregator()

    # Fetch stories with mock data
    stories = aggregator.fetch_all_stories(use_mock=True)

    print(f"Fetched {len(stories)} stories from all sources")

    if not stories.empty:
        print("\nSource distribution:")
        print(stories['source'].value_counts())

        print("\nTop 5 stories by normalized score:")
        for _, row in stories.head(5).iterrows():
            print(f"- {row['title'][:60]}... ({row['source']}) - Score: {row['normalized_score']}")

    return stories
```

#### Task 7.2: Update Main App with Multi-Source Support

In `app.py`, update the data loading section:

```python
# Near the top, add imports
from src.phase7.source_connectors.aggregator import DataAggregator

# Replace the existing refresh_data function with:
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_multi_source_data():
    """Load data from multiple sources"""
    aggregator = DataAggregator()

    try:
        # Use mock data for now, set use_mock=False for real data
        df = aggregator.fetch_all_stories(use_mock=True)

        if not df.empty:
            # Add sentiment analysis
            from data_loader import analyze_sentiment
            df = analyze_sentiment(df)

            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def refresh_data():
    """Refresh data from all sources"""
    st.cache_data.clear()
    st.success("Data refreshed from all sources!")

# In the main app flow, replace data loading with:
if 'df' not in st.session_state or st.session_state.real_time_mode:
    with st.spinner("Fetching stories from multiple sources..."):
        st.session_state.df = load_multi_source_data()
        if st.session_state.df.empty:
            st.error("Unable to fetch stories. Please try again.")
            st.stop()

df = st.session_state.df

# Add source filter in sidebar
st.sidebar.markdown("### 📡 News Sources")
all_sources = ['All'] + list(df['source'].unique()) if not df.empty else ['All']
selected_source = st.sidebar.selectbox("Filter by source:", all_sources)

if selected_source != 'All':
    df = df[df['source'] == selected_source]

# Add source statistics
if not df.empty:
    source_stats = df['source'].value_counts()
    st.sidebar.markdown("**Source Breakdown:**")
    for source, count in source_stats.items():
        st.sidebar.write(f"• {source}: {count} stories")
```

#### Task 7.3: Create Comprehensive Tests
**File:** `test/phase7/test_multi_source.py`

```python
"""
Tests for multi-source data aggregation
"""

import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from phase7.source_connectors.aggregator import DataAggregator, test_aggregator

class TestMultiSource(unittest.TestCase):
    """Test multi-source data aggregation"""

    def setUp(self):
        self.aggregator = DataAggregator()

    def test_fetch_all_stories_mock(self):
        """Test fetching stories with mock data"""
        stories = self.aggregator.fetch_all_stories(use_mock=True)

        # Check that we got stories
        self.assertFalse(stories.empty)
        self.assertGreater(len(stories), 0)

        # Check required columns
        required_columns = ['id', 'title', 'url', 'source', 'timestamp', 'normalized_score']
        for col in required_columns:
            self.assertIn(col, stories.columns)

    def test_source_diversity(self):
        """Test that stories come from multiple sources"""
        stories = self.aggregator.fetch_all_stories(use_mock=True)

        if not stories.empty:
            sources = stories['source'].unique()
            self.assertGreater(len(sources), 1)

            # Check for expected sources
            expected_sources = ['Hacker News', 'Reddit', 'TechCrunch', 'Ars Technica', 'The Verge']
            for source in expected_sources:
                self.assertIn(source, sources, f"Missing source: {source}")

    def test_score_normalization(self):
        """Test that scores are properly normalized"""
        stories = self.aggregator.fetch_all_stories(use_mock=True)

        if not stories.empty:
            # Check that normalized scores exist
            self.assertIn('normalized_score', stories.columns)

            # Check that scores are reasonable
            max_score = stories['normalized_score'].max()
            min_score = stories['normalized_score'].min()

            self.assertGreater(max_score, 0)
            self.assertGreaterEqual(min_score, 0)

    def test_deduplication(self):
        """Test story deduplication"""
        # Create test data with duplicates
        test_data = pd.DataFrame([
            {'url': 'http://example.com/1', 'title': 'Test 1', 'source': 'Source A'},
            {'url': 'http://example.com/1', 'title': 'Test 1', 'source': 'Source B'},
            {'url': 'http://example.com/2', 'title': 'Test 2', 'source': 'Source A'}
        ])

        deduped = self.aggregator.deduplicate_stories(test_data)

        # Should have removed duplicate
        self.assertEqual(len(deduped), 2)

    def test_source_stats(self):
        """Test source statistics generation"""
        stories = self.aggregator.fetch_all_stories(use_mock=True)

        if not stories.empty:
            stats = self.aggregator.get_source_stats(stories)

            # Check required stats
            self.assertIn('story_counts', stats)
            self.assertIn('avg_scores', stats)
            self.assertIn('top_sources', stats)

            # Check story counts
            self.assertGreater(len(stats['story_counts']), 0)

if __name__ == '__main__':
    unittest.main()
```

## Testing Instructions

### Running Tests

1. **Run all Phase 7 tests:**
```bash
python -m pytest test/phase7/ -v
```

2. **Run specific test files:**
```bash
python -m pytest test/phase7/test_predictor.py -v
python -m pytest test/phase7/test_multi_source.py -v
```

3. **Run tests with coverage:**
```bash
coverage run -m pytest test/phase7/
coverage report -m
```

### Test Coverage Goals
- Aim for at least 90% coverage on new code
- Ensure all public methods have tests
- Test error conditions and edge cases

## Troubleshooting Guide

### Common Issues

1. **Import Errors:**
   - Ensure src directory is in Python path
   - Check that all required dependencies are installed
   - Run: `export PYTHONPATH="${PYTHONPATH}:./src"`

2. **Reddit API Issues:**
   - You'll need to set up Reddit API credentials
   - For testing, use the mock functions provided
   - Set environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET

3. **Database Errors:**
   - Ensure data directory exists
   - Check SQLite file permissions
   - Run: `mkdir -p data`

4. **Performance Issues:**
   - Use st.cache_data for expensive operations
   - Limit the number of concurrent API calls
   - Implement rate limiting

### Debug Mode

Add this to your code for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

After completing Phase 7:
1. Deploy to staging environment for testing
2. Collect user feedback
3. Monitor performance metrics
4. Plan Phase 8 based on learnings

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Reddit API Documentation](https://praw.readthedocs.io/)
- [SQLite Documentation](https://docs.python.org/3/library/sqlite3.html)

Remember to:
- Commit your changes regularly
- Write tests before writing code
- Ask for help when stuck
- Document any deviations from this plan