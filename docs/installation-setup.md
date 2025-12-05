# Tech-Pulse Installation & Setup Guide

## Overview

This guide walks you through installing and running Tech-Pulse on your local machine. While the easiest way to use Tech-Pulse is through our [live dashboard](https://tech-pulse.streamlit.app), local installation gives you more control and customization options.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Options](#installation-options)
3. [Local Setup Steps](#local-setup-steps)
4. [Docker Installation](#docker-installation)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Setup](#advanced-setup)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.13 or newer
- **RAM**: 4GB minimum
- **Disk Space**: 1GB free space
- **Internet**: Stable connection for data fetching

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.13+
- **RAM**: 8GB or more
- **Disk Space**: 2GB free space
- **Internet**: High-speed connection
- **Browser**: Chrome, Firefox, Safari, or Edge (latest version)

### What Gets Installed
- Python packages (~500MB)
- Machine learning models (~100MB, downloaded on first run)
- Vector database storage (~10-50MB)
- Cache files (~100MB)

## Installation Options

### Option 1: Quick Install (Recommended)

```bash
# One-line installation (requires curl)
curl -sSL https://raw.githubusercontent.com/dustinober1/tech-pulse/main/install.sh | bash
```

### Option 2: Manual Install
Follow the detailed steps below for full control over the installation process.

### Option 3: Docker Install
Use Docker for isolated environment (see Docker section below).

## Local Setup Steps

### Step 1: Prerequisites

#### Install Python 3.13+

**Windows:**
1. Download from [python.org](https://python.org)
2. Run installer and check "Add to PATH"
3. Verify installation:
   ```cmd
   python --version
   ```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.13

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.13 python3.13-pip python3.13-venv
```

#### Install Git
```bash
# Windows: Download from git-scm.com
# macOS: brew install git
# Linux: sudo apt install git
```

### Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/dustinober1/tech-pulse.git

# Navigate to project directory
cd tech-pulse

# Verify structure
ls -la
```

You should see files like:
- `app.py` - Main dashboard application
- `data_loader.py` - Data processing module
- `requirements.txt` - Python dependencies

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python
```

### Step 4: Install Dependencies

#### Option A: Install All Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For dashboard-specific dependencies
pip install -r requirements-dashboard.txt
```

#### Option B: Install Manually
```bash
# Core dependencies
pip install pandas requests nltk

# Analysis modules
pip install bertopic sentence-transformers chromadb

# Dashboard
pip install streamlit plotly

# Additional modules
pip install numpy psutil tqdm
```

### Step 5: Download Language Models

```bash
# Download NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
print('NLTK data downloaded successfully')
"
```

### Step 6: Verify Installation

```bash
# Run tests to verify installation
python test/run_tests.py

# Expected output: All tests should pass
```

### Step 7: Run the Application

```bash
# Start the dashboard
streamlit run app.py

# Open browser to http://localhost:8501
```

You should see:
- Terminal showing server status
- Browser opening automatically
- Dashboard loading with sample data

## Docker Installation

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 4GB+ RAM allocated to Docker

### Quick Docker Setup

```bash
# Clone repository
git clone https://github.com/dustinober1/tech-pulse.git
cd tech-pulse

# Build and run with Docker Compose
docker-compose up

# Or build manually
docker build -t tech-pulse .
docker run -p 8501:8501 tech-pulse
```

### Docker Compose Configuration

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  tech-pulse:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Docker Troubleshooting

**Build Fails:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

**Permission Errors (Linux):**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again
```

## Configuration

### Environment Variables

Create `.env` file (optional):
```env
# API Keys (optional)
OPENAI_API_KEY=your_openai_key_here

# Data Settings
DEFAULT_STORIES=30
CACHE_DURATION=3600

# Performance
MAX_WORKERS=4
MEMORY_LIMIT=4096

# Development
DEBUG=False
LOG_LEVEL=INFO
```

### Custom Configuration

Edit `dashboard_config.py`:
```python
# Change default settings
DEFAULT_SETTINGS = {
    'default_stories': 50,  # Default: 30
    'min_stories': 10,
    'max_stories': 100
}

# Customize colors
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4'
}

# Adjust performance
CHART_CONFIG = {
    'animation': False,  # Disable for better performance
    'responsive': True
}
```

### Database Configuration

For persistent vector storage:
```python
# In vector_cache_manager.py
VECTOR_DB_SETTINGS = {
    'persist_directory': './data/vector_db',
    'collection_name': 'tech_stories',
    'distance_metric': 'cosine'
}
```

## Troubleshooting

### Common Installation Issues

#### Python Version Error
```bash
# Error: Python 3.13+ required
# Solution: Install correct Python version
python --version  # Should show 3.13.x
```

#### Permission Denied (macOS/Linux)
```bash
# Error: Permission denied when installing packages
# Solution: Use user directory or virtual environment
pip install --user -r requirements.txt
# OR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### SSL Certificate Error (Windows)
```bash
# Error: SSL: CERTIFICATE_VERIFY_FAILED
# Solution: Update certificates or use trusted store
pip install --upgrade pip setuptools --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

#### Memory Error
```bash
# Error: MemoryError during installation
# Solution: Install packages individually
pip install pandas
pip install numpy
pip install -r requirements.txt --no-deps
```

### Runtime Issues

#### Module Not Found Error
```bash
# Error: ModuleNotFoundError: No module named 'xxx'
# Solution: Install missing module
pip install xxx

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

#### Port Already in Use
```bash
# Error: Port 8501 is already in use
# Solution: Use different port
streamlit run app.py --server.port 8502
```

#### NLTK Data Missing
```bash
# Error: Resource 'punkt' not found
# Solution: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

#### Performance Issues
- **Slow Loading**: Reduce default story count
- **High Memory**: Close other applications
- **Analysis Errors**: Check internet connection

### Getting Help

1. **Check Logs**:
   ```bash
   # Enable debug logging
   streamlit run app.py --logger.level debug
   ```

2. **Run Diagnostics**:
   ```bash
   python -c "
   import sys
   print('Python:', sys.version)
   import pandas as pd
   print('Pandas:', pd.__version__)
   import streamlit as st
   print('Streamlit:', st.__version__)
   "
   ```

3. **Search Issues**:
   - Check [GitHub Issues](https://github.com/dustinober1/tech-pulse/issues)
   - Review [FAQ](docs/faq.md)

## Advanced Setup

### Production Deployment

#### Using Streamlit Sharing
1. Connect GitHub repository
2. Set Python version to 3.13
3. Add requirements.txt
4. Deploy automatically

#### Custom Server
```bash
# Install Streamlit on server
pip install streamlit

# Run with custom configuration
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

### Multi-User Setup

#### User Authentication
```python
# In app.py
import streamlit_authenticator as stauth

# Add authentication
names = ['John', 'Jane']
usernames = ['john', 'jane']
passwords = ['123', '456']

authenticator = stauth.Authenticate(names, usernames, passwords)
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show dashboard
else:
    st.error('Username/password is incorrect')
```

#### Database Backend
```python
# Using PostgreSQL for persistent storage
import psycopg2

# Connection settings
DATABASE_URL = "postgresql://user:password@localhost/techpulse"
```

### Custom Data Sources

#### Add New Source
```python
# In src/phase7/source_connectors/
class CustomConnector:
    def fetch_stories(self, limit=30):
        # Implement custom API logic
        return stories
```

#### Configure Sources
```python
# In data_loader.py
SOURCES = {
    'hackernews': HackerNewsConnector(),
    'reddit': RedditConnector(),
    'custom': CustomConnector()
}
```

### Performance Optimization

#### Caching Strategy
```python
# Use Redis for caching
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_cached_data():
    # Fetch and cache data
    pass
```

#### Parallel Processing
```python
# Use concurrent processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(fetch_source, source) for source in sources]
    results = [f.result() for f in futures]
```

## Next Steps

After successful installation:

1. **Read the User Guide**: [docs/user-guide.md](user-guide.md)
2. **Check the Quick Start**: [docs/quick-start.md](quick-start.md)
3. **Explore Features**: Semantic search, predictive analytics, reports
4. **Customize**: Adjust settings to your needs
5. **Join Community**: Contribute and get support

## Need More Help?

- **Documentation**: [docs/](../docs/)
- **GitHub Issues**: Report installation problems
- **Community Forum**: Connect with other users
- **Email Support**: For enterprise installations

---

**Happy analyzing with Tech-Pulse!** 🚀

Remember: You can always use the live dashboard at [https://tech-pulse.streamlit.app](https://tech-pulse.streamlit.app) if installation becomes challenging.