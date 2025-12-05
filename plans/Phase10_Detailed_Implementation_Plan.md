# Phase 10: Automation & Deployment - Detailed Implementation Plan

## Overview

Phase 10 focuses on productionizing the Tech-Pulse dashboard with enterprise-grade automation, deployment pipelines, monitoring, and maintenance capabilities. This phase ensures the application is production-ready, scalable, secure, and maintainable.

## Prerequisites

- Completed Phase 9: Advanced Analytics & Intelligence
- All tests passing from previous phases
- Access to cloud hosting platform (AWS, GCP, or Azure)
- Domain name (optional for production deployment)
- SSL certificate (if not using cloud provider's managed certificates)

## Work Packages

### WP 10.1: CI/CD Pipeline Implementation (8 hours)

**Objective**: Establish automated testing, building, and deployment pipelines

#### Task 10.1.1: GitHub Actions Workflow Setup (2 hours)
**Files to Create**: `.github/workflows/ci.yml`, `.github/workflows/cd.yml`
**Dependencies**: None
**Implementation Details**:
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest-cov

    - name: Run linting
      run: |
        pip install flake8 black isort
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check .
        isort --check-only .

    - name: Run tests
      run: |
        pytest test/ -v --cov=. --cov-report=xml --cov-report=html

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

**Acceptance Criteria**:
- [ ] CI pipeline runs on all pushes and PRs
- [ ] Tests pass on Python 3.9, 3.10, and 3.11
- [ ] Code coverage reports are generated
- [ ] Linting checks are enforced
- [ ] Pipeline completes within 10 minutes

#### Task 10.1.2: Docker Containerization (2 hours)
**Files to Create**: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `docker-compose.prod.yml`
**Dependencies**: Application code must be complete
**Implementation Details**:
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py"]
```

**Acceptance Criteria**:
- [ ] Docker image builds successfully
- [ ] Application runs in container without errors
- [ ] All features work correctly in containerized environment
- [ ] Image size is optimized (< 1GB)
- [ ] Health checks pass

#### Task 10.1.3: Deployment Pipeline (2 hours)
**Files to Create**: `.github/workflows/deploy.yml`, `scripts/deploy.sh`
**Dependencies**: Docker setup, cloud provider account
**Implementation Details**:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build, tag, and push image to Amazon ECR
      id: build-image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster tech-pulse --service tech-pulse-service --force-new-deployment
```

**Acceptance Criteria**:
- [ ] Deployment triggers on main branch pushes
- [ ] Docker images are pushed to container registry
- [ ] Application is deployed to production environment
- [ ] Zero-downtime deployment is implemented
- [ ] Rollback mechanism is available

#### Task 10.1.4: Environment Management (2 hours)
**Files to Create**: `.env.example`, `scripts/setup-env.sh`, `config/environments.py`
**Dependencies**: Deployment pipeline
**Implementation Details**:
```python
# config/environments.py
import os
from typing import Dict, Any

class Environment:
    def __init__(self):
        self.env = os.getenv('ENVIRONMENT', 'development')
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        base_config = {
            'debug': False,
            'log_level': 'INFO',
            'cache_ttl': 3600,
            'max_workers': 4,
        }

        if self.env == 'development':
            base_config.update({
                'debug': True,
                'log_level': 'DEBUG',
                'cache_ttl': 300,
            })
        elif self.env == 'production':
            base_config.update({
                'debug': False,
                'log_level': 'WARNING',
                'cache_ttl': 7200,
                'max_workers': 8,
            })

        return base_config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
```

**Acceptance Criteria**:
- [ ] Environment-specific configurations work
- [ ] Secrets are properly managed
- [ ] Development and production environments are isolated
- [ ] Environment validation is implemented

---

### WP 10.2: Production Infrastructure Setup (10 hours)

**Objective**: Set up scalable, secure production infrastructure

#### Task 10.2.1: Cloud Infrastructure Setup (3 hours)
**Files to Create**: `infrastructure/aws/cloudformation.yml`, `infrastructure/aws/terraform.tf`
**Dependencies**: Cloud provider account, deployment pipeline
**Implementation Details**:
```yaml
# infrastructure/aws/cloudformation.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Tech-Pulse Production Infrastructure'

Parameters:
  DomainName:
    Type: String
    Default: 'tech-pulse.example.com'
  Environment:
    Type: String
    Default: 'production'
    AllowedValues: ['development', 'staging', 'production']

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-tech-pulse-vpc'

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: true

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${Environment}-tech-pulse'
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      RequiresCompatibilities:
        - FARGATE
      NetworkMode: awsvpc
      Cpu: 512
      Memory: 1024
      ExecutionRoleArn: !Ref ECSExecutionRole
      TaskRoleArn: !Ref ECSTaskRole
      ContainerDefinitions:
        - Name: tech-pulse-app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/tech-pulse:latest'
          PortMappings:
            - ContainerPort: 8501
          Environment:
            - Name: ENVIRONMENT
              Value: !Ref Environment
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref CloudWatchLogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs
```

**Acceptance Criteria**:
- [ ] VPC and networking are properly configured
- [ ] ECS cluster is created and running
- [ ] Load balancer is configured
- [ ] Auto-scaling is implemented
- [ ] Infrastructure can be deployed via IaC

#### Task 10.2.2: Database Setup (2 hours)
**Files to Create**: `infrastructure/database/init.sql`, `scripts/backup-db.sh`
**Dependencies**: Infrastructure setup
**Implementation Details**:
```sql
-- infrastructure/database/init.sql
-- PostgreSQL initialization script for production

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application user
CREATE USER tech_pulse_app WITH PASSWORD '${APP_PASSWORD}';
CREATE USER tech_pulse_readonly WITH PASSWORD '${READONLY_PASSWORD}';

-- Create schemas
CREATE SCHEMA IF NOT EXISTS app_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions
GRANT USAGE ON SCHEMA app_data TO tech_pulse_app;
GRANT CREATE ON SCHEMA app_data TO tech_pulse_app;
GRANT USAGE ON SCHEMA analytics TO tech_pulse_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO tech_pulse_readonly;

-- Create tables for persistent data
CREATE TABLE IF NOT EXISTS app_data.user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    preferences JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app_data.analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON app_data.user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_timestamp ON app_data.analytics_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON app_data.analytics_events(event_type);
```

**Acceptance Criteria**:
- [ ] PostgreSQL RDS instance is created
- [ ] Database is properly secured
- [ ] Backup strategy is implemented
- [ ] Monitoring and alerts are configured
- [ ] Migration scripts work correctly

#### Task 10.2.3: Monitoring and Logging Setup (2 hours)
**Files to Create**: `config/monitoring.py`, `infrastructure/monitoring/grafana-dashboards.json`
**Dependencies**: Database setup, application deployment
**Implementation Details**:
```python
# config/monitoring.py
import logging
import json
import time
from typing import Dict, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

class ProductionMonitoring:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('tech_pulse')
        logger.setLevel(logging.INFO)

        # CloudWatch Logs handler
        # (Would be configured based on environment)

        return logger

    def track_metric(self, metric_name: str, value: float, unit: str = 'Count'):
        """Send custom metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='TechPulse',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
        except ClientError as e:
            self.logger.error(f"Failed to track metric {metric_name}: {e}")

    def track_request(self, endpoint: str, duration: float, status_code: int):
        """Track API request metrics"""
        self.track_metric('RequestDuration', duration, 'Seconds')
        self.track_metric('RequestCount', 1, 'Count')

        if status_code >= 400:
            self.track_metric('ErrorCount', 1, 'Count')

    def track_data_refresh(self, source: str, records_count: int, duration: float):
        """Track data refresh operations"""
        self.track_metric('DataRefreshDuration', duration, 'Seconds')
        self.track_metric('RecordsProcessed', records_count, 'Count')
        self.track_metric(f'{source}RecordsProcessed', records_count, 'Count')
```

**Acceptance Criteria**:
- [ ] CloudWatch metrics are collected
- [ ] Application logs are centralized
- [ ] Grafana dashboards are configured
- [ ] Alert notifications are set up
- [ ] Performance monitoring is active

#### Task 10.2.4: Security Hardening (2 hours)
**Files to Create**: `security/security-groups.yml`, `security/iam-roles.yml`, `config/security.py`
**Dependencies**: Infrastructure setup
**Implementation Details**:
```python
# config/security.py
import os
import hashlib
import hmac
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import secrets

class SecurityManager:
    def __init__(self):
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key().decode()
            # In production, this should be stored securely
        return key.encode()

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def generate_api_key(self, user_id: str) -> str:
        """Generate secure API key for user"""
        timestamp = str(int(time.time()))
        message = f"{user_id}:{timestamp}"
        signature = hmac.new(
            self.encryption_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"tp_{signature[:32]}"

    def validate_api_key(self, api_key: str, user_id: str) -> bool:
        """Validate API key"""
        # Implementation would validate against stored keys
        return api_key.startswith("tp_") and len(api_key) == 35

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize user input"""
        if isinstance(data, str):
            # Remove potentially harmful characters
            dangerous_chars = ['<', '>', '"', "'", '&', 'javascript:', 'data:']
            for char in dangerous_chars:
                data = data.replace(char, '')
        return data
```

**Acceptance Criteria**:
- [ ] Security groups are properly configured
- [ ] IAM roles follow least privilege principle
- [ ] Data encryption is implemented
- [ ] API security is enforced
- [ ] Security scanning is automated

#### Task 10.2.5: SSL/TLS Configuration (1 hour)
**Files to Create**: `infrastructure/ssl/cert-manager.yml`, `nginx/nginx.conf`
**Dependencies**: Infrastructure setup, domain name
**Implementation Details**:
```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

    upstream tech_pulse_app {
        server app:8501;
    }

    server {
        listen 443 ssl http2;
        server_name tech-pulse.example.com;

        ssl_certificate /etc/ssl/certs/app.crt;
        ssl_certificate_key /etc/ssl/private/app.key;

        # Main application
        location / {
            proxy_pass http://tech_pulse_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://tech_pulse_app;
            # ... same proxy settings
        }

        # Login endpoint with stricter rate limiting
        location /login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://tech_pulse_app;
            # ... same proxy settings
        }
    }
}
```

**Acceptance Criteria**:
- [ ] SSL certificate is installed and valid
- [ ] HTTPS is enforced
- [ ] Security headers are configured
- [ ] Certificate auto-renewal is set up
- [ ] SSL/TLS configuration passes security tests

---

### WP 10.3: Performance Optimization (6 hours)

**Objective**: Optimize application performance for production workloads

#### Task 10.3.1: Caching Strategy Implementation (2 hours)
**Files to Create**: `src/cache/redis_cache.py`, `config/cache_config.py`
**Dependencies**: Infrastructure setup, Redis deployment
**Implementation Details**:
```python
# src/cache/redis_cache.py
import redis
import json
import pickle
from typing import Any, Optional, Union
from datetime import timedelta
import logging

class RedisCache:
    def __init__(self, host: str, port: int = 6379, password: Optional[str] = None):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Union[int, timedelta] = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            serialized_value = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def get_many(self, keys: list) -> dict:
        """Get multiple values from cache"""
        try:
            values = self.redis_client.mget(keys)
            result = {}
            for i, key in enumerate(keys):
                if values[i]:
                    result[key] = pickle.loads(values[i])
            return result
        except Exception as e:
            self.logger.error(f"Cache get_many error: {e}")
            return {}

    def set_many(self, mapping: dict, ttl: Union[int, timedelta] = 3600) -> bool:
        """Set multiple values in cache"""
        try:
            pipe = self.redis_client.pipeline()
            for key, value in mapping.items():
                serialized_value = pickle.dumps(value)
                pipe.setex(key, ttl, serialized_value)
            pipe.execute()
            return True
        except Exception as e:
            self.logger.error(f"Cache set_many error: {e}")
            return False
```

**Acceptance Criteria**:
- [ ] Redis cache is implemented and configured
- [ ] Cache hit rate is above 80%
- [ ] API response times are improved by 50%
- [ ] Cache invalidation strategy works correctly
- [ ] Cache monitoring is implemented

#### Task 10.3.2: Database Query Optimization (1.5 hours)
**Files to Create**: `src/database/query_optimizer.py`, `migrations/add_indexes.sql`
**Dependencies**: Database setup
**Implementation Details**:
```python
# src/database/query_optimizer.py
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Tuple
import logging
from contextlib import contextmanager

class QueryOptimizer:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        """Database connection context manager"""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def batch_insert(self, table: str, data: List[Dict[str, Any]], batch_size: int = 1000):
        """Efficient batch insert operation"""
        if not data:
            return

        columns = data[0].keys()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                values = [[row[col] for col in columns] for row in batch]

                query = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """

                execute_values(cursor, query, values)
                self.logger.info(f"Inserted batch {i//batch_size + 1}, size: {len(batch)}")

    def get_top_stories_optimized(self, limit: int = 10, filters: Dict = None) -> List[Dict]:
        """Optimized query for top stories"""
        base_query = """
            SELECT id, title, url, score, sentiment_label, topic_keyword
            FROM stories
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """

        params = []
        if filters:
            if 'topic' in filters:
                base_query += " AND topic_keyword = %s"
                params.append(filters['topic'])
            if 'sentiment' in filters:
                base_query += " AND sentiment_label = %s"
                params.append(filters['sentiment'])

        base_query += " ORDER BY score DESC, created_at DESC LIMIT %s"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(base_query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def analyze_query_performance(self) -> List[Dict]:
        """Get slow queries and performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get slow queries
            cursor.execute("""
                SELECT query, mean_time, calls, total_time
                FROM pg_stat_statements
                WHERE mean_time > 100  -- queries taking more than 100ms
                ORDER BY mean_time DESC
                LIMIT 10
            """)

            slow_queries = []
            for row in cursor.fetchall():
                slow_queries.append({
                    'query': row[0][:100],  # Truncate long queries
                    'mean_time': row[1],
                    'calls': row[2],
                    'total_time': row[3]
                })

            return slow_queries
```

**Acceptance Criteria**:
- [ ] Database indexes are optimized
- [ ] Query execution times are reduced by 60%
- [ ] Batch operations are implemented
- [ ] Connection pooling is configured
- [ ] Query performance monitoring is active

#### Task 10.3.3: Async Processing Implementation (1.5 hours)
**Files to Create**: `src/async/task_queue.py`, `src/async/workers.py`, `config/celery_config.py`
**Dependencies**: Redis cache, infrastructure setup
**Implementation Details**:
```python
# src/async/task_queue.py
from celery import Celery
from celery.schedules import crontab
import logging
from typing import Dict, Any

# Celery configuration
celery_app = Celery('tech_pulse')
celery_app.config_from_object('config.celery_config')

@celery_app.task(bind=True, max_retries=3)
def refresh_hackernews_data(self):
    """Background task to refresh HackerNews data"""
    try:
        from data_loader import fetch_hn_data, analyze_sentiment, get_topics

        # Fetch new data
        df = fetch_hn_data(limit=100)
        if not df.empty:
            # Process data
            df = analyze_sentiment(df)
            df = get_topics(df)

            # Update cache
            from cache_manager import CacheManager
            cache = CacheManager()
            cache.set('hn_stories', df, ttl=3600)

            return {
                'status': 'success',
                'stories_count': len(df),
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        logging.error(f"HN data refresh failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (self.request.retries + 1))
        return {
            'status': 'error',
            'error': str(e),
            'retries': self.request.retries
        }

@celery_app.task
def generate_pdf_report(user_id: str, stories_count: int, include_charts: bool):
    """Background task to generate PDF report"""
    try:
        from src.pdf_generator.report_builder import ExecutiveBriefingBuilder

        builder = ExecutiveBriefingBuilder()
        pdf_bytes = builder.generate_briefing(
            stories_count=stories_count,
            include_charts=include_charts
        )

        # Store PDF for download
        import uuid
        import os
        pdf_id = str(uuid.uuid4())
        pdf_path = f"temp/pdfs/{pdf_id}.pdf"

        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

        return {
            'status': 'success',
            'pdf_id': pdf_id,
            'user_id': user_id,
            'file_size': len(pdf_bytes)
        }

    except Exception as e:
        logging.error(f"PDF generation failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'user_id': user_id
        }

@celery_app.task
def send_weekly_digest():
    """Weekly task to send digest emails"""
    try:
        # Generate weekly report
        from src.email.email_sender import EmailSender
        from analytics.weekly_report import WeeklyReportGenerator

        generator = WeeklyReportGenerator()
        report = generator.generate_report()

        sender = EmailSender()
        # Send to subscribed users
        # Implementation would fetch user list and send emails

        return {
            'status': 'success',
            'report_id': report['id'],
            'sent_count': 0  # Actual count would be implemented
        }

    except Exception as e:
        logging.error(f"Weekly digest failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# Schedule periodic tasks
celery_app.conf.beat_schedule = {
    'refresh-hn-data': {
        'task': 'src.async.task_queue.refresh_hackernews_data',
        'schedule': crontab(minute=0),  # Every hour
    },
    'weekly-digest': {
        'task': 'src.async.task_queue.send_weekly_digest',
        'schedule': crontab(day_of_week=1, hour=9, minute=0),  # Monday 9 AM
    },
}
```

**Acceptance Criteria**:
- [ ] Celery workers are configured and running
- [ ] Background tasks execute successfully
- [ ] Task monitoring is implemented
- [ ] Error handling and retries work correctly
- [ ] Periodic tasks are scheduled and executed

#### Task 10.3.4: CDN and Static Asset Optimization (1 hour)
**Files to Create**: `config/storage_config.py`, `scripts/deploy-static.sh`
**Dependencies**: Infrastructure setup, S3 bucket
**Implementation Details**:
```python
# config/storage_config.py
import boto3
from botocore.exceptions import ClientError
import logging
from typing import Optional

class CDNManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.cloudfront_client = boto3.client('cloudfront')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.distribution_id = os.getenv('CLOUDFRONT_DISTRIBUTION_ID')

    def upload_static_asset(self, local_path: str, s3_key: str, content_type: str) -> bool:
        """Upload static asset to S3 with CDN optimization"""
        try:
            # Determine cache control based on file type
            if s3_key.endswith(('.js', '.css')):
                cache_control = 'public, max-age=31536000, immutable'
            elif s3_key.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                cache_control = 'public, max-age=2592000'
            else:
                cache_control = 'public, max-age=3600'

            extra_args = {
                'ContentType': content_type,
                'CacheControl': cache_control,
                'Metadata': {
                    'version': self._get_file_hash(local_path)
                }
            }

            # Enable compression for text files
            if content_type.startswith('text/'):
                extra_args['ContentEncoding'] = 'gzip'

            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )

            # Invalidate CDN cache if needed
            if self.distribution_id and not s3_key.endswith(('.js', '.css')):
                self._invalidate_cdn_path(s3_key)

            return True

        except ClientError as e:
            logging.error(f"Failed to upload {s3_key}: {e}")
            return False

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file versioning"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _invalidate_cdn_path(self, path: str):
        """Invalidate CDN cache for specific path"""
        try:
            self.cloudfront_client.create_invalidation(
                DistributionId=self.distribution_id,
                InvalidationBatch={
                    'Paths': {'Quantity': 1, 'Items': [f'/{path}']},
                    'CallerReference': f'{path}_{int(time.time())}'
                }
            )
        except ClientError as e:
            logging.error(f"Failed to invalidate CDN for {path}: {e}")
```

**Acceptance Criteria**:
- [ ] Static assets are served via CDN
- [ ] Asset compression is enabled
- [ ] Cache headers are properly configured
- [ ] Bundle sizes are optimized
- [ ] Asset versioning is implemented

---

### WP 10.4: Monitoring and Alerting (4 hours)

**Objective**: Implement comprehensive monitoring, logging, and alerting

#### Task 10.4.1: Application Performance Monitoring (1.5 hours)
**Files to Create**: `src/monitoring/apm.py`, `middleware/performance_middleware.py`
**Dependencies**: Infrastructure setup, application deployment
**Implementation Details**:
```python
# src/monitoring/apm.py
import time
import psutil
import threading
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None

class APMCollector:
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.active_requests = {}
        self.lock = threading.Lock()
        self.start_time = datetime.now()

    def start_request_trace(self, request_id: str, endpoint: str, user_id: str = None):
        """Start tracing a request"""
        with self.lock:
            self.active_requests[request_id] = {
                'endpoint': endpoint,
                'user_id': user_id,
                'start_time': time.time(),
                'memory_start': psutil.Process().memory_info().rss
            }

    def end_request_trace(self, request_id: str, status_code: int):
        """End tracing a request and collect metrics"""
        with self.lock:
            if request_id not in self.active_requests:
                return

            request_data = self.active_requests.pop(request_id)
            duration = time.time() - request_data['start_time']
            memory_delta = psutil.Process().memory_info().rss - request_data['memory_start']

            # Record metrics
            self.record_metric('request_duration', duration, 'seconds', {
                'endpoint': request_data['endpoint'],
                'status_code': str(status_code),
                'user_id': request_data['user_id']
            })

            self.record_metric('memory_usage', memory_delta, 'bytes', {
                'endpoint': request_data['endpoint']
            })

            # Record error if applicable
            if status_code >= 400:
                self.record_metric('error_count', 1, 'count', {
                    'endpoint': request_data['endpoint'],
                    'status_code': str(status_code)
                })

    def record_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )

        with self.lock:
            self.metrics.append(metric)

            # Keep only last 10000 metrics in memory
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-10000:]

    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.lock:
            recent_metrics = [
                m for m in self.metrics
                if m.timestamp >= cutoff_time
            ]

        summary = {
            'total_requests': 0,
            'avg_response_time': 0,
            'error_rate': 0,
            'active_requests': len(self.active_requests),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

        # Calculate request metrics
        request_metrics = [
            m for m in recent_metrics
            if m.name == 'request_duration'
        ]
        error_metrics = [
            m for m in recent_metrics
            if m.name == 'error_count'
        ]

        if request_metrics:
            summary['total_requests'] = len(request_metrics)
            summary['avg_response_time'] = sum(m.value for m in request_metrics) / len(request_metrics)

        if error_metrics:
            summary['error_rate'] = sum(m.value for m in error_metrics) / max(summary['total_requests'], 1) * 100

        return summary
```

**Acceptance Criteria**:
- [ ] Request tracing is implemented
- [ ] Performance metrics are collected
- [ ] Memory and CPU usage is monitored
- [ ] Error tracking is active
- [ ] Metrics dashboard displays real-time data

#### Task 10.4.2: Health Checks and Diagnostics (1 hour)
**Files to Create**: `src/health/health_checker.py`, `endpoints/health.py`
**Dependencies**: APM implementation
**Implementation Details**:
```python
# src/health/health_checker.py
import asyncio
import aiohttp
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    response_time: float
    message: str
    last_check: float

class HealthChecker:
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.session = aiohttp.ClientSession()

    async def check_database_health(self) -> HealthCheck:
        """Check database connectivity"""
        start_time = time.time()
        try:
            # Query database
            from data_loader import fetch_hn_data
            df = fetch_hn_data(limit=1)

            response_time = time.time() - start_time

            if response_time > 5:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    response_time=response_time,
                    message="Slow database response",
                    last_check=time.time()
                )

            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                message="Database connection OK",
                last_check=time.time()
            )

        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Database error: {str(e)}",
                last_check=time.time()
            )

    async def check_cache_health(self) -> HealthCheck:
        """Check Redis cache connectivity"""
        start_time = time.time()
        try:
            from src.cache.redis_cache import RedisCache
            cache = RedisCache(host='localhost')

            # Test cache operations
            test_key = "health_check_test"
            cache.set(test_key, "test_value", ttl=10)
            result = cache.get(test_key)
            cache.delete(test_key)

            response_time = time.time() - start_time

            if result != "test_value":
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    response_time=response_time,
                    message="Cache read/write test failed",
                    last_check=time.time()
                )

            return HealthCheck(
                name="cache",
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                message="Cache connection OK",
                last_check=time.time()
            )

        except Exception as e:
            return HealthCheck(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Cache error: {str(e)}",
                last_check=time.time()
            )

    async def check_external_apis(self) -> HealthCheck:
        """Check external API availability"""
        start_time = time.time()
        try:
            # Check HackerNews API
            async with self.session.get('https://hacker-news.firebaseio.com/v0/topstories.json') as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        return HealthCheck(
                            name="external_apis",
                            status=HealthStatus.HEALTHY,
                            response_time=time.time() - start_time,
                            message="External APIs accessible",
                            last_check=time.time()
                        )

            return HealthCheck(
                name="external_apis",
                status=HealthStatus.DEGRADED,
                response_time=time.time() - start_time,
                message="External APIs slow or limited",
                last_check=time.time()
            )

        except Exception as e:
            return HealthCheck(
                name="external_apis",
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"External API error: {str(e)}",
                last_check=time.time()
            )

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return overall status"""
        # Run checks concurrently
        checks = await asyncio.gather(
            self.check_database_health(),
            self.check_cache_health(),
            self.check_external_apis(),
            return_exceptions=True
        )

        # Process results
        for check in checks:
            if isinstance(check, Exception):
                self.checks["error"] = HealthCheck(
                    name="error",
                    status=HealthStatus.UNHEALTHY,
                    response_time=0,
                    message=f"Health check error: {str(check)}",
                    last_check=time.time()
                )
            else:
                self.checks[check.name] = check

        # Determine overall status
        statuses = [check.status for check in self.checks.values()]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            'status': overall_status.value,
            'timestamp': time.time(),
            'checks': {
                name: {
                    'status': check.status.value,
                    'response_time': check.response_time,
                    'message': check.message,
                    'last_check': check.last_check
                }
                for name, check in self.checks.items()
            }
        }
```

**Acceptance Criteria**:
- [ ] Health check endpoint is accessible
- [ ] All critical services are monitored
- [ ] Response times are measured
- [ ] Degraded performance is detected
- [ ] Health status is accurately reported

#### Task 10.4.3: Alerting System (1.5 hours)
**Files to Create**: `src/alerting/alert_manager.py`, `config/alert_rules.py`
**Dependencies**: Health checks, APM monitoring
**Implementation Details**:
```python
# src/alerting/alert_manager.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: datetime = None

class AlertManager:
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.logger = logging.getLogger(__name__)

        # Configuration (would come from environment/config)
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.alert_emails = os.getenv('ALERT_EMAILS', '').split(',')

    def check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        # Error rate alert
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 10:  # More than 10% errors
            self.trigger_alert(
                name="high_error_rate",
                severity=AlertSeverity.CRITICAL,
                message=f"High error rate detected: {error_rate:.1f}%"
            )
        elif error_rate > 5:  # More than 5% errors
            self.trigger_alert(
                name="high_error_rate",
                severity=AlertSeverity.WARNING,
                message=f"Elevated error rate: {error_rate:.1f}%"
            )
        else:
            self.resolve_alert("high_error_rate")

        # Response time alert
        avg_response_time = metrics.get('avg_response_time', 0)
        if avg_response_time > 5:  # More than 5 seconds
            self.trigger_alert(
                name="slow_response",
                severity=AlertSeverity.CRITICAL,
                message=f"Very slow response time: {avg_response_time:.1f}s"
            )
        elif avg_response_time > 2:  # More than 2 seconds
            self.trigger_alert(
                name="slow_response",
                severity=AlertSeverity.WARNING,
                message=f"Slow response time: {avg_response_time:.1f}s"
            )
        else:
            self.resolve_alert("slow_response")

        # Memory usage alert
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            self.trigger_alert(
                name="high_memory",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical memory usage: {memory_percent:.1f}%"
            )
        elif memory_percent > 80:
            self.trigger_alert(
                name="high_memory",
                severity=AlertSeverity.WARNING,
                message=f"High memory usage: {memory_percent:.1f}%"
            )
        else:
            self.resolve_alert("high_memory")

    def trigger_alert(self, name: str, severity: AlertSeverity, message: str):
        """Trigger a new alert"""
        timestamp = datetime.now()

        # Check if alert already exists
        if name in self.active_alerts:
            existing = self.active_alerts[name]
            if existing.severity == severity and existing.message == message:
                return  # Same alert already active

        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            timestamp=timestamp
        )

        self.active_alerts[name] = alert
        self.alert_history.append(alert)

        # Send notifications
        self._send_notifications(alert)

        self.logger.warning(f"Alert triggered: {name} - {message}")

    def resolve_alert(self, name: str):
        """Resolve an active alert"""
        if name in self.active_alerts:
            alert = self.active_alerts[name]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            del self.active_alerts[name]

            # Send resolution notification
            self._send_resolution_notification(alert)

            self.logger.info(f"Alert resolved: {name}")

    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        # Send email
        if alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
            self._send_email_alert(alert)

        # Send Slack notification
        if self.slack_webhook:
            self._send_slack_alert(alert)

    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['Subject'] = f"[{alert.severity.value.upper()}] Tech-Pulse Alert: {alert.name}"

            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity.value}
            Message: {alert.message}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            Please investigate this issue immediately.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)

            for email in self.alert_emails:
                if email.strip():
                    msg['To'] = email.strip()
                    server.send_message(msg)

            server.quit()

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def _send_slack_alert(self, alert: Alert):
        """Send Slack notification"""
        try:
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Tech-Pulse Alert: {alert.name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ]
                    }
                ]
            }

            requests.post(self.slack_webhook, json=payload)

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        return [
            {
                'name': alert.name,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
        ]
```

**Acceptance Criteria**:
- [ ] Alert rules are configured
- [ ] Email notifications are sent for critical alerts
- [ ] Slack notifications are working
- [ ] Alert resolution is tracked
- [ ] Alert fatigue is minimized through smart grouping

---

### WP 10.5: Backup and Disaster Recovery (3 hours)

**Objective**: Implement reliable backup and disaster recovery procedures

#### Task 10.5.1: Database Backup Strategy (1 hour)
**Files to Create**: `scripts/backup_database.py`, `scripts/restore_database.py`
**Dependencies**: Database setup
**Implementation Details**:
```python
# scripts/backup_database.py
import subprocess
import boto3
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

class DatabaseBackupManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.backup_bucket = os.getenv('BACKUP_BUCKET_NAME')
        self.retention_days = 30

    def create_backup(self, backup_type: str = 'full') -> Dict[str, Any]:
        """Create database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"tech_pulse_backup_{backup_type}_{timestamp}.sql"

        try:
            if backup_type == 'full':
                # Create full backup
                cmd = [
                    'pg_dump',
                    '-h', os.getenv('DB_HOST'),
                    '-U', os.getenv('DB_USER'),
                    '-d', os.getenv('DB_NAME'),
                    '--no-password',
                    '--verbose',
                    '--clean',
                    '--if-exists',
                    '--format=custom',
                    f'--file={backup_filename}'
                ]
            else:
                # Create incremental backup (using WAL files)
                cmd = [
                    'pg_basebackup',
                    '-h', os.getenv('DB_HOST'),
                    '-U', os.getenv('DB_USER'),
                    '-D', backup_filename,
                    '--format=tar',
                    '--gzip',
                    '--progress'
                ]

            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = os.getenv('DB_PASSWORD')

            # Run backup command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                raise Exception(f"Backup failed: {result.stderr}")

            # Upload to S3
            s3_key = f"database_backups/{backup_filename}"
            self.s3_client.upload_file(
                backup_filename,
                self.backup_bucket,
                s3_key
            )

            # Clean up local file
            os.remove(backup_filename)

            # Record backup metadata
            backup_info = {
                'type': backup_type,
                'filename': backup_filename,
                's3_key': s3_key,
                'size_bytes': os.path.getsize(backup_filename) if os.path.exists(backup_filename) else 0,
                'created_at': timestamp,
                'status': 'completed'
            }

            # Store backup metadata in DynamoDB or S3 metadata
            self._record_backup_metadata(backup_info)

            return backup_info

        except Exception as e:
            logging.error(f"Database backup failed: {e}")
            return {
                'type': backup_type,
                'status': 'failed',
                'error': str(e),
                'created_at': timestamp
            }

    def _record_backup_metadata(self, backup_info: Dict[str, Any]):
        """Record backup metadata"""
        try:
            metadata_key = f"database_backups/metadata/{backup_info['filename']}.json"
            self.s3_client.put_object(
                Bucket=self.backup_bucket,
                Key=metadata_key,
                Body=json.dumps(backup_info),
                ContentType='application/json'
            )
        except Exception as e:
            logging.error(f"Failed to record backup metadata: {e}")

    def restore_backup(self, backup_s3_key: str, target_host: str = None) -> bool:
        """Restore database from backup"""
        try:
            # Download backup from S3
            local_backup = f"/tmp/restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            self.s3_client.download_file(self.backup_bucket, backup_s3_key, local_backup)

            # Restore database
            cmd = [
                'pg_restore',
                '-h', target_host or os.getenv('DB_HOST'),
                '-U', os.getenv('DB_USER'),
                '-d', os.getenv('DB_NAME'),
                '--clean',
                '--if-exists',
                '--verbose',
                local_backup
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = os.getenv('DB_PASSWORD')

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600
            )

            # Clean up local file
            os.remove(local_backup)

            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")

            logging.info(f"Database restored successfully from {backup_s3_key}")
            return True

        except Exception as e:
            logging.error(f"Database restore failed: {e}")
            return False

    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            # List all backup objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.backup_bucket,
                Prefix='database_backups/'
            )

            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.s3_client.delete_object(
                            Bucket=self.backup_bucket,
                            Key=obj['Key']
                        )
                        logging.info(f"Deleted old backup: {obj['Key']}")

        except Exception as e:
            logging.error(f"Failed to cleanup old backups: {e}")
```

**Acceptance Criteria**:
- [ ] Automated daily backups are scheduled
- [ ] Backups are stored in secure cloud storage
- [ ] Backup retention policy is enforced
- [ ] Restore procedure is tested and documented
- [ ] Backup status monitoring is active

#### Task 10.5.2: Application State Backup (1 hour)
**Files to Create**: `scripts/backup_app_state.py`, `config/backup_config.py`
**Dependencies**: Database backup strategy
**Implementation Details**:
```python
# scripts/backup_app_state.py
import json
import tarfile
import tempfile
import boto3
import os
from datetime import datetime
from typing import Dict, Any, List

class AppStateBackupManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.backup_bucket = os.getenv('BACKUP_BUCKET_NAME')

    def backup_application_state(self) -> Dict[str, Any]:
        """Backup complete application state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"app_state_backup_{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Backup configuration files
                self._backup_configurations(temp_dir)

                # Backup cache data
                self._backup_cache_data(temp_dir)

                # Backup user data
                self._backup_user_data(temp_dir)

                # Create archive
                archive_path = os.path.join(temp_dir, f"{backup_name}.tar.gz")
                self._create_archive(temp_dir, archive_path, backup_name)

                # Upload to S3
                s3_key = f"app_state_backups/{backup_name}.tar.gz"
                self.s3_client.upload_file(archive_path, self.backup_bucket, s3_key)

                backup_size = os.path.getsize(archive_path)

                return {
                    'backup_name': backup_name,
                    's3_key': s3_key,
                    'size_bytes': backup_size,
                    'timestamp': timestamp,
                    'status': 'completed'
                }

        except Exception as e:
            logging.error(f"Application state backup failed: {e}")
            return {
                'backup_name': backup_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': timestamp
            }

    def _backup_configurations(self, temp_dir: str):
        """Backup configuration files"""
        config_dir = os.path.join(temp_dir, 'configurations')
        os.makedirs(config_dir)

        # Backup dashboard configuration
        try:
            from dashboard_config import load_config
            config = load_config()
            with open(os.path.join(config_dir, 'dashboard_config.json'), 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Failed to backup dashboard config: {e}")

        # Backup environment variables (excluding sensitive data)
        env_backup = {}
        sensitive_keys = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']

        for key, value in os.environ.items():
            if not any(sensitive in key.upper() for sensitive in sensitive_keys):
                env_backup[key] = value

        with open(os.path.join(config_dir, 'environment.json'), 'w') as f:
            json.dump(env_backup, f, indent=2)

    def _backup_cache_data(self, temp_dir: str):
        """Backup cache data"""
        cache_dir = os.path.join(temp_dir, 'cache')
        os.makedirs(cache_dir)

        try:
            # Export Redis data
            import redis
            r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'))

            # Get all keys
            keys = r.keys('*')

            cache_data = {}
            for key in keys:
                try:
                    value = r.dump(key)
                    cache_data[key.decode()] = value.hex() if value else None
                except:
                    continue

            with open(os.path.join(cache_dir, 'redis_cache.json'), 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logging.warning(f"Failed to backup cache data: {e}")

    def _backup_user_data(self, temp_dir: str):
        """Backup user-generated data"""
        user_data_dir = os.path.join(temp_dir, 'user_data')
        os.makedirs(user_data_dir)

        # Export user preferences
        try:
            from src.database.user_manager import get_all_user_preferences
            user_prefs = get_all_user_preferences()

            with open(os.path.join(user_data_dir, 'user_preferences.json'), 'w') as f:
                json.dump(user_prefs, f, indent=2, default=str)

        except Exception as e:
            logging.warning(f"Failed to backup user preferences: {e}")

        # Backup generated reports
        reports_dir = 'reports'
        if os.path.exists(reports_dir):
            import shutil
            shutil.copytree(reports_dir, os.path.join(user_data_dir, 'reports'))

    def _create_archive(self, temp_dir: str, archive_path: str, backup_name: str):
        """Create compressed archive"""
        with tarfile.open(archive_path, 'w:gz') as tar:
            for item in os.listdir(temp_dir):
                if item.endswith('.json') or os.path.isdir(os.path.join(temp_dir, item)):
                    tar.add(os.path.join(temp_dir, item), arcname=item)
```

**Acceptance Criteria**:
- [ ] Application state is backed up regularly
- [ ] Configuration files are preserved
- [ ] User data is included in backups
- [ ] Archive creation and upload works correctly
- [ ] Backup verification is implemented

#### Task 10.5.3: Disaster Recovery Plan (1 hour)
**Files to Create**: `docs/ disaster_recovery_plan.md`, `scripts/failover.py`
**Dependencies**: Backup strategies
**Implementation Details**:
```python
# scripts/failover.py
import boto3
import subprocess
import logging
import time
from typing import Dict, Any, List

class DisasterRecoveryManager:
    def __init__(self):
        self.ec2_client = boto3.client('ec2')
        self.rds_client = boto3.client('rds')
        self.route53_client = boto3.client('route53')

        # Configuration
        self.primary_region = os.getenv('PRIMARY_REGION', 'us-east-1')
        self.backup_region = os.getenv('BACKUP_REGION', 'us-west-2')
        self.hosted_zone_id = os.getenv('ROUTE53_HOSTED_ZONE_ID')

    def initiate_failover(self, reason: str, auto_approve: bool = False) -> Dict[str, Any]:
        """Initiate disaster recovery failover"""
        failover_id = f"failover_{int(time.time())}"

        logging.info(f"Initiating failover {failover_id}: {reason}")

        steps = [
            self._validate_failover_conditions,
            self._promote_backup_database,
            self._start_backup_instances,
            self._update_dns_records,
            self._verify_services_health
        ]

        results = []

        for step in steps:
            try:
                if not auto_approve:
                    # In production, would require manual approval here
                    pass

                result = step(failover_id)
                results.append(result)

                if not result['success']:
                    logging.error(f"Failover step failed: {result['message']}")
                    break

            except Exception as e:
                logging.error(f"Failover step error: {e}")
                results.append({
                    'step': step.__name__,
                    'success': False,
                    'error': str(e)
                })
                break

        success = all(r['success'] for r in results)

        if success:
            logging.info(f"Failover {failover_id} completed successfully")
        else:
            logging.error(f"Failover {failover_id} failed")

        return {
            'failover_id': failover_id,
            'success': success,
            'reason': reason,
            'steps': results,
            'timestamp': datetime.now().isoformat()
        }

    def _validate_failover_conditions(self, failover_id: str) -> Dict[str, Any]:
        """Validate that failover conditions are met"""
        try:
            # Check primary region health
            primary_health = self._check_region_health(self.primary_region)

            if primary_health['healthy']:
                return {
                    'step': 'validate_conditions',
                    'success': False,
                    'message': 'Primary region is healthy - failover not required'
                }

            # Check backup region availability
            backup_health = self._check_region_health(self.backup_region)

            if not backup_health['available']:
                return {
                    'step': 'validate_conditions',
                    'success': False,
                    'message': 'Backup region not available for failover'
                }

            return {
                'step': 'validate_conditions',
                'success': True,
                'message': 'Failover conditions validated'
            }

        except Exception as e:
            return {
                'step': 'validate_conditions',
                'success': False,
                'error': str(e)
            }

    def _check_region_health(self, region: str) -> Dict[str, Any]:
        """Check health of a specific region"""
        # Implementation would check:
        # - Database connectivity
        # - Application instances
        # - Network connectivity
        # - External dependencies

        return {
            'healthy': False,  # Would be determined by actual checks
            'available': True,
            'checks': {
                'database': 'unhealthy',
                'instances': 'unavailable',
                'network': 'healthy'
            }
        }

    def _promote_backup_database(self, failover_id: str) -> Dict[str, Any]:
        """Promote backup database to primary"""
        try:
            # Promote read replica to primary
            response = self.rds_client.promote_read_replica(
                DBInstanceIdentifier=f'tech-pulse-backup-{self.backup_region}',
                BackupRetentionPeriod=7,
                PreferredBackupWindow='03:00-04:00'
            )

            # Wait for promotion to complete
            waiter = self.rds_client.get_waiter('db_instance_available')
            waiter.wait(
                DBInstanceIdentifier=f'tech-pulse-backup-{self.backup_region}',
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )

            return {
                'step': 'promote_database',
                'success': True,
                'message': 'Database promoted successfully'
            }

        except Exception as e:
            return {
                'step': 'promote_database',
                'success': False,
                'error': str(e)
            }

    def _start_backup_instances(self, failover_id: str) -> Dict[str, Any]:
        """Start application instances in backup region"""
        try:
            # Get backup instance templates
            response = self.ec2_client.describe_launch_templates(
                Filters=[
                    {'Name': 'tag:Environment', 'Values': ['backup']},
                    {'Name': 'tag:Application', 'Values': ['tech-pulse']}
                ]
            )

            for template in response['LaunchTemplates']:
                # Launch instances from template
                self.ec2_client.run_instances(
                    LaunchTemplate={'LaunchTemplateId': template['LaunchTemplateId']},
                    MinCount=1,
                    MaxCount=1,
                    TagSpecifications=[
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {'Key': 'Name', 'Value': f'tech-pulse-{failover_id}'},
                                {'Key': 'Failover', 'Value': failover_id}
                            ]
                        }
                    ]
                )

            return {
                'step': 'start_instances',
                'success': True,
                'message': 'Backup instances started'
            }

        except Exception as e:
            return {
                'step': 'start_instances',
                'success': False,
                'error': str(e)
            }

    def _update_dns_records(self, failover_id: str) -> Dict[str, Any]:
        """Update DNS to point to backup region"""
        try:
            # Get backup instance IP
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:Failover', 'Values': [failover_id]},
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )

            if not response['Reservations']:
                return {
                    'step': 'update_dns',
                    'success': False,
                    'message': 'No running instances found for DNS update'
                }

            backup_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']

            # Update Route53 record
            self.route53_client.change_resource_record_sets(
                HostedZoneId=self.hosted_zone_id,
                ChangeBatch={
                    'Changes': [
                        {
                            'Action': 'UPSERT',
                            'ResourceRecordSet': {
                                'Name': 'tech-pulse.example.com',
                                'Type': 'A',
                                'TTL': 60,
                                'ResourceRecords': [{'Value': backup_ip}]
                            }
                        }
                    ]
                }
            )

            return {
                'step': 'update_dns',
                'success': True,
                'message': f'DNS updated to point to {backup_ip}'
            }

        except Exception as e:
            return {
                'step': 'update_dns',
                'success': False,
                'error': str(e)
            }

    def _verify_services_health(self, failover_id: str) -> Dict[str, Any]:
        """Verify all services are healthy after failover"""
        try:
            # Wait for DNS propagation
            time.sleep(60)

            # Check application health
            import requests
            response = requests.get('https://tech-pulse.example.com/health', timeout=30)

            if response.status_code == 200:
                return {
                    'step': 'verify_health',
                    'success': True,
                    'message': 'All services healthy after failover'
                }
            else:
                return {
                    'step': 'verify_health',
                    'success': False,
                    'message': f'Health check failed with status {response.status_code}'
                }

        except Exception as e:
            return {
                'step': 'verify_health',
                'success': False,
                'error': str(e)
            }
```

**Acceptance Criteria**:
- [ ] Disaster recovery plan is documented
- [ ] Automated failover procedures are implemented
- [ ] RTO (Recovery Time Objective) is under 30 minutes
- [ ] RPO (Recovery Point Objective) is under 5 minutes
- [ ] Failover testing is conducted regularly

---

### WP 10.6: Documentation and Training (2 hours)

**Objective**: Create comprehensive documentation and training materials

#### Task 10.6.1: Technical Documentation (1 hour)
**Files to Create**: `docs/DEPLOYMENT_GUIDE.md`, `docs/TROUBLESHOOTING.md`, `docs/ARCHITECTURE.md`
**Dependencies**: All previous work packages
**Implementation Details**:

Write comprehensive technical documentation covering:
- Architecture overview and system design
- Deployment procedures and environment setup
- Monitoring and maintenance procedures
- Troubleshooting common issues
- Security guidelines and best practices
- Performance optimization techniques
- Backup and recovery procedures

#### Task 10.6.2: User Documentation (1 hour)
**Files to Create**: `docs/USER_GUIDE.md`, `docs/API_DOCUMENTATION.md`, `docs/FAQ.md`
**Dependencies**: Application features
**Implementation Details**:

Create user-friendly documentation:
- Getting started guide
- Feature explanations and tutorials
- API reference and examples
- Frequently asked questions
- Best practices and tips

---

## Testing Strategy

### Unit Testing
- All new modules must have unit tests with >90% coverage
- Mock external dependencies for reliable testing
- Test error conditions and edge cases

### Integration Testing
- Test CI/CD pipeline with sample commits
- Verify deployment process works end-to-end
- Test monitoring and alerting systems

### Performance Testing
- Load test application with simulated traffic
- Verify auto-scaling behavior under load
- Test database performance with large datasets

### Security Testing
- Run vulnerability scanners on the application
- Test SSL/TLS configuration
- Verify security headers and protections

### Disaster Recovery Testing
- Test backup and restore procedures
- Simulate failover scenarios
- Verify recovery time objectives

## Quality Gates

### Code Quality
- All tests must pass (100% success rate)
- Code coverage must be >90%
- No critical security vulnerabilities
- Code must pass linting and formatting checks

### Performance Standards
- API response time <2 seconds (95th percentile)
- Page load time <3 seconds
- Error rate <1%
- System uptime >99.9%

### Security Requirements
- All data must be encrypted at rest and in transit
- Security patches must be applied within 7 days
- Access controls must follow least privilege principle
- Security audit must pass quarterly

## Deliverables

### Infrastructure
- CloudFormation/Terraform templates for all infrastructure
- Docker images and container orchestration configurations
- CI/CD pipeline definitions
- Monitoring and alerting setup

### Application
- Production-ready Streamlit application
- All source code with comprehensive tests
- Configuration management system
- Performance optimizations

### Documentation
- Complete technical documentation
- User guides and tutorials
- API documentation
- Troubleshooting procedures

### Operations
- Backup and disaster recovery procedures
- Monitoring and alerting configurations
- Security hardening guide
- Maintenance playbooks

## Timeline

This phase is estimated to take **33 hours** total:

- WP 10.1: CI/CD Pipeline Implementation (8 hours)
- WP 10.2: Production Infrastructure Setup (10 hours)
- WP 10.3: Performance Optimization (6 hours)
- WP 10.4: Monitoring and Alerting (4 hours)
- WP 10.5: Backup and Disaster Recovery (3 hours)
- WP 10.6: Documentation and Training (2 hours)

## Success Criteria

### Functional Requirements
- [ ] Application deployed and accessible in production
- [ ] CI/CD pipeline automates testing and deployment
- [ ] Monitoring and alerting systems are active
- [ ] Backup and recovery procedures are tested

### Non-Functional Requirements
- [ ] 99.9% uptime achieved
- [ ] Page load times under 3 seconds
- [ ] Security audit passes with zero critical findings
- [ ] Disaster recovery time under 30 minutes

### Operational Readiness
- [ ] Documentation is complete and accessible
- [ ] Team is trained on operational procedures
- [ ] Monitoring dashboards provide comprehensive visibility
- [ ] Support procedures are documented and tested

## Risk Mitigation

### Technical Risks
- **Deployment Failures**: Implement blue-green deployment for zero downtime
- **Performance Issues**: Conduct thorough performance testing before production
- **Security Vulnerabilities**: Regular security scanning and prompt patching
- **Data Loss**: Multiple backup strategies with regular restore testing

### Operational Risks
- **Configuration Drift**: Use infrastructure as code for consistency
- **Human Error**: Implement peer reviews and automated checks
- **Vendor Lock-in**: Design with multi-cloud compatibility
- **Cost Overruns**: Implement cost monitoring and alerts

### Business Risks
- **Downtime**: Implement high availability and failover mechanisms
- **Data Breaches**: Encrypt sensitive data and implement access controls
- **Compliance Violations**: Regular security audits and compliance checks
- **Reputation Damage**: Proactive monitoring and rapid incident response