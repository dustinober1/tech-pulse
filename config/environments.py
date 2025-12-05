"""
Environment Configuration Management
Handles environment-specific settings and configurations
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    name: str = "tech_pulse"
    user: str = "tech_pulse_user"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = ""
    encryption_key: str = ""
    session_timeout: int = 3600
    password_min_length: int = 8
    enable_csrf: bool = True
    allowed_origins: list = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings"""
    enabled: bool = False
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_file: str = "logs/tech_pulse.log"
    enable_apm: bool = False
    health_check_interval: int = 30


class Environment:
    """
    Environment configuration manager
    Handles different environments (development, staging, production)
    """

    def __init__(self):
        self.env = os.getenv('ENVIRONMENT', 'development').lower()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration based on environment"""
        base_config = {
            'debug': False,
            'log_level': 'INFO',
            'cache_ttl': 3600,
            'max_workers': 4,
            'request_timeout': 30,
            'enable_monitoring': False,
            'feature_flags': {
                'enable_phase9_analytics': True,
                'enable_realtime_features': True,
                'enable_pdf_generation': True,
                'enable_advanced_predictions': True
            }
        }

        # Environment-specific configurations
        if self.env == 'development':
            base_config.update({
                'debug': True,
                'log_level': 'DEBUG',
                'cache_ttl': 300,  # 5 minutes for development
                'max_workers': 2,
                'enable_monitoring': True,
                'mock_external_apis': True
            })
        elif self.env == 'staging':
            base_config.update({
                'debug': True,
                'log_level': 'INFO',
                'cache_ttl': 1800,  # 30 minutes
                'max_workers': 4,
                'enable_monitoring': True,
                'mock_external_apis': False
            })
        elif self.env == 'production':
            base_config.update({
                'debug': False,
                'log_level': 'WARNING',
                'cache_ttl': 7200,  # 2 hours
                'max_workers': 8,
                'enable_monitoring': True,
                'mock_external_apis': False
            })

        return base_config

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            name=os.getenv('DB_NAME', 'tech_pulse'),
            user=os.getenv('DB_USER', 'tech_pulse_user'),
            password=os.getenv('DB_PASSWORD', ''),
            ssl_mode=os.getenv('DB_SSL_MODE', 'prefer'),
            pool_size=int(os.getenv('DB_POOL_SIZE', 5)),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 10))
        )

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', 0)),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', 5)),
            socket_connect_timeout=int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', 5)),
            retry_on_timeout=os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true'
        )

    @property
    def security(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            secret_key=os.getenv('SECRET_KEY', self._generate_secret_key()),
            encryption_key=os.getenv('ENCRYPTION_KEY', self._generate_encryption_key()),
            session_timeout=int(os.getenv('SESSION_TIMEOUT', 3600)),
            password_min_length=int(os.getenv('PASSWORD_MIN_LENGTH', 8)),
            enable_csrf=os.getenv('ENABLE_CSRF', 'true').lower() == 'true',
            allowed_origins=os.getenv('ALLOWED_ORIGINS', '*').split(',')
        )

    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            enabled=self.config['enable_monitoring'],
            metrics_port=int(os.getenv('METRICS_PORT', 9090)),
            log_level=os.getenv('LOG_LEVEL', self.config['log_level']),
            log_file=os.getenv('LOG_FILE', 'logs/tech_pulse.log'),
            enable_apm=os.getenv('ENABLE_APM', 'false').lower() == 'true',
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', 30))
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.env == 'development'

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env == 'production'

    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.env == 'staging'

    def get_database_url(self) -> str:
        """Get database connection URL"""
        db = self.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        redis = self.redis
        auth_part = f":{redis.password}@" if redis.password else ""
        return f"redis://{auth_part}{redis.host}:{redis.port}/{redis.db}"

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration completeness"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check required environment variables
        required_vars = ['SECRET_KEY']
        if self.is_production():
            required_vars.extend(['DB_PASSWORD', 'ENCRYPTION_KEY'])

        for var in required_vars:
            if not os.getenv(var):
                validation_results['errors'].append(f"Missing required environment variable: {var}")
                validation_results['valid'] = False

        # Check for insecure settings in production
        if self.is_production():
            if self.config.get('debug', False):
                validation_results['warnings'].append("Debug mode is enabled in production")

            if os.getenv('SECRET_KEY') == 'your_secret_key_change_me_in_production':
                validation_results['errors'].append("Default secret key being used in production")
                validation_results['valid'] = False

        return validation_results

    def _generate_secret_key(self) -> str:
        """Generate a secret key for development"""
        import secrets
        return secrets.token_urlsafe(32)

    def _generate_encryption_key(self) -> str:
        """Generate an encryption key for development"""
        import secrets
        return secrets.token_urlsafe(32)

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag status"""
        feature_flags = self.config.get('feature_flags', {})
        env_flag = os.getenv(flag_name.upper(), '').lower() == 'true'
        return env_flag or feature_flags.get(flag_name, default)

    def get_external_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get external API configuration"""
        api_configs = {
            'alpha_vantage': {
                'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
                'base_url': 'https://www.alphavantage.co/query',
                'timeout': int(os.getenv('API_TIMEOUT', 30))
            },
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': 'https://api.openai.com/v1',
                'timeout': int(os.getenv('API_TIMEOUT', 30))
            }
        }

        return api_configs.get(api_name, {})

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return {
            'type': os.getenv('CACHE_TYPE', 'redis'),
            'ttl': int(os.getenv('CACHE_TTL', self.config['cache_ttl'])),
            'key_prefix': os.getenv('CACHE_KEY_PREFIX', 'tech_pulse'),
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'password': self.redis.password,
                'db': self.redis.db
            }
        }

    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            'username': os.getenv('EMAIL_USER'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('FROM_EMAIL', os.getenv('EMAIL_USER')),
            'alert_emails': os.getenv('ALERT_EMAILS', '').split(',')
        }

    def get_slack_config(self) -> Dict[str, Any]:
        """Get Slack integration configuration"""
        return {
            'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
            'channel': os.getenv('SLACK_CHANNEL', '#alerts'),
            'username': os.getenv('SLACK_USERNAME', 'Tech-Pulse Bot')
        }

    def __str__(self) -> str:
        """String representation of environment"""
        return f"Environment(env={self.env}, debug={self.config.get('debug', False)})"