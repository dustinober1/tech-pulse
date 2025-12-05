#!/bin/bash

# Tech-Pulse Deployment Script
# Automated deployment for production and staging environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Default values
ENVIRONMENT=${1:-staging}
BACKUP_BEFORE_DEPLOY=${2:-true}
SKIP_TESTS=${3:-false}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
IMAGE_TAG=${IMAGE_TAG:-latest}

log "Starting Tech-Pulse deployment to $ENVIRONMENT environment"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    error "Invalid environment. Must be 'staging' or 'production'"
fi

# Check required tools
command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is required but not installed"

# Load environment variables
if [[ -f ".env.$ENVIRONMENT" ]]; then
    log "Loading environment variables from .env.$ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
elif [[ -f ".env" ]]; then
    log "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    warning "No environment file found, using default settings"
fi

# Set environment-specific variables
export ENVIRONMENT=$ENVIRONMENT
export IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

log "Using image tag: $IMAGE_TAG"

# Function to run pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks"

    # Check if we're on the right branch
    if [[ "$ENVIRONMENT" == "production" ]]; then
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
        if [[ "$CURRENT_BRANCH" != "main" ]]; then
            error "Production deployments must be from the main branch. Current branch: $CURRENT_BRANCH"
        fi
    fi

    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        error "Working directory is not clean. Please commit or stash changes before deploying."
    fi

    # Validate environment configuration
    if [[ -f "config/environments.py" ]]; then
        python3 -c "
import sys
sys.path.append('.')
from config.environments import Environment
env = Environment()
validation = env.validate_config()
if not validation['valid']:
    print('Configuration validation failed:')
    for error in validation['errors']:
        print(f'  ERROR: {error}')
    exit(1)
for warning in validation['warnings']:
    print(f'  WARNING: {warning}')
print('Configuration validation passed')
" || error "Environment configuration validation failed"
    fi

    success "Pre-deployment checks passed"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warning "Skipping tests as requested"
        return
    fi

    log "Running test suite"

    # Run unit tests
    python -m pytest test/ -v --tb=short --cov=. --cov-report=term-missing || error "Tests failed"

    success "All tests passed"
}

# Function to backup current deployment
backup_current_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log "Skipping backup as requested"
        return
    fi

    log "Creating backup of current deployment"

    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)_$ENVIRONMENT"
    mkdir -p "$BACKUP_DIR"

    # Backup configuration
    cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true

    # Backup database if running
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up database"
        docker-compose exec -T postgres pg_dump -U tech_pulse_user tech_pulse > "$BACKUP_DIR/database.sql" || warning "Database backup failed"
    fi

    # Save current Docker images
    docker images tech-pulse --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}" > "$BACKUP_DIR/docker_images.txt" 2>/dev/null || true

    success "Backup created at $BACKUP_DIR"
}

# Function to build and push Docker image
build_and_push_image() {
    log "Building Docker image"

    IMAGE_NAME="tech-pulse"
    FULL_IMAGE_TAG="$IMAGE_NAME:$IMAGE_TAG"

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        FULL_IMAGE_TAG="$DOCKER_REGISTRY/$FULL_IMAGE_TAG"
    fi

    # Build the image
    docker build -t "$FULL_IMAGE_TAG" . || error "Docker build failed"

    # Push to registry if registry is specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log "Pushing image to registry: $DOCKER_REGISTRY"
        docker push "$FULL_IMAGE_TAG" || error "Docker push failed"
    fi

    success "Docker image built and tagged as $FULL_IMAGE_TAG"
}

# Function to deploy application
deploy_application() {
    log "Deploying application to $ENVIRONMENT"

    # Update docker-compose.yml with the new image tag
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        FULL_IMAGE_TAG="$DOCKER_REGISTRY/tech-pulse:$IMAGE_TAG"
    else
        FULL_IMAGE_TAG="tech-pulse:$IMAGE_TAG"
    fi

    # Create environment-specific docker-compose override
    cat > "docker-compose.$ENVIRONMENT.override.yml" <<EOF
version: '3.8'
services:
  tech-pulse-dashboard:
    image: $FULL_IMAGE_TAG
    environment:
      - ENVIRONMENT=$ENVIRONMENT
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
EOF

    # Stop existing services
    log "Stopping existing services"
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.override.yml" down || true

    # Pull latest images
    log "Pulling latest images"
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.override.yml" pull || true

    # Start services
    log "Starting services"
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.override.yml" up -d || error "Failed to start services"

    # Wait for services to be healthy
    log "Waiting for services to be healthy"
    sleep 10

    # Check service health
    for i in {1..30}; do
        if docker-compose ps | grep -q "Up (healthy)"; then
            success "All services are healthy"
            break
        elif [[ $i -eq 30 ]]; then
            error "Services did not become healthy within expected time"
        fi
        log "Waiting for services... ($i/30)"
        sleep 5
    done
}

# Function to run post-deployment verification
post_deployment_verification() {
    log "Running post-deployment verification"

    # Check if the application is responding
    APP_URL="http://localhost:8501"

    for i in {1..10}; do
        if curl -f "$APP_URL/_stcore/health" >/dev/null 2>&1; then
            success "Application is responding at $APP_URL"
            break
        elif [[ $i -eq 10 ]]; then
            error "Application is not responding at $APP_URL"
        fi
        log "Checking application health... ($i/10)"
        sleep 10
    done

    # Run smoke tests
    log "Running smoke tests"
    python -c "
import requests
import sys

try:
    response = requests.get('http://localhost:8501', timeout=30)
    if response.status_code == 200:
        print('Application smoke test passed')
    else:
        print(f'Application returned status code: {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'Smoke test failed: {e}')
    sys.exit(1)
" || error "Smoke tests failed"

    success "Post-deployment verification completed"
}

# Function to cleanup old images
cleanup_old_images() {
    log "Cleaning up old Docker images"

    # Remove old versions of our image (keep last 3)
    docker images tech-pulse --format "table {{.Repository}}:{{.Tag}}" | \
    tail -n +2 | \
    tail -n +4 | \
    awk '{print $1":"$2}' | \
    xargs -r docker rmi -f 2>/dev/null || true

    # Remove dangling images
    docker image prune -f || true

    success "Docker image cleanup completed"
}

# Function to send deployment notification
send_notification() {
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        log "Sending deployment notification to Slack"

        MESSAGE="✅ Tech-Pulse deployment to $ENVIRONMENT completed successfully\n"
        MESSAGE+="Image tag: $IMAGE_TAG\n"
        MESSAGE+="Deployed by: $(git config user.name) <$(git config user.email)>"

        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$MESSAGE\"}" \
            "$SLACK_WEBHOOK_URL" || warning "Failed to send Slack notification"
    fi
}

# Main deployment flow
main() {
    log "Starting deployment process for Tech-Pulse"

    pre_deployment_checks
    run_tests
    backup_current_deployment
    build_and_push_image
    deploy_application
    post_deployment_verification
    cleanup_old_images
    send_notification

    success "Deployment completed successfully! 🎉"
    log "Tech-Pulse is now running in $ENVIRONMENT environment"
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main