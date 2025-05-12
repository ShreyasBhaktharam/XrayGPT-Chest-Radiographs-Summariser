#!/bin/bash
# deploy.sh - Script to manage promotion between environments for XrayGPT

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
VERSION=${VERSION:-$(git rev-parse --short HEAD)}
TARGET_ENV=${1:-staging}
PROMOTE_FROM=${2}
SKIP_TESTS=${3:-false}

# Check if environment is valid
if [[ ! "$TARGET_ENV" =~ ^(staging|canary|production)$ ]]; then
    echo -e "${RED}Error: Invalid environment '$TARGET_ENV'. Must be staging, canary, or production.${NC}"
    exit 1
fi

# Define port mappings for environments
declare -A ENV_PORTS
ENV_PORTS[staging]=8001
ENV_PORTS[canary]=8002
ENV_PORTS[production]=8000

# Define promotion paths
declare -A PROMOTION_PATHS
PROMOTION_PATHS[staging]="canary"
PROMOTION_PATHS[canary]="production"

echo -e "${GREEN}Deploying XrayGPT version ${VERSION} to ${TARGET_ENV}${NC}"

# Function to wait for service health
wait_for_health() {
    local service=$1
    local port=$2
    local max_retries=10
    local retry_count=0
    
    echo -e "${YELLOW}Waiting for $service to become healthy...${NC}"
    
    while [ $retry_count -lt $max_retries ]; do
        health_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
        
        if [ "$health_status" == "200" ]; then
            echo -e "${GREEN}$service is healthy!${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo -e "${YELLOW}Attempt $retry_count/$max_retries - $service not yet healthy, retrying in 10s...${NC}"
        sleep 10
    done
    
    echo -e "${RED}Error: $service failed to become healthy after $max_retries attempts${NC}"
    return 1
}

# Function to run load tests
run_load_test() {
    local env=$1
    local port=${ENV_PORTS[$env]}
    
    if [ "$SKIP_TESTS" == "true" ]; then
        echo -e "${YELLOW}Skipping load tests for $env${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Running load tests for $env on port $port...${NC}"
    
    # Check if k6 is installed
    if ! command -v k6 &> /dev/null; then
        echo -e "${YELLOW}k6 not found, installing...${NC}"
        sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install -y k6
    fi
    
    # Run k6 load test
    k6 run tests/load-test.js -e URL=http://localhost:$port
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo -e "${RED}Load tests failed for $env${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Load tests passed for $env${NC}"
    return 0
}

# Function to monitor canary deployment
monitor_canary() {
    local duration=${1:-600} # Default 10 minutes
    
    if [ "$SKIP_TESTS" == "true" ]; then
        echo -e "${YELLOW}Skipping canary monitoring${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Monitoring canary deployment for ${duration}s...${NC}"
    
    # Check if Python and required packages are installed
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is required but not installed${NC}"
        return 1
    fi
    
    pip3 install --quiet prometheus-api-client requests
    
    # Run monitoring script
    python3 tests/monitor_canary.py --host localhost --port 9092 --duration $duration
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo -e "${RED}Canary monitoring detected issues${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Canary monitoring passed${NC}"
    return 0
}

# Function to deploy to an environment
deploy_to_env() {
    local env=$1
    local port=${ENV_PORTS[$env]}
    
    echo -e "${GREEN}Starting deployment to $env environment...${NC}"
    
    # Use Docker Compose to deploy
    echo -e "${YELLOW}Deploying services for $env...${NC}"
    VERSION=$VERSION docker-compose -f docker-compose.$env.yml up -d
    
    # Wait for service to become healthy
    wait_for_health "$env" $port
    
    # Additional steps based on environment
    case $env in
        staging)
            echo -e "${YELLOW}Running load tests in staging environment...${NC}"
            run_load_test $env
            ;;
        canary)
            echo -e "${YELLOW}Starting canary monitoring...${NC}"
            monitor_canary 300 # Monitor for 5 minutes
            ;;
        production)
            echo -e "${YELLOW}Verifying production deployment...${NC}"
            run_load_test $env
            ;;
    esac
    
    echo -e "${GREEN}Deployment to $env completed successfully!${NC}"
}

# Function to handle promotion between environments
promote() {
    local from_env=$1
    local to_env=$2
    
    echo -e "${YELLOW}Promoting from $from_env to $to_env...${NC}"
    
    # Verify source environment is healthy
    wait_for_health $from_env ${ENV_PORTS[$from_env]}
    
    # Deploy to target environment
    deploy_to_env $to_env
    
    echo -e "${GREEN}Successfully promoted from $from_env to $to_env!${NC}"
}

# Main deployment logic
case $TARGET_ENV in
    staging)
        deploy_to_env staging
        ;;
    canary)
        if [ -z "$PROMOTE_FROM" ]; then
            # Direct deployment to canary
            deploy_to_env canary
        else
            # Promotion from staging to canary
            promote staging canary
        fi
        ;;
    production)
        if [ -z "$PROMOTE_FROM" ]; then
            echo -e "${YELLOW}Warning: Direct deployment to production is not recommended.${NC}"
            echo -e "${YELLOW}Continuing in 10 seconds. Press Ctrl+C to abort...${NC}"
            sleep 10
            deploy_to_env production
        else
            # Promotion to production
            promote $PROMOTE_FROM production
        fi
        ;;
esac

echo -e "${GREEN}Deployment process completed successfully!${NC}"
exit 0