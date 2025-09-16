#!/bin/bash

# setup_prerequisites.sh - Automated prerequisites setup for MV3DT
# This script automates all the manual setup steps described in README.md

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Global variables
export DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:8.0-triton-multiarch}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR=${BASE_DIR:-$HOME}
USE_INFERENCE_BUILDER=${USE_INFERENCE_BUILDER:-false}
KAFKA_VERSION="4.0.0"
SCALA_VERSION="2.13"

# Standardized paths
KAFKA_DIR="$BASE_DIR/kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
INFERENCE_BUILDER_DIR="$BASE_DIR/inference_builder"

# Utility functions
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_os() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "This script only supports Linux. Detected OS: $OSTYPE"
        exit 1
    fi
    
    if ! command_exists lsb_release; then
        log_warning "lsb_release not found, checking if lsb-release package is installed"
        if ! dpkg-query -W -f='${Status}' lsb-release 2>/dev/null | grep -q "install ok installed"; then
            log_info "Installing lsb-release package..."
            sudo apt update && sudo apt install -y lsb-release || true
        else
            log_info "lsb-release package is installed but command not found, may need PATH update"
        fi
    fi
    
    if command_exists lsb_release; then
        local distro=$(lsb_release -si)
        local version=$(lsb_release -sr)
        log_info "Detected OS: $distro $version"
        
        if [[ "$distro" != "Ubuntu" ]]; then
            log_warning "This script is optimized for Ubuntu 24.04, but will attempt to continue on $distro"
        fi
    fi
}

check_gpu() {
    log_info "Checking NVIDIA GPU availability..."
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
            local major_version=$(echo "$driver_version" | cut -d'.' -f1)
            
            if [[ $major_version -ge 570 ]]; then
                log_success "NVIDIA GPU detected with driver version: $driver_version"
                return 0
            else
                log_error "NVIDIA driver version $driver_version is too old (need 570+ series)"
                return 1
            fi
        else
            log_error "nvidia-smi command failed to execute"
        fi
    else
        log_error "nvidia-smi not found"
    fi
    
    log_error "NVIDIA GPU or drivers not properly installed"
    log_info "Please install NVIDIA drivers (version 570.86.15 or higher) and try again"
    log_info "Visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
    return 1
}


setup_deepstream_container() {    
    # Check if DeepStream container is already available
    if docker images | grep -q $DEEPSTREAM_IMAGE; then
        log_success "DeepStream container already available"
    else
        log_info "Pulling DeepStream container image..."
        if docker pull $DEEPSTREAM_IMAGE; then
            log_success "DeepStream container pulled successfully"
        else
            log_error "Failed to pull DeepStream container"
            log_info "Please check your network connection and Docker credentials"
            return 1
        fi
    fi
}

setup_git_lfs() {
    log_info "Setting up Git LFS and extracting datasets and models..."
    
    cd "$REPO_ROOT"
    
    # Check if datasets are already extracted
    if [[ -d "datasets/mtmc_4cam" && -d "datasets/mtmc_12cam" ]]; then
        log_success "Datasets already extracted"
    else
        # Extract assets if they exist
        if [[ -f "assets/datasets.zip" ]]; then
            log_info "Extracting datasets..."
            unzip -q assets/datasets.zip
            log_success "Datasets extracted"
        else
            log_error "assets/datasets.zip not found"
            return 1
        fi
    fi
    
    if [[ -f "models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx" && -f "models/BodyPose3DNet/bodypose3dnet_accuracy.onnx" ]]; then
        log_success "Models already extracted"
    else
        # Download models from NGC
        log_info "Downloading PeopleNet Transformer model..."
        if ! wget -q --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx' -O 'models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx'; then
            log_error "Failed to download PeopleNet Transformer model"
            return 1
        fi

        log_info "Downloading BodyPose3DNet model..."
        mkdir -p models/BodyPose3DNet
        if ! wget -q --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/bodypose3dnet/deployable_accuracy_onnx_1.0/files?redirect=true&path=bodypose3dnet_accuracy.onnx' -O 'models/BodyPose3DNet/bodypose3dnet_accuracy.onnx'; then
            log_error "Failed to download BodyPose3DNet model"
            return 1
        fi
    fi
}

setup_mosquitto() {
    log_info "Setting up Mosquitto MQTT broker..."
    
    if ! command_exists mosquitto; then
        log_info "Checking Mosquitto packages..."
        local mosquitto_packages=("mosquitto" "mosquitto-clients")
        local packages_to_install=()
        
        for package in "${mosquitto_packages[@]}"; do
            if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
                packages_to_install+=("$package")
            else
                log_info "$package is already installed"
            fi
        done
        
        if [[ ${#packages_to_install[@]} -gt 0 ]]; then
            log_info "Installing missing Mosquitto packages: ${packages_to_install[*]}"
            sudo apt-add-repository -y ppa:mosquitto-dev/mosquitto-ppa
            sudo apt update
            sudo apt install -y "${packages_to_install[@]}"
        else
            log_success "All Mosquitto packages are already installed"
        fi
    fi
    
    # Start mosquitto service
    if ! systemctl is-active --quiet mosquitto; then
        log_info "Starting Mosquitto service..."
        sudo systemctl enable mosquitto
        sudo systemctl start mosquitto
    fi
    
    # Test MQTT broker
    sleep 2
    cd "$REPO_ROOT"
    if [[ -x "scripts/mosquitto_test.sh" ]]; then
        if ./scripts/mosquitto_test.sh >/dev/null 2>&1; then
            log_success "Mosquitto broker is running and accessible"
        else
            log_warning "Mosquitto service test failed, trying manual start..."
            # Try starting manually
            mosquitto -p 1883 -d
            sleep 2
            if ./scripts/mosquitto_test.sh >/dev/null 2>&1; then
                log_success "Mosquitto broker started manually"
            else
                log_error "Failed to start Mosquitto broker"
                return 1
            fi
        fi
    else
        log_warning "Mosquitto test script not found, assuming broker is working"
    fi
}

setup_java() {
    log_info "Setting up Java for Kafka..."
    
    if command_exists java; then
        local java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [[ "$java_version" -ge 17 ]]; then
            log_success "Java $java_version already installed"
            return 0
        fi
    fi
    
    log_info "Checking OpenJDK 17 package..."
    if ! dpkg-query -W -f='${Status}' openjdk-17-jdk 2>/dev/null | grep -q "install ok installed"; then
        log_info "Installing OpenJDK 17..."
        sudo apt update && sudo apt install -y openjdk-17-jdk
    else
        log_success "OpenJDK 17 package is already installed"
    fi
    
    # Verify installation
    if java -version >/dev/null 2>&1; then
        log_success "Java installed successfully"
    else
        log_error "Java installation failed"
        return 1
    fi
}

setup_kafka() {
    log_info "Setting up Kafka..."
    
    # Check if Kafka is already running
    if netstat -ln 2>/dev/null | grep -q ":9092 "; then
        log_info "Kafka appears to be running on port 9092, checking topic..."
        if check_kafka_topic; then
            log_success "Kafka is already running with mv3dt topic"
            return 0
        fi
    fi
    
    setup_java
    
    # Download and extract Kafka if not exists
    if [[ ! -d "$KAFKA_DIR" ]]; then
        log_info "Downloading Kafka ${KAFKA_VERSION}..."
        cd "$BASE_DIR"
        
        local kafka_url="https://dlcdn.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        
        if ! wget -q "$kafka_url" -O "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"; then
            log_error "Failed to download Kafka from $kafka_url"
            return 1
        fi
        
        log_info "Extracting Kafka..."
        tar -xzf "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        rm "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        log_success "Kafka extracted to $KAFKA_DIR"
    fi
    
    cd "$KAFKA_DIR"
    
    # Check for existing cluster ID or create new one
    local meta_properties="/tmp/kraft-combined-logs/meta.properties"
    if [[ -f "$meta_properties" ]]; then
        # Extract existing cluster ID from meta.properties
        export KAFKA_CLUSTER_ID=$(grep "cluster.id=" "$meta_properties" | cut -d'=' -f2)
        if [[ -n "$KAFKA_CLUSTER_ID" ]]; then
            log_info "Reusing existing Kafka cluster ID: $KAFKA_CLUSTER_ID"
        else
            log_warning "Found meta.properties but no cluster.id, generating new one"
            export KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"
        fi
    else
        log_info "No existing Kafka cluster found, generating new cluster ID"
        export KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"
    fi
    
    # Format storage only if meta.properties doesn't exist or is invalid
    if [[ ! -f "$meta_properties" || -z "$KAFKA_CLUSTER_ID" ]]; then
        log_info "Formatting Kafka storage with cluster ID: $KAFKA_CLUSTER_ID"
        if ! bin/kafka-storage.sh format --standalone -t $KAFKA_CLUSTER_ID -c config/server.properties >/dev/null 2>&1; then
            log_error "Failed to format Kafka storage"
            return 1
        fi
    else
        log_info "Using existing Kafka storage (skipping format)"
    fi
    
    # Start Kafka
    log_info "Starting Kafka..."
    
    # Start Kafka in background
    nohup bin/kafka-server-start.sh config/server.properties >/dev/null 2>&1 &
    local kafka_pid=$!
    
    # Wait for Kafka to start
    log_info "Waiting for Kafka to start..."
    for i in {1..30}; do
        if netstat -ln 2>/dev/null | grep -q ":9092 "; then
            log_success "Kafka started successfully"
            break
        fi
        sleep 2
        if ! kill -0 $kafka_pid 2>/dev/null; then
            log_error "Kafka startup failed"
            return 1
        fi
    done
    
    if ! netstat -ln 2>/dev/null | grep -q ":9092 "; then
        log_error "Kafka failed to start on port 9092"
        return 1
    fi
    
    # Create mv3dt topic
    create_kafka_topic "$KAFKA_DIR"
}

check_kafka_topic() {
    if [[ ! -d "$KAFKA_DIR" || ! -x "$KAFKA_DIR/bin/kafka-topics.sh" ]]; then
        return 1
    fi
    
    local topics=$("$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list 2>/dev/null)
    echo "$topics" | grep -q "mv3dt"
}

create_kafka_topic() {
    local kafka_dir="$1"
    
    log_info "Creating mv3dt topic..."
    if "$kafka_dir/bin/kafka-topics.sh" --bootstrap-server localhost:9092 \
        --create --topic mv3dt --partitions 1 --replication-factor 1 \
        --config retention.ms=30000 --if-not-exists >/dev/null 2>&1; then
        log_success "mv3dt topic created successfully"
    else
        log_error "Failed to create mv3dt topic"
        return 1
    fi
}

setup_python_env() {
    log_info "Setting up Python environment..."
    
    cd "$REPO_ROOT"
    
    # Check and install system packages
    log_info "Checking system Python packages..."
    local packages_to_install=()
    local required_packages=("python3-tk" "python3.12-venv" "python3.12-dev" "python3-pip")
    
    for package in "${required_packages[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
            packages_to_install+=("$package")
        else
            log_info "$package is already installed"
        fi
    done
    
    if [[ ${#packages_to_install[@]} -gt 0 ]]; then
        log_info "Installing missing packages: ${packages_to_install[*]}"
        sudo apt update
        sudo apt install -y "${packages_to_install[@]}"
    else
        log_success "All required Python packages are already installed"
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "mv3dt_venv" ]]; then
        log_info "Creating mv3dt_venv virtual environment..."
        python3 -m venv mv3dt_venv
    fi
    
    # Activate and install requirements
    log_info "Installing Python dependencies..."
    source mv3dt_venv/bin/activate
    
    if [[ -f "requirements.txt" ]]; then
        pip install --quiet --upgrade pip
        pip install --quiet -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_error "requirements.txt not found"
        return 1
    fi
    
    # Verify key packages
    python -c "import kafka, google.protobuf" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        log_success "Python environment setup completed successfully"
    else
        log_error "Failed to import required Python packages"
        return 1
    fi
}

build_custom_parser() {
    log_info "Building custom parser..."

    # Set correct GPU flag considering diffent platforms
    if docker info | grep -q 'Runtimes.*nvidia'; then
        GPU_FLAG="--runtime=nvidia"
    elif docker run --help | grep -q -- "--gpus"; then
        GPU_FLAG="--gpus all"
    else
        echo "No GPU support found in Docker."
        exit 1
    fi
    
    # Capture output and error code
    build_output=$(docker run --privileged --rm --net=host $GPU_FLAG \
        -v $PWD/models:/workspace/models \
        -w /workspace/models/PeopleNetTransformer \
        --entrypoint /bin/bash \
        $DEEPSTREAM_IMAGE \
        -c "cd custom_parser && make clean 2>&1 && make 2>&1" 2>&1)
    build_exit_code=$?

    # Show output only if there were errors
    if [[ $build_exit_code -ne 0 ]]; then
        log_error "Build failed with errors:"
        echo "$build_output"
        return 1
    fi

    if [[ -f "$PWD/models/PeopleNetTransformer/custom_parser/libnvds_infercustomparser_tao.so" ]]; then
        log_success "Custom parser built successfully"
    else
        log_error "Failed to build custom parser: libnvds_infercustomparser_tao.so not found"
        echo "$build_output"
        return 1
    fi
}


setup_inference_builder() {
    if [[ "$USE_INFERENCE_BUILDER" != "true" ]]; then
        log_info "Skipping Inference Builder setup (USE_INFERENCE_BUILDER not set to true)"
        return 0
    fi
    
    log_info "Setting up DeepStream Inference Builder..."
    
    if [[ ! -d "$INFERENCE_BUILDER_DIR" ]]; then
        log_info "Cloning Inference Builder repository..."
        cd "$BASE_DIR"
        if ! git clone https://github.com/NVIDIA-AI-IOT/inference_builder.git; then
            log_warning "Failed to clone Inference Builder repository"
            log_info "This is optional - Inference Builder is skipped by default. Set USE_INFERENCE_BUILDER=true to enable it."
            return 0
        fi
        
        cd inference_builder
        git submodule update --init --recursive
    else
        log_success "Inference Builder directory already exists"
        cd "$INFERENCE_BUILDER_DIR"
    fi
    
    # Install system dependencies
    log_info "Checking protobuf-compiler package..."
    if ! dpkg-query -W -f='${Status}' protobuf-compiler 2>/dev/null | grep -q "install ok installed"; then
        log_info "Installing protobuf-compiler..."
        sudo apt update && sudo apt install -y protobuf-compiler
    else
        log_success "protobuf-compiler is already installed"
    fi
    
    # Create virtual environment
    if [[ ! -d "ib_venv" ]]; then
        log_info "Creating Inference Builder virtual environment..."
        python -m venv ib_venv
    fi
    
    # Install requirements
    log_info "Installing Inference Builder dependencies..."
    source ib_venv/bin/activate
    
    if [[ -f "requirements.txt" ]]; then
        pip install --quiet --upgrade pip
        pip install --quiet -r requirements.txt
    else
        log_warning "Inference Builder requirements.txt not found, skipping pip install"
    fi

    # Create custom DeepStream Docker image with MV3DT packages (always)
    log_info "Installing inference builder dependencies into DeepStream container..."

    # Create Dockerfile in temporary location
    cat > ./Dockerfile.mv3dt << EOF
FROM ${DEEPSTREAM_IMAGE}

# Install IB python packages
RUN pip3 install torch==2.7.0 omegaconf==2.3.0

# Set environment variables
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes

# Set working directory
WORKDIR /mv3dt_app
EOF

    # Build the custom image
    # Before building, check whether the image already exists
    if docker inspect inference-builder-mv3dt:latest >/dev/null 2>&1; then
        log_info "Target docker image inference-builder-mv3dt:latest already exists"
        log_success "Inference Builder setup completed"
        return 0
    fi

    # Build the Docker image
    if docker build -f ./Dockerfile.mv3dt -t inference-builder-mv3dt:latest .; then
        log_info "DeepStream container with Inference Builder dependencies saved as docker image inference-builder-mv3dt:latest"
        rm -f ./Dockerfile.mv3dt
    else
        log_error "Failed to build custom DeepStream Docker image"
        rm -f ./Dockerfile.mv3dt
        return 1
    fi

    log_success "Inference Builder setup completed"
}

run_prerequisites_check() {
    log_info "Running prerequisites check..."
    cd "$REPO_ROOT"
    
    if [[ -x "scripts/check_prerequisites.sh" ]]; then
        # Make mosquitto test script executable if it exists
        [[ -f "scripts/mosquitto_test.sh" ]] && chmod +x scripts/mosquitto_test.sh
        
        if bash scripts/check_prerequisites.sh; then
            return 0
        else
            log_error "Prerequisites check failed"
            return 1
        fi
    else
        log_error "Prerequisites check script not found"
        return 1
    fi
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated setup script for MV3DT prerequisites.

OPTIONS:
    -h, --help                 Show this help message
    --check-only               Only run prerequisites check without setup
    
ENVIRONMENT VARIABLES:
    BASE_DIR                   Base directory for installations (default: $HOME)
    USE_INFERENCE_BUILDER      Set to 'true' to enable Inference Builder setup (default: false)

EOF
}

main() {
    local check_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    echo "=============================================="
    echo "    MV3DT Prerequisites Setup Script"
    echo "=============================================="
    echo
    
    if [[ "$check_only" == "true" ]]; then
        run_prerequisites_check
        exit $?
    fi
    
    check_os
    
    # Check if we're in the right directory
    if [[ ! -f "$REPO_ROOT/README.md" ]] || ! grep -q "Multi-View 3D Tracking" "$REPO_ROOT/README.md"; then
        log_error "Please run this script from the deepstream-tracker-3d-multi-view repository root"
        exit 1
    fi
    
    log_info "Starting automated prerequisites setup..."
    
    # Run setup steps
    local failed_steps=()
    
    if ! check_gpu; then
        failed_steps+=("GPU/NVIDIA drivers")
    fi
    
    if ! setup_deepstream_container; then
        failed_steps+=("DeepStream container")
    fi
    
    if ! setup_git_lfs; then
        failed_steps+=("Git LFS/Datasets/Models")
    fi

    if ! build_custom_parser; then
        failed_steps+=("Build Custom parser")
    fi
    
    if ! setup_mosquitto; then
        failed_steps+=("Mosquitto MQTT")
    fi
    
    if ! setup_kafka; then
        failed_steps+=("Kafka")
    fi
    
    if ! setup_python_env; then
        failed_steps+=("Python environment")
    fi
    
    if ! setup_inference_builder; then
        failed_steps+=("Inference Builder")
    fi
    
    echo
    echo "=============================================="
    echo "           Setup Summary"
    echo "=============================================="
    
    if [[ ${#failed_steps[@]} -eq 0 ]]; then
        log_success "All setup steps completed successfully!"
        echo
        log_info "Configured Paths:"
        echo -e "  ${BLUE}Base Directory:${NC}          $BASE_DIR"
        echo -e "  ${BLUE}Kafka Installation:${NC}      $KAFKA_DIR"
        if [[ "$USE_INFERENCE_BUILDER" == "true" ]]; then
            echo -e "  ${BLUE}Inference Builder:${NC}       $INFERENCE_BUILDER_DIR"
        fi
        echo -e "  ${BLUE}MV3DT Repo:${NC}              $REPO_ROOT"
        echo
        
        # Run final check
        if run_prerequisites_check; then
            log_success "Prerequisites check passed! You're ready to use MV3DT."
            echo
        else
            log_error "Setup completed but prerequisites check still failed"
            echo "Please review the error messages and run the script again"
            exit 1
        fi
    else
        log_error "The following setup steps failed:"
        for step in "${failed_steps[@]}"; do
            echo "  - $step"
        done
        echo
        echo "Please review the error messages above and:"
        echo "  1. Fix the issues manually"
        echo "  2. Run this script again"
        echo "  3. Or run specific setup steps as needed"
        exit 1
    fi
    
    echo "=============================================="
}

# Handle script interruption
trap 'log_error "Setup interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"