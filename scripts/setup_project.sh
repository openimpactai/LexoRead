#!/bin/bash
# setup_project.sh
#
# This script sets up the development environment for the LexoRead project.
# It installs dependencies, downloads models, and configures the project.

set -e  # Exit on error

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Print header
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}     LexoRead Project Setup Script       ${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python $python_version"

python_version_major=$(echo $python_version | cut -d. -f1)
python_version_minor=$(echo $python_version | cut -d. -f2)

if [ $python_version_major -lt 3 ] || ([ $python_version_major -eq 3 ] && [ $python_version_minor -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or higher is required (detected $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}Python version check passed.${NC}"

# Check if running in the correct directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"
if [ ! -f "$repo_root/README.md" ]; then
    echo -e "${RED}Error: Script must be run from the repository root or scripts directory${NC}"
    exit 1
fi

cd "$repo_root"
echo -e "\n${YELLOW}Working directory: $(pwd)${NC}"

# Create virtual environment if it doesn't exist
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Created virtual environment in ./venv${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"

# Check if Poetry is installed
if command -v poetry &> /dev/null; then
    echo "Poetry found, installing dependencies with Poetry..."
    # Install API dependencies
    if [ -f "api/pyproject.toml" ]; then
        echo "Installing API dependencies..."
        cd api
        poetry install
        cd ..
    fi
else
    echo "Poetry not found, installing dependencies with pip..."
    # Install common dependencies
    pip install -r requirements.txt
    
    # Install API dependencies
    if [ -f "api/requirements.txt" ]; then
        echo "Installing API dependencies..."
        pip install -r api/requirements.txt
    fi
fi

echo -e "${GREEN}Dependencies installed.${NC}"

# Set up configuration
echo -e "\n${YELLOW}Setting up configuration...${NC}"

# Create .env file for API if it doesn't exist
if [ ! -f "api/.env" ] && [ -f "api/.env.example" ]; then
    echo "Creating API .env file from example..."
    cp api/.env.example api/.env
    echo -e "${GREEN}Created api/.env file.${NC}"
    echo -e "${YELLOW}Note: You should edit the .env file to set your own secret key and other configuration.${NC}"
else
    echo -e "${GREEN}API .env file already exists.${NC}"
fi

# Create config directory if it doesn't exist
if [ ! -d "models/config" ]; then
    echo "Creating models config directory..."
    mkdir -p models/config
    echo -e "${GREEN}Created models/config directory.${NC}"
fi

# Download models
echo -e "\n${YELLOW}Downloading pre-trained models...${NC}"
if [ ! -f "models/download_pretrained.py" ]; then
    echo -e "${RED}Error: models/download_pretrained.py script not found${NC}"
else
    python models/download_pretrained.py --demo
    echo -e "${GREEN}Demo models downloaded.${NC}"
    echo -e "${YELLOW}Note: These are demo models. For production use, download the full models with:${NC}"
    echo -e "${YELLOW}  python models/download_pretrained.py --models all${NC}"
fi

# Download datasets
echo -e "\n${YELLOW}Downloading datasets...${NC}"
if [ ! -f "datasets/download_scripts/download_all.py" ]; then
    echo -e "${RED}Error: datasets/download_scripts/download_all.py script not found${NC}"
else
    python datasets/download_scripts/download_all.py --sample
    echo -e "${GREEN}Sample datasets downloaded.${NC}"
    echo -e "${YELLOW}Note: These are sample datasets. For full datasets, run:${NC}"
    echo -e "${YELLOW}  python datasets/download_scripts/download_all.py${NC}"
fi

# Set up git hooks
echo -e "\n${YELLOW}Setting up git hooks...${NC}"
if [ -d ".git" ]; then
    if [ ! -d ".git/hooks" ]; then
        mkdir -p .git/hooks
    fi
    
    # Create pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to run linters

# Stash unstaged changes
git stash -q --keep-index

# Run linters
echo "Running linters..."
python -m black api models datasets scripts --check
RESULT_BLACK=$?

python -m isort api models datasets scripts --check
RESULT_ISORT=$?

python -m flake8 api models datasets scripts
RESULT_FLAKE8=$?

# Restore unstaged changes
git stash pop -q

# Return status
if [ $RESULT_BLACK -ne 0 ] || [ $RESULT_ISORT -ne 0 ] || [ $RESULT_FLAKE8 -ne 0 ]; then
    echo "Linting failed. Please fix the issues before committing."
    exit 1
fi

exit 0
EOF
    
    chmod +x .git/hooks/pre-commit
    echo -e "${GREEN}Git hooks set up.${NC}"
else
    echo -e "${YELLOW}Not a git repository, skipping git hooks setup.${NC}"
fi

# Final message
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}     LexoRead setup complete!            ${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "\nTo run the API:"
echo -e "  cd api"
echo -e "  uvicorn app:app --reload"
echo -e "\nTo deactivate the virtual environment:"
echo -e "  deactivate"
echo -e "\nHappy coding!"
