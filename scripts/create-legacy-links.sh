#!/bin/bash

# LeanVibe Agent Hive 2.0 - Create Legacy Compatibility Links
# Creates wrapper links for backward compatibility during transition

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SCRIPTS_DIR="$PROJECT_ROOT/scripts"

cd "$PROJECT_ROOT"

# Color codes
readonly BLUE='\033[0;34m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

echo -e "${BLUE}Creating legacy compatibility links...${NC}"

# Create wrapper scripts for common legacy commands
cat > setup-wrapper.sh << 'EOF'
#!/bin/bash
echo -e "\033[1;33m📢 MIGRATION NOTICE\033[0m"
echo "The legacy setup.sh script is deprecated."
echo -e "Please use: \033[0;34mmake setup\033[0m"
echo
echo "Redirecting to new setup script..."
exec ./scripts/setup.sh fast
EOF

cat > start-wrapper.sh << 'EOF'
#!/bin/bash
echo -e "\033[1;33m📢 MIGRATION NOTICE\033[0m"
echo "The legacy start-fast.sh script is deprecated." 
echo -e "Please use: \033[0;34mmake start\033[0m"
echo
echo "Redirecting to new start script..."
exec ./scripts/start.sh fast
EOF

chmod +x setup-wrapper.sh start-wrapper.sh

# Backup existing scripts
if [[ -f "setup.sh" ]] && [[ ! -L "setup.sh" ]]; then
    mv setup.sh setup-legacy-backup.sh
    echo -e "${YELLOW}Backed up original setup.sh to setup-legacy-backup.sh${NC}"
fi

if [[ -f "start.sh" ]] && [[ ! -L "start.sh" ]]; then
    mv start.sh start-legacy-backup.sh  
    echo -e "${YELLOW}Backed up original start.sh to start-legacy-backup.sh${NC}"
fi

# Create compatibility links
ln -sf setup-wrapper.sh setup.sh
ln -sf scripts/setup.sh setup-new.sh
ln -sf scripts/start.sh start-new.sh

echo -e "${GREEN}✅ Legacy compatibility links created${NC}"
echo
echo "Available commands:"
echo "  make setup          # Recommended new command"
echo "  ./setup.sh          # Legacy compatibility (with migration notice)"
echo "  ./scripts/setup.sh  # Direct access to new script"
echo