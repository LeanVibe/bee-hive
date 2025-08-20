#!/usr/bin/env python3
"""
Configuration & Environment Debt Analyzer
==========================================

Phase 7: Final phase of advanced technical debt remediation.
Analyzes configuration files, environment setups, Docker files, CI/CD pipelines,
and deployment scripts for redundancy and consolidation opportunities.

Target Areas:
- Database configurations across environments
- API configurations and endpoints
- Logging configurations 
- Security configurations
- Docker and containerization setup
- CI/CD pipeline redundancy
- Environment variable management
- Deployment script consolidation
"""

import os
import re
import json
import yaml
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
import tempfile
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationPattern:
    """Represents a configuration consolidation pattern."""
    pattern_type: str
    pattern_name: str
    similar_configs: List[Dict]
    content_similarity: float
    consolidation_potential: int
    affected_files: List[Path]
    template_config: Optional[Dict] = None

class ConfigurationDebtAnalyzer:
    """Analyzes configuration and environment debt for consolidation opportunities."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.consolidated_configs_dir = self.project_root / "config" / "consolidated"
        self.backup_dir = Path(tempfile.mkdtemp(prefix="config_backups_"))
        
        # Configuration file extensions and patterns
        self.config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', 
            '.env', '.properties', '.xml'
        }
        
        # Docker and deployment file patterns
        self.deployment_patterns = {
            'docker': ['Dockerfile', 'docker-compose', '.dockerignore'],
            'ci_cd': ['.github', '.gitlab-ci', 'jenkinsfile', '.travis', '.circleci'],
            'deployment': ['deploy', 'kubernetes', 'helm', 'terraform'],
            'scripts': ['setup', 'install', 'configure', 'build', 'start']
        }
        
        # Configuration categories for analysis
        self.config_categories = {
            'database': ['db', 'database', 'postgres', 'mysql', 'mongo', 'redis'],
            'api': ['api', 'server', 'host', 'port', 'endpoint', 'url'],
            'logging': ['log', 'logger', 'logLevel', 'logging'],
            'security': ['auth', 'jwt', 'secret', 'key', 'cert', 'ssl', 'tls'],
            'cache': ['cache', 'redis', 'memcache', 'ttl'],
            'environment': ['env', 'NODE_ENV', 'ENVIRONMENT', 'stage', 'prod'],
            'monitoring': ['monitor', 'metrics', 'health', 'telemetry'],
            'messaging': ['queue', 'kafka', 'rabbitmq', 'pubsub', 'websocket']
        }
    
    def discover_configuration_files(self) -> Dict[str, List[Path]]:
        """Discover all configuration and deployment files."""
        config_files = defaultdict(list)
        
        # Find configuration files by extension
        for ext in self.config_extensions:
            for config_file in self.project_root.rglob(f"*{ext}"):
                if not any(skip in str(config_file).lower() for skip in ['.venv', 'venv', 'node_modules', '__pycache__', '.git']):
                    config_files['config_files'].append(config_file)
        
        # Find deployment and infrastructure files
        for category, patterns in self.deployment_patterns.items():
            for pattern in patterns:
                # Look for exact matches and patterns
                for file_path in self.project_root.rglob(f"*{pattern}*"):
                    if file_path.is_file() and not any(skip in str(file_path).lower() for skip in ['.venv', 'venv', 'node_modules', '__pycache__', '.git']):
                        config_files[category].append(file_path)
        
        # Print discovery summary
        total_files = sum(len(files) for files in config_files.values())
        print(f"🔍 Discovered {total_files} configuration and deployment files:")
        for category, files in config_files.items():
            if files:
                print(f"   📄 {category}: {len(files)} files")
        
        return dict(config_files)
    
    def analyze_configuration_content(self, config_files: Dict[str, List[Path]]) -> Dict[str, List[Dict]]:
        """Analyze content of configuration files for patterns."""
        print("⚙️ Analyzing configuration content patterns...")
        
        config_analysis = {}
        content_patterns = defaultdict(list)
        
        all_files = []
        for file_list in config_files.values():
            all_files.extend(file_list)
        
        for config_file in all_files:
            try:
                analysis = self.analyze_single_configuration(config_file)
                if analysis:
                    config_analysis[str(config_file)] = analysis
                    
                    # Group by configuration patterns
                    for pattern_type in analysis.get('categories', []):
                        content_patterns[pattern_type].append({
                            'file': config_file,
                            'analysis': analysis
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to analyze {config_file}: {e}")
        
        print(f"   📄 Analyzed {len(config_analysis)} configuration files")
        print(f"   🔍 Found {len(content_patterns)} configuration pattern types")
        
        return dict(content_patterns)
    
    def analyze_single_configuration(self, config_file: Path) -> Optional[Dict]:
        """Analyze a single configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse content based on file type
            parsed_content = self.parse_configuration_content(config_file, content)
            
            # Extract configuration keys and values
            config_keys = self.extract_configuration_keys(parsed_content, content)
            
            # Classify configuration categories
            categories = self.classify_configuration_categories(config_file, config_keys, content)
            
            # Calculate content fingerprint
            normalized_content = self.normalize_configuration_content(content)
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()[:12]
            
            # Analyze configuration structure
            structure_info = self.analyze_configuration_structure(parsed_content, content)
            
            return {
                'file_path': config_file,
                'file_type': config_file.suffix,
                'parsed_content': parsed_content,
                'config_keys': config_keys,
                'categories': categories,
                'content_hash': content_hash,
                'structure': structure_info,
                'size': len(content),
                'line_count': len(content.split('\n'))
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing configuration {config_file}: {e}")
            return None
    
    def parse_configuration_content(self, config_file: Path, content: str) -> Optional[Dict]:
        """Parse configuration content based on file type."""
        try:
            file_ext = config_file.suffix.lower()
            
            if file_ext in ['.json']:
                return json.loads(content)
            elif file_ext in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif file_ext in ['.env']:
                return self.parse_env_file(content)
            elif file_ext in ['.ini', '.cfg', '.conf']:
                return self.parse_ini_file(content)
            elif config_file.name.lower().startswith('dockerfile'):
                return self.parse_dockerfile(content)
            else:
                return {'raw_content': content}
                
        except Exception as e:
            logger.debug(f"Failed to parse {config_file}: {e}")
            return {'raw_content': content}
    
    def parse_env_file(self, content: str) -> Dict:
        """Parse .env file content."""
        env_vars = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
        return env_vars
    
    def parse_ini_file(self, content: str) -> Dict:
        """Parse INI/CFG file content."""
        import configparser
        config = configparser.ConfigParser()
        try:
            config.read_string(content)
            result = {}
            for section in config.sections():
                result[section] = dict(config[section])
            return result
        except:
            return {'raw_content': content}
    
    def parse_dockerfile(self, content: str) -> Dict:
        """Parse Dockerfile content."""
        instructions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if parts:
                    instructions.append({
                        'instruction': parts[0].upper(),
                        'value': parts[1] if len(parts) > 1 else ''
                    })
        return {'instructions': instructions}
    
    def extract_configuration_keys(self, parsed_content: Optional[Dict], raw_content: str) -> List[str]:
        """Extract configuration keys from content."""
        keys = []
        
        if parsed_content and isinstance(parsed_content, dict):
            keys.extend(self.get_nested_keys(parsed_content))
        
        # Also extract from raw content using regex
        key_patterns = [
            r'(\w+)\s*[=:]\s*',  # key=value or key: value
            r'--(\w+)',  # command line args
            r'\$\{?(\w+)\}?',  # environment variables
            r'(\w+)_\w+',  # compound keys like DB_HOST
        ]
        
        for pattern in key_patterns:
            matches = re.findall(pattern, raw_content, re.IGNORECASE)
            keys.extend(matches)
        
        return list(set(key.lower() for key in keys if key))
    
    def get_nested_keys(self, data: Dict, prefix: str = '') -> List[str]:
        """Recursively extract nested dictionary keys."""
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self.get_nested_keys(value, full_key))
        
        return keys
    
    def classify_configuration_categories(self, config_file: Path, config_keys: List[str], content: str) -> List[str]:
        """Classify configuration into categories."""
        categories = []
        file_name = config_file.name.lower()
        content_lower = content.lower()
        
        for category, keywords in self.config_categories.items():
            # Check filename
            filename_matches = sum(1 for keyword in keywords if keyword in file_name)
            
            # Check configuration keys
            key_matches = sum(1 for key in config_keys for keyword in keywords if keyword in key.lower())
            
            # Check content
            content_matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            # Calculate category score
            total_score = filename_matches * 3 + key_matches * 2 + content_matches
            
            if total_score >= 2:  # Minimum threshold
                categories.append(category)
        
        return categories if categories else ['general']
    
    def normalize_configuration_content(self, content: str) -> str:
        """Normalize configuration content for similarity comparison."""
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip().lower()
        
        # Remove environment-specific values but keep structure
        content = re.sub(r'(localhost|127\.0\.0\.1|\d+\.\d+\.\d+\.\d+)', 'HOST', content)
        content = re.sub(r':\d+', ':PORT', content)
        content = re.sub(r'(dev|development|prod|production|staging|test)', 'ENV', content)
        
        return content
    
    def analyze_configuration_structure(self, parsed_content: Optional[Dict], raw_content: str) -> Dict:
        """Analyze configuration structure and complexity."""
        structure = {
            'type': 'unknown',
            'complexity': 0,
            'depth': 0,
            'key_count': 0
        }
        
        if parsed_content:
            if isinstance(parsed_content, dict):
                structure['type'] = 'dictionary'
                structure['key_count'] = len(parsed_content)
                structure['depth'] = self.calculate_dict_depth(parsed_content)
                structure['complexity'] = self.calculate_complexity_score(parsed_content)
            elif isinstance(parsed_content, list):
                structure['type'] = 'array'
                structure['complexity'] = len(parsed_content)
        else:
            structure['type'] = 'text'
            structure['complexity'] = len(raw_content.split('\n'))
        
        return structure
    
    def calculate_dict_depth(self, data: Dict, current_depth: int = 1) -> int:
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(data, dict):
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self.calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def calculate_complexity_score(self, data: Any) -> int:
        """Calculate configuration complexity score."""
        if isinstance(data, dict):
            return len(data) + sum(self.calculate_complexity_score(v) for v in data.values())
        elif isinstance(data, list):
            return len(data) + sum(self.calculate_complexity_score(item) for item in data)
        else:
            return 1
    
    def find_similar_configurations(self, content_patterns: Dict[str, List[Dict]]) -> List[ConfigurationPattern]:
        """Find similar configurations for consolidation."""
        print("🔍 Finding similar configuration patterns...")
        
        similar_patterns = []
        
        for pattern_type, configurations in content_patterns.items():
            if len(configurations) < 2:  # Need at least 2 configurations
                continue
            
            print(f"   ⚙️ Analyzing {pattern_type}: {len(configurations)} configurations")
            
            # Group similar configurations within this pattern type
            similar_groups = self.group_similar_configurations(configurations)
            
            for group in similar_groups:
                if len(group) >= 2:  # At least 2 similar configurations
                    pattern = self.create_configuration_pattern(pattern_type, group)
                    similar_patterns.append(pattern)
        
        return similar_patterns
    
    def group_similar_configurations(self, configurations: List[Dict]) -> List[List[Dict]]:
        """Group similar configurations for consolidation."""
        groups = []
        processed = set()
        
        for i, config1 in enumerate(configurations):
            if i in processed:
                continue
            
            group = [config1]
            processed.add(i)
            
            for j, config2 in enumerate(configurations[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_configuration_similarity(
                    config1['analysis'], config2['analysis']
                )
                
                if similarity > 0.6:  # Similarity threshold
                    group.append(config2)
                    processed.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def calculate_configuration_similarity(self, config1: Dict, config2: Dict) -> float:
        """Calculate similarity between two configurations."""
        similarities = []
        
        # Key similarity
        keys1 = set(config1.get('config_keys', []))
        keys2 = set(config2.get('config_keys', []))
        if keys1 or keys2:
            key_sim = len(keys1 & keys2) / len(keys1 | keys2)
            similarities.append(key_sim)
        
        # Category similarity
        cats1 = set(config1.get('categories', []))
        cats2 = set(config2.get('categories', []))
        if cats1 or cats2:
            cat_sim = len(cats1 & cats2) / len(cats1 | cats2)
            similarities.append(cat_sim)
        
        # Structure similarity
        struct1 = config1.get('structure', {})
        struct2 = config2.get('structure', {})
        if struct1.get('type') == struct2.get('type'):
            similarities.append(0.5)  # Same structure type
            
            # Size similarity
            size1, size2 = struct1.get('key_count', 0), struct2.get('key_count', 0)
            if size1 and size2:
                size_sim = min(size1, size2) / max(size1, size2)
                similarities.append(size_sim)
        
        # Content hash similarity (exact matches get bonus)
        if config1.get('content_hash') == config2.get('content_hash'):
            similarities.append(1.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def create_configuration_pattern(self, pattern_type: str, configurations: List[Dict]) -> ConfigurationPattern:
        """Create a configuration consolidation pattern."""
        # Calculate consolidation potential (based on file sizes)
        total_size = sum(config['analysis'].get('size', 0) for config in configurations)
        consolidation_potential = int(total_size * 0.6)  # Estimate 60% reduction
        
        # Calculate average similarity
        similarities = []
        for i, config1 in enumerate(configurations):
            for config2 in configurations[i+1:]:
                sim = self.calculate_configuration_similarity(config1['analysis'], config2['analysis'])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Generate pattern name
        file_names = [config['file'].stem for config in configurations]
        common_words = self.find_common_words(file_names)
        pattern_name = f"{pattern_type}_{common_words}" if common_words else f"{pattern_type}_consolidated"
        
        # Create template configuration
        template_config = self.create_template_configuration(configurations)
        
        return ConfigurationPattern(
            pattern_type=pattern_type,
            pattern_name=pattern_name,
            similar_configs=configurations,
            content_similarity=avg_similarity,
            consolidation_potential=consolidation_potential,
            affected_files=[config['file'] for config in configurations],
            template_config=template_config
        )
    
    def find_common_words(self, names: List[str]) -> str:
        """Find common words in configuration names."""
        words = []
        for name in names:
            words.extend(re.findall(r'[a-zA-Z]+', name.lower()))
        
        word_counts = Counter(words)
        common_words = [word for word, count in word_counts.most_common(2) if count > 1]
        return '_'.join(common_words[:2]) if common_words else 'unified'
    
    def create_template_configuration(self, configurations: List[Dict]) -> Dict:
        """Create a template configuration from similar configurations."""
        template = {
            'type': 'consolidated_configuration',
            'description': f'Template configuration from {len(configurations)} similar files',
            'common_keys': [],
            'environment_variables': {},
            'default_values': {}
        }
        
        # Find common configuration keys
        all_keys = []
        for config in configurations:
            all_keys.extend(config['analysis'].get('config_keys', []))
        
        key_counts = Counter(all_keys)
        common_keys = [key for key, count in key_counts.items() if count >= len(configurations) * 0.5]
        template['common_keys'] = common_keys
        
        # Extract common patterns
        for config in configurations:
            parsed = config['analysis'].get('parsed_content', {})
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if key.lower() in common_keys:
                        template['default_values'][key] = value
        
        return template
    
    def generate_consolidation_report(self, patterns: List[ConfigurationPattern], dry_run: bool = False) -> Dict:
        """Generate configuration consolidation report."""
        print("📋 Generating configuration consolidation report...")
        
        total_patterns = len(patterns)
        total_files = sum(len(pattern.affected_files) for pattern in patterns)
        total_savings = sum(pattern.consolidation_potential for pattern in patterns)
        
        # Group by pattern type
        by_type = defaultdict(list)
        for pattern in patterns:
            by_type[pattern.pattern_type].append(pattern)
        
        type_analysis = {}
        for pattern_type, type_patterns in by_type.items():
            type_analysis[pattern_type] = {
                'count': len(type_patterns),
                'total_files': sum(len(p.affected_files) for p in type_patterns),
                'total_savings': sum(p.consolidation_potential for p in type_patterns),
                'avg_similarity': sum(p.content_similarity for p in type_patterns) / len(type_patterns)
            }
        
        # Create consolidated configurations if not dry run
        if not dry_run:
            self.create_consolidated_configurations(patterns)
        
        return {
            'summary': {
                'total_configuration_patterns': total_patterns,
                'total_files_affected': total_files,
                'total_consolidation_potential': total_savings,
                'top_opportunities': sorted(by_type.items(), 
                                          key=lambda x: type_analysis[x[0]]['total_savings'], 
                                          reverse=True)[:5]
            },
            'by_type': type_analysis,
            'patterns': patterns
        }
    
    def create_consolidated_configurations(self, patterns: List[ConfigurationPattern]) -> None:
        """Create consolidated configuration files."""
        print("⚙️ Creating consolidated configuration files...")
        
        self.consolidated_configs_dir.mkdir(parents=True, exist_ok=True)
        
        created_count = 0
        for pattern in patterns:
            if len(pattern.similar_configs) >= 2:
                consolidated_file = self.create_consolidated_config_file(pattern)
                if consolidated_file:
                    created_count += 1
                    logger.info(f"Created consolidated configuration: {consolidated_file}")
        
        print(f"   ✅ Created {created_count} consolidated configuration files")
    
    def create_consolidated_config_file(self, pattern: ConfigurationPattern) -> Optional[Path]:
        """Create a consolidated configuration file from a pattern."""
        # Determine appropriate file extension
        file_extensions = [config['analysis'].get('file_type', '.yaml') for config in pattern.similar_configs]
        most_common_ext = Counter(file_extensions).most_common(1)[0][0] if file_extensions else '.yaml'
        
        consolidated_path = self.consolidated_configs_dir / f"{pattern.pattern_name}{most_common_ext}"
        
        # Generate consolidated content
        if most_common_ext in ['.yaml', '.yml']:
            content = self.generate_yaml_config(pattern)
        elif most_common_ext == '.json':
            content = self.generate_json_config(pattern)
        elif most_common_ext == '.env':
            content = self.generate_env_config(pattern)
        else:
            content = self.generate_generic_config(pattern)
        
        try:
            consolidated_path.write_text(content)
            return consolidated_path
        except Exception as e:
            logger.error(f"Failed to create consolidated config {consolidated_path}: {e}")
            return None
    
    def generate_yaml_config(self, pattern: ConfigurationPattern) -> str:
        """Generate consolidated YAML configuration."""
        config_data = {
            'metadata': {
                'name': pattern.pattern_name,
                'type': pattern.pattern_type,
                'description': f'Consolidated {pattern.pattern_type} configuration',
                'generated_on': datetime.now().isoformat(),
                'original_files': [str(config['file']) for config in pattern.similar_configs],
                'consolidation_potential': f'{pattern.consolidation_potential} bytes'
            },
            'configuration': pattern.template_config or {}
        }
        
        return yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    
    def generate_json_config(self, pattern: ConfigurationPattern) -> str:
        """Generate consolidated JSON configuration."""
        config_data = {
            'metadata': {
                'name': pattern.pattern_name,
                'type': pattern.pattern_type,
                'description': f'Consolidated {pattern.pattern_type} configuration',
                'generated_on': datetime.now().isoformat(),
                'original_files': [str(config['file']) for config in pattern.similar_configs],
                'consolidation_potential': f'{pattern.consolidation_potential} bytes'
            },
            'configuration': pattern.template_config or {}
        }
        
        return json.dumps(config_data, indent=2, sort_keys=False)
    
    def generate_env_config(self, pattern: ConfigurationPattern) -> str:
        """Generate consolidated .env configuration."""
        content = f"""# Consolidated {pattern.pattern_type} configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Original files: {', '.join(config['file'].name for config in pattern.similar_configs)}
# Consolidation potential: {pattern.consolidation_potential} bytes

"""
        
        # Add common environment variables
        if pattern.template_config and 'default_values' in pattern.template_config:
            for key, value in pattern.template_config['default_values'].items():
                if isinstance(value, str):
                    content += f"{key.upper()}={value}\n"
        
        return content
    
    def generate_generic_config(self, pattern: ConfigurationPattern) -> str:
        """Generate consolidated generic configuration."""
        content = f"""# Consolidated {pattern.pattern_type} Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Original configurations consolidated:
"""
        for config in pattern.similar_configs:
            content += f"- {config['file']} ({config['analysis'].get('size', 0)} bytes)\n"
        
        content += f"""
Consolidation Benefits:
- Estimated size reduction: {pattern.consolidation_potential} bytes
- Content similarity: {pattern.content_similarity:.1%}
- Single source of truth for {pattern.pattern_type} configuration

Template Configuration:
{json.dumps(pattern.template_config, indent=2) if pattern.template_config else 'Template not available'}
"""
        
        return content

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze configuration and environment debt')
    parser.add_argument('--analyze', action='store_true', help='Analyze configuration consolidation opportunities')
    parser.add_argument('--consolidate', action='store_true', help='Create consolidated configuration files')
    
    args = parser.parse_args()
    
    analyzer = ConfigurationDebtAnalyzer()
    
    print("⚙️ Configuration & Environment Debt Analyzer - Phase 7")
    print("=" * 60)
    
    # Discover configuration files
    config_files = analyzer.discover_configuration_files()
    
    if not any(config_files.values()):
        print("❌ No configuration files found.")
        return
    
    # Analyze content patterns
    content_patterns = analyzer.analyze_configuration_content(config_files)
    
    # Find similar configurations
    similar_patterns = analyzer.find_similar_configurations(content_patterns)
    
    if not similar_patterns:
        print("❌ No configuration consolidation patterns found.")
        return
    
    # Generate report
    if args.consolidate:
        print("🚀 Creating consolidated configuration files...")
        report = analyzer.generate_consolidation_report(similar_patterns, dry_run=False)
    else:
        print("📋 Analyzing configuration consolidation opportunities...")
        report = analyzer.generate_consolidation_report(similar_patterns, dry_run=True)
    
    # Print results
    summary = report['summary']
    total_config_files = sum(len(files) for files in config_files.values())
    
    print(f"\n📊 Configuration & Environment Debt Analysis Results:")
    print(f"   ⚙️ {total_config_files} total configuration files discovered")
    print(f"   🔍 {summary['total_configuration_patterns']} consolidation patterns found")
    print(f"   📄 {summary['total_files_affected']} files affected")
    print(f"   💰 {summary['total_consolidation_potential']:,} bytes consolidation potential")
    
    print(f"\n🏆 Top Configuration Consolidation Opportunities:")
    for pattern_type, patterns in summary['top_opportunities']:
        analysis = report['by_type'][pattern_type]
        print(f"   • {pattern_type.title()}: {analysis['count']} patterns, {analysis['total_savings']:,} bytes savings")
    
    if args.consolidate:
        consolidated_count = len([p for p in similar_patterns if len(p.similar_configs) >= 2])
        print(f"\n✅ Created {consolidated_count} consolidated configuration files")
        print(f"📂 Consolidated configurations in: config/consolidated/")

if __name__ == "__main__":
    main()