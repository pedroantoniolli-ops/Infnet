
import sys
import json
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

# Definiçoes

ROOT_DIR = Path(__file__).parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG_PATH = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG_PATH = CONFIG_DIR / 'data.yaml'
QUALITY_CONFIG_PATH = CONFIG_DIR / 'quality.yaml'

# Adiciona o caminho do diretório raiz ao sys.path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(CONFIG_DIR))

from src.utils.logger import get_logger, logging
from src.utils.config_loader import load_yaml
from src.quality_checks import run_quality_checks, save_quality_report

# Carregar os arquivos de configuração
pipeline_config = load_yaml(PIPELINE_CONFIG_PATH)
data_config = load_yaml(DATA_CONFIG_PATH)
quality_config = load_yaml(QUALITY_CONFIG_PATH)

# Unir os dicionários em um único
config = {}
config.update(pipeline_config)
config.update(data_config)
config.update(quality_config)

# Configurar o logger
log_cfg = config.get('logging', {})
logger = get_logger(
    name=log_cfg.get('name', 'QualidadeLogger'),
    level=getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO
    ),
    log_to_file=log_cfg.get('log_to_file', False),
    log_dir=log_cfg.get('log_dir', 'logs'),
    log_file=log_cfg.get('log_file', 'quality.log')
)
logger.info("Logger configured successfully for quality checks.")

logger.info("Starting quality checks with the following configuration:")
for key, value in config.items():
    logger.info(f"{key}: {value}")


# Obtendo o caminho do arquivo Parquet processado
processed_dir = ROOT_DIR / Path(config.get('pipeline', {}).get('processed_data_dir', 'data/processed'))
parquet_file = processed_dir / config.get('paths').get('output_filename')

if not parquet_file.exists():
    logger.error(f"Processed data file not found at: {parquet_file}")
    raise FileNotFoundError(f"Processed data file not found at: {parquet_file}")

logger.info(f"Output path for processed data: {parquet_file}")


schema = pq.read_schema(parquet_file)
logger.info("Schema of the processed Parquet file:")
for field in schema:
    logger.info(f" - {field.name}: {field.type}")


logger.info("Loading the processed Parquet file into a pandas DataFrame...")
df = pd.read_parquet(parquet_file)
logger.info(f"DataFrame loaded successfully with shape: {df.shape}")


# YAML as SST
logger.info("Performing quality checks based on the quality configuration...")
summary = run_quality_checks(df, config=config, logging_config=log_cfg)
# print(json.dumps(summary, indent=4))
# logger.info(summary)

# Salvar o resumo das verificações de qualidade em um arquivo JSON
output_dir = ROOT_DIR / config.get('output_dir', 'outputs/quality')

report_path = save_quality_report(summary, output_dir, logging_config=log_cfg)

if summary['success']:
    logger.info("Quality checks passed successfully. Report saved at: %s", report_path)
else:
    logger.warning("Quality checks failed. %d/%d checks failed. Report saved at: %s", summary['failed_expectations'], summary['total_expectations'], report_path)
logger.info(f"Quality report saved at: {report_path}")