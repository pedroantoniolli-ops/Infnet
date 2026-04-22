# %%
import sys
from pathlib import Path
# %%
# Definiçoes

ROOT_DIR = Path(__file__).parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG_PATH = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG_PATH = CONFIG_DIR / 'data.yaml'

# Adiciona o caminho do diretório raiz ao sys.path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(CONFIG_DIR))

from src.ingestion import ingest_csv_to_parquet
from src.utils.logger import get_logger, logging
from src.utils.config_loader import load_yaml
from src.downloader import check_kaggle_credentials, list_remote_files, download_file_from_kaggle, _unzip_file
# %%
data_cfg = load_yaml(DATA_CONFIG_PATH)
pipeline_cfg = load_yaml(PIPELINE_CONFIG_PATH)

# %%
log_cfg = pipeline_cfg.get('logging', {})
logger = get_logger(
    name=log_cfg.get('name', 'IngestaoLogger'),
    level=getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO),
    log_to_file=log_cfg.get('log_to_file', False),
    log_dir=log_cfg.get('log_dir', 'logs'),
    log_file=log_cfg.get('log_file', 'pipeline.log')
)
# %%
# Verifica as credenciais do Kaggle

kaggle_credentials_valid = check_kaggle_credentials(SECRETS_PATH)
if not kaggle_credentials_valid:
    logger.error("Kaggle credentials are not valid. Please check your secrets.env file.")
    raise EnvironmentError("Kaggle credentials are not valid. Please check your secrets.env file.")
else:
    logger.info("Kaggle credentials are valid. Proceeding with data ingestion.")
# %%
# data discovery
dataset = data_cfg.get('kaggle').get('dataset')
logger.info(f"Dataset to be downloaded: {dataset}")
file_pattern = data_cfg.get('kaggle').get('file_pattern')
expected_files = data_cfg.get('kaggle').get('expected_files')

logger.info(f"Expected files: {expected_files}" or "(auto-detect)")
logger.info(f"File pattern: {file_pattern}")
logger.info(f"Dataset: {dataset}")

if not expected_files:
    logger.info("No expected files specified. Attempting to auto-detect files in the dataset.")
    try:
        expected_files = list_remote_files(dataset, logging_config=log_cfg, file_pattern=file_pattern)
        logger.info(f"Auto-detected {len(expected_files)} files: {expected_files}")
    except Exception as e:
        logger.error(f"Error during auto-detection of files: {e}")
        raise
# %%
# Directory for raw data
raw_data_dir = ROOT_DIR / pipeline_cfg.get('data', {}).get('raw_data_dir', 'data/raw')
raw_data_dir.mkdir(parents=True, exist_ok=True)

# Configurações de download
skip_existing = pipeline_cfg.get('execution', {}).get('skip_existing', True)
force_download = pipeline_cfg.get('execution', {}).get('force_download', False)

# Download files
downloaded = download_file_from_kaggle(
    dataset=dataset,
    expected_files=expected_files,
    output_dir=raw_data_dir,
    logging_config=log_cfg,
    skip_existing=skip_existing,
    force_download=force_download,
    secrets_path=SECRETS_PATH
)

zip_files = list(raw_data_dir.glob("*.zip"))

if zip_files:
    zip_path = zip_files[0]
    logger.info(f"ZIP encontrado: {zip_path}")
    _unzip_file(zip_path, raw_data_dir, logger)
else:
    logger.warning("Nenhum arquivo ZIP encontrado no diretório.")


for f in sorted(raw_data_dir.glob("*.csv")):
    logger.info('%s (%.1f KB)', f.name, f.stat().st_size / 1024)

# %%
# Defining Parquet output path
processed_data_dir = ROOT_DIR / pipeline_cfg.get('data', {}).get('processed_data_dir', 'data/processed')
processed_data_dir.mkdir(parents=True, exist_ok=True)
output_path = processed_data_dir / pipeline_cfg.get('data', {}).get('output_filename',
                                                                    'house_price_predictions.parquet'
                                                                    )
logger.info(f"Output path for processed data: {output_path}")

# obtaining processing configurations
compression = data_cfg.get('ingest', {}).get('compression', 'snappy')
logger.info(f"Compression method for Parquet output: {compression}")
chunk_size = data_cfg.get('ingest', {}).get('chunk_size', 100000)
logger.info(f"Chunk size for processing: {chunk_size}")
validate = data_cfg.get('ingest', {}).get('validate_schema', True)
logger.info(f"Validation enabled: {validate}")
required_columns = data_cfg.get('schema', {}).get('required_columns', [])
logger.info(f"Required columns for validation: {required_columns}")
skip_ingest_if_exists = pipeline_cfg.get('execution', {}).get('skip_ingest_if_exists', True)
logger.info(f"Skip ingestion if output file exists: {skip_ingest_if_exists}")
force_ingest = pipeline_cfg.get('execution', {}).get('force_ingest', False)
logger.info(f"Force ingestion even if output file exists: {force_ingest}")

results = ingest_csv_to_parquet(
    raw_dir=raw_data_dir,
    output_path=output_path,
    compression=compression,
    chunk_size=chunk_size,
    validate_schema=validate,
    required_columns=required_columns,
    skip_if_exists=skip_ingest_if_exists,
    force=force_ingest,
    logging_config=log_cfg
)

logger.info(f"Ingestion completed. Output file: {results}")