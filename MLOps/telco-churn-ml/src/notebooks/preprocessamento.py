import sys
from pathlib import Path
import yaml
import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]
prep_yaml_path = CONFIG_DIR / "preprocessing.yaml"

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.append(_p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import (
    CustomImputer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    OceanProximityEncoder,
    FeatureSelector
)

logger = get_logger("Preprocessing")
logger.info("Loading configuration files")
logger.info("Starting Pre-processing notebook /////////////////////////////////////////////")

logger.info("Loading config files ------------------------------>")

config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
preprocessing = load_yaml(prep_yaml_path)
config.update(preprocessing)

prep_cfg = config.get('preprocessing', {})
logger.info("Preprocessing configuration loaded: %s", prep_cfg)
output_dir = ROOT_DIR / prep_cfg.get('output_dir', 'data/features')
output_path = output_dir / prep_cfg.get('output_filename', 'house_price_features.parquet')
compression = prep_cfg.get('compression', 'snappy')
logger.info("Output directory: %s", output_dir)
logger.info("Output file path: %s", output_path)
logger.info("Compression format: %s", compression)

logger.info("Imputation configurations: %d", len(config.get('imputation', [])))

processed_dir = ROOT_DIR / config['paths']['processed_data_dir']
parquet_path = processed_dir / config['paths']['output_filename']

logger.info("Reading processed data from: %s", parquet_path)

if not parquet_path.exists():
    logger.error("Processed data file not found at: %s", parquet_path)
    raise FileNotFoundError(f"Processed data file not found at: {parquet_path}")

schema = pq.read_schema(parquet_path)
logger.info("Schema of the input data: %s", schema)

for field in schema:
    logger.info("Column: %s, Type: %s", field.name, field.type)

# Loading parquet file and converting to Pandas Data Frame
df = pq.read_table(parquet_path).to_pandas()
logger.info("Data loaded successfully. Shape: %s", df.shape)

print(df.head())

def _check_missing_values(df: pd.DataFrame) -> None:
    logger.info("Missing values per column before imputation: ------------------------------->")
    logger.info("Checking for missing values in the DataFrame")
    
    total_missing = 0
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        total_missing += missing_count
        msg = (
            f"⛔ MISSING DETECTED → Column: {col}, Missing: {missing_count} 🚫"       
            if missing_count > 0
            else f"✔ OK → Column: {col}, Missing: {missing_count}"
        )
        
        logger.info(msg)
    logger.info("Missing values per column before imputation: %d <-------------------------------", total_missing)

_check_missing_values(df)

print(df.describe())

# imp_specs = config.get('imputation', [])
# logger.info("Applying imputation strategies: %d", len(imp_specs))
# for spec in imp_specs:
#     group_col = spec.get('groupby')
#     target_col = spec.get('column')
#     logger.info("Applying imputation for target column '%s' grouped by '%s'", target_col, group_col)
#     imputer = CustomImputer(group_col=group_col, target_col=target_col, logger=logger)
#     df = imputer.fit_transform(df)
# 
# _check_missing_values(df)
# 
# for grupo, mediana in imputer.medians_.items():
#     logger.info("Group: %s, Median: %s", grupo, mediana)
# logger.info("Global median used for imputation: %s", imputer.global_median_)
# 
# print(df[['total_bedrooms', 'ocean_proximity']].describe())

# Carregando as transformações de flags binárias a partir da configuração
binary_flags_config = config.get('binary_flags', [])

def _list_values_above_threshold(df: pd.DataFrame, flags_config: list, logger) -> None:
    """Lista os valores acima dos limites definidos em preprocessing.yaml para cada binary_flag."""
    logger.info("Listando valores acima dos limites definidos para binary_flags:")

    # Caso seja um dicionário, converte para lista de dicts
    if isinstance(flags_config, dict):
        flags_config = [{"column": k, "value": v} for k, v in flags_config.items()]

    logger.info("Configurações de binary_flags: %s", flags_config)

    for flag_spec in flags_config:
        col = flag_spec.get('column')
        threshold = flag_spec.get('value')
        
        if col is None:
            logger.warning(f"Configuraçao inválida: %s", flag_spec)
            continue

        if col not in df.columns:
            logger.warning("Coluna '%s' nao encontrada no DF", col)
            continue
        
        above_threshold = df[df[col] == threshold]
        count = len(above_threshold)

        logger.info(f"Registros acima do limite: {count} para colune {col}")


_list_values_above_threshold(df, binary_flags_config, logger)

logger.info("Applying binary flag transformations: %d", len(binary_flags_config))
for flag_spec in binary_flags_config:
    col = flag_spec.get('column')
    value = flag_spec.get('value')
    new_col = flag_spec.get('new_column')
    logger.info("Creating binary flag '%s' for column '%s' with threshold '%s'", new_col, col, value)
    flag_transformer = BinaryFlagTransformer(flags=[flag_spec], logger=logger)
    df = flag_transformer.fit_transform(df)

print(df)

ratio_cfg = config.get('ratio_features', [])
logger.info('-'*60)
logger.info('SEÇAO 4: Features de Razao')

ratio_transformer = RatioFeatureTransformer(ratio_cfg, logger=logger)
df = ratio_transformer.transform(df)

new_ratio_cols = [spec['name'] for spec in ratio_cfg]
df[new_ratio_cols].describe()

logger.info('Correlaçao das razoes com median_house_value:')
for col in new_ratio_cols:
    corr = df[col].corr(df['median_house_value'])
    logger.info('   %-30s r - %.3f', col, corr)

log_cols = config.get('log_transformer', {}).get('columns', [])
logger.info('-' * 100)
logger.info('SEÇAO 5: Transformaçao logarítmica (log1p)')

log_transformer = LogTransformer(log_cols, logger=logger)
df = log_transformer.transform(df)

logger.info('Comparaçao de assimetria (skewness):')
for col in log_cols:
    if col in df.columns:
        log_col = f'log_{col}'
        skew_raw = df[col].dropna().skew()
        skew_log = df[log_col].dropna().skew() if log_col in df.columns else float('nan')
        logger.info('   %-30s raw: %+.2f -> log: %+.2f', col, skew_raw, skew_log)

log_created = [f'log_{c}' for c in log_cols if f'log_{c}' in df.columns]


geo_cfg = config.get('geo_distance', {})
logger.info('%' * 60)
logger.info('SEÇAO 6: Distancias Geograficas')
logger.info('Cidades de referência: %s', [c['name'] for c in geo_cfg.get('cities', [])])

geo_transformer = GeoDistanceTransformer(geo_cfg, logger=logger)
df = geo_transformer.transform(df)

dist_cols = [f'dist_{c['name']}' for c in geo_cfg.get('cities', [])]
nearest_col = geo_cfg.get('nearest_city_column', 'nearest_city_distance')
all_dist_cols = dist_cols + [nearest_col]

print(df[all_dist_cols].describe())

logger.info('#' * 60)
logger.info('\n' * 2)
logger.info('SEÇAO 7: Funçoes Polinomiais')

pol_cfg = config.get('polynomial_features', [])

logger.info('Colunas: %s', [spec['columns'] for spec in pol_cfg])

pol_transformer = PolynomialFeatureTransformer(pol_cfg, logger=logger)
df = pol_transformer.transform(df)
print(df.describe())

enc_cfg = config.get('categorical_encoding', {})
logger.info('/' * 60)
logger.info('SEÇAO 8: Categorical Encoder')
logger.info('Mapa ordinal: %s', enc_cfg.get('ordinal_map'))

encoder = OceanProximityEncoder(enc_cfg, logger=logger)
df = encoder.transform(df)

logger.info('Distribuiçao do encoding ordinal:')
ordinal_col = enc_cfg.get('ordinal_column', 'ocean_proximity_encoded')
df[[enc_cfg.get('column', 'ocean_proximity'), ordinal_col]].value_counts().sort_index()

prefix = enc_cfg.get('one_hot_prefix', 'op')
dummy_cols = [c for c in df.columns if c.startswith(f'{prefix}_')]
logger.info('Dummies criadas: %s', dummy_cols)

logger.info('Correlaçao das dummies com median_house_value:')
for col in dummy_cols:
    corr = df[col].corr(df['median_house_value'])
    logger.info('   %-20s r = %.3f', col, corr)


feat_cfg = config.get('feature_selection', {})
features_to_keep = feat_cfg.get('features_to_keep', [])
logger.info('Features pré-selecionadas (manualmente): %s', features_to_keep)

logger.info('"' * 60)
logger.info('SEÇAO 9: Feature Selection')
logger.info('Configuraçao: %s', features_to_keep)

selector = FeatureSelector(features_to_keep, logger=logger)

df = selector.transform(df)

logger.info('Features selecionadas: %s', features_to_keep)
print(df[features_to_keep].describe())
logger.info("Saving preprocessed features to: %s", output_path)
output_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False, compression=compression)
logger.info("Preprocessed features saved successfully.")

print(df.head())