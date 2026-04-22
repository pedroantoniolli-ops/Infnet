import sys
import json
import time
import warnings
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import yaml
from pathlib import Path
import optuna
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, VotingRegressor

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import CustomImputer
from src.feature_reducer import FeatureReducer
from sklearn.pipeline import Pipeline as SKlearnPipeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def _run_cv(model, X: pd.DataFrame, y: pd.Series, cv: KFold) -> list[dict]:
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = m.predict(X.iloc[val_idx])
        metrics = _compute_metrics(y.iloc[val_idx].values, y_pred)
        metrics['fold'] = fold_i + 1
        fold_metrics.append(metrics)        
    return fold_metrics

def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in ['rmse', 'mae', 'r2', 'mape']:
        result[f'cv_{col}_mean'] = float(df[col].mean())
        result[f'cv_{col}_std'] = float(df[col].std())
    return result

def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    ptype = spec['type']
    if ptype == 'log_float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
    elif ptype == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']))
    elif ptype == 'int':
        return trial.suggest_int(name, int(spec['low']), int(spec['high']))
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    else:
        raise ValueError(f'Tipo de search_space desconhecido: {ptype!r}')
    
def _build_model(model_cfg: dict, extra_params: dict | None = None):
    """
    Instancia um modelo usando importlib a partir do config (module + class).

    Mescla default_params com extra_params (extra_params sobrescreve o default).
    Permite instanciar qualquer modelo sklearn-compatível sem hardcode.
    """
    module = importlib.import_module(model_cfg['module'])
    cls = getattr(module, model_cfg['class'])
    params = dict(model_cfg.get('default_params') or {})
    if extra_params:
        params.update(extra_params)
    return cls(**params)

def _build_pipeline(
        model_cfg: dict,
        model_params: dict | None,
        reducer_params: dict | None,
        pipe_cfg: dict
) -> SKlearnPipeline:
    steps = []

    # Imputação (stateful - aprende medianas só no treino)
    for imp_spec in pipe_cfg.get('imputation', []):
        step_name = f"imputer_{imp_spec['column']}".replace('/', '-')
        steps.append((
            step_name,
            CustomImputer(
                group_col=imp_spec['group_by'],
                target_col=imp_spec['column'],
            ),
        ))

    # Escalonamento (stateful - aprende μ/σ só no treino)
    # scale_cols = pipe_cfg.get('scaling', {}).get('columns', [])
    # if scale_cols:
    #     steps.append(('scaler', StandardScalerTransformer(columns=scale_cols)))

    # Redução de features (opcional, tunable pelo Optuna)
    reducer_kw = reducer_params or {}
    steps.append(('reducer', FeatureReducer(**reducer_kw)))

    # Estimador final
    steps.append(('estimator', _build_model(model_cfg, model_params)))

    return SKlearnPipeline(steps)

def _get_feature_importance(
    model, feature_names: list[str],
    X_val: pd.DataFrame, y_val: pd.Series,
) -> pd.Series:
    """
    Extrai importância de features do modelo treinado.

    Suporta sklearn Pipeline: extrai o estimador final via named_steps['estimator']
    e usa os nomes de features pós-redução de named_steps['reducer'] quando
    disponível.

    Prioridade:
    1. feature_importances_ (árvores, ensembles baseados em árvores)
    2. coef_ (modelos lineares - usa valor absoluto)
    3. permutation_importance (fallback model-agnóstico: SVR, KNN, ensembles mistos)
    """
    # Desempacota Pipeline para obter estimador + nomes de features pós-redução
    if isinstance(model, SklearnPipeline):
        estimator = model.named_steps['estimator']
        reducer = model.named_steps.get('reducer')
        if reducer is not None and reducer.selected_features is not None:
            # RFE mantém nomes originais; PCA/kPCA usa 'pc_0', 'pc_1', ...
            imp_feature_names = reducer.selected_features
        else:
            imp_feature_names = feature_names
    else:
        estimator = model
        imp_feature_names = feature_names

    if hasattr(estimator, 'feature_importances_'):
        return pd.Series(estimator.feature_importances_, index=imp_feature_names)
    elif hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_)
        if coef.ndim > 1:
            coef = coef.flatten()
        return pd.Series(coef, index=imp_feature_names)
    else:
        # Permutation importance sobre o PIPELINE completo (inclui transformações)
        # - usa X_val original para que o pipeline transforme consistentemente
        sample_size = min(2000, len(X_val))
        idx = np.random.default_rng(42).choice(len(X_val), sample_size, replace=False)
        r = permutation_importance(
            model, X_val.iloc[idx], y_val.iloc[idx],
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        return pd.Series(r.importances_mean, index=feature_names)
    
# 1. Configuração geral do pipeline
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
modeling = load_yaml(CONFIG_DIR / 'modeling.yaml')
config.update(modeling)

# Cria o logger
log_cfg = config.get('logging')
logger = get_logger("Modelagem")

logger.info('=== Experimentação e Modelagem — MLflow + Optuna ===')
logger.info('Config carregada: pipeline.yaml + modeling.yaml')

modeling_cfg = config.get('modeling', {})
tracking_uri = modeling_cfg.get('tracking_uri', 'mlruns')
experiment_name = modeling_cfg.get('experiment_name', 'california-housing-experiments')
SEED = modeling_cfg.get('random_seed', 42)
pipe_cfg = config.get('pipeline', {})
feat_red_cfg = config.get('feature_reduction', {})
optuna_cfg = config.get('optuna', {})
_global_n_trials = optuna_cfg.get('default_trials', 50)

# Path.as_uri() converte o caminho absoluto para file:///E:/... no Windows,
# evitando que o MLflow interprete a letra do drive (E:) como URI scheme.

# mlflow.set_tracking_uri((ROOT_DIR / tracking_uri).as_uri())
mlflow.set_tracking_uri("sqlite:///home/wagner/MLOps/mlflow.db")

mlflow.set_experiment(experiment_name)

# logger.info('MLFlow tracking URI    : %s', (ROOT_DIR / tracking_uri).as_uri())
logger.info('MLFlow tracking URI    : sqlite:///mlflow.db')

logger.info('MLFlow experiment      : %s', experiment_name)
logger.info('Random seed            : %d', SEED)

cv_cfg = config.get('cv', {})
print(f"Configuraçao: {cv_cfg}")
holdout_cfg = config.get('holdout', {})
models_cfg = config.get('models', {})
ensembles_cfg = config.get('ensembles', {})
artifacts_cfg = config.get('artifacts', {})

logger.info('CV              : %s (%d folds)', cv_cfg.get('strategy'), cv_cfg.get('n_splits'))
logger.info('Holdout         : %.0f%%', holdout_cfg.get('test_size', 0.2) * 100)
logger.info('Modelos         : %d configurados', len(models_cfg))
enabled_models = [k for k, v in models_cfg.items() if v.get('enabled', True)]
logger.info('  Habilitados   : %s', enabled_models)
logger.info('Imputadores     : %d step(s)', len(pipe_cfg.get('imputation', [])))
logger.info('Scaling cols    : %d', len(pipe_cfg.get('scaling', {}).get('columns', [])))
logger.info('Feature reducer : method=%s', feat_red_cfg.get('method', 'none'))

features_dir = ROOT_DIR / config.get('paths', {}).get('features_data_dir', 'data/features')
features_file = features_dir / config.get('paths', {}).get('features_filename', 'house_price_features.parquet')

logger.info('-' * 100)
logger.info('SEÇÃO 10: Carregar Features')
logger.info('Lendo: %s', features_file)

if not features_file.exists():
    raise FileNotFoundError(
        f"Arquivo de features não encontrado: {features_file}\n"
        "Execute preprocessamento_walkthrough.py antes deste script."
    )

# Inspeciona schema sem carregar dados (leitura de metadados é barata)
schema = pq.read_schema(str(features_file))
logger.info('Schema (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-35s %s', field.name, field.type)

# %% 
# Carrega o DataFrame completo
df = pq.read_table(str(features_file)).to_pandas()
logger.info('Shape: %s', df.shape)

df = pq.read_table(str(features_file)).to_pandas()
logger.info('Shape: %s', df.shape)

# Separa features (X) e target (y)
# target vem do config de seleção de features do preprocessing.yaml
sel_cfg = config.get('feature_selection', {})
target_col = sel_cfg.get('target', 'median_house_value')

feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col]

# XGBoost rejeita nomes de colunas com '[', ']' ou '<' (ex: op_<1H OCEAN).
# Sanitizamos aqui, uma vez, para que todos os modelos downstream usem nomes limpos.
_rename_map = {
    c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
    for c in X.columns
    if any(ch in c for ch in ('<', '[', ']'))
}
if _rename_map:
    X = X.rename(columns=_rename_map)
    logger.info('Colunas renomeadas para compatibilidade com XGBoost: %s', _rename_map)

logger.info('Features: %d colunas', len(feature_cols))
logger.info('Target: %s (min=%.0f, max=%.0f, média=%.0f)',
            target_col, y.min(), y.max(), y.mean())

logger.info(y.describe())

n_nulls = X.isna().sum().sum()
if n_nulls > 0:
    logger.warning('ATENÇAO: %d valores nulos encontrados nas features!', n_nulls)
else:
    logger.info('Sem valores nulos nas features ✔')

# SEÇÃO 2 – Divisão Treino / Holdout
#
# O holdout é separado ANTES de qualquer treino, tuning ou seleção de modelo.
# É o “cofre selado” – nunca será visto até a avaliação final (Seção 9).
#
# Estratificação por quantis do target:
# • Garante que treino e holdout têm distribuições similares do target
# • Evita que toda a cauda superior (imóveis caros) caia em um único set

n_bins = holdout_cfg.get('stratify_bins', 10)
test_size = holdout_cfg.get('test_size', 0.2)

# Cria bins de quantis do target para estratificação
y_bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

logger.info('-' * 60)
logger.info('SEÇÃO 2: Divisão Treino / Holdout')
logger.info('Test size: %.0f%%  Bins de estratificação: %d', test_size * 100, n_bins)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=test_size,
    random_state=SEED,
    stratify=y_bins,
)

logger.info('Treino : %d amostras (%.1f%%)', len(X_train), 100 * len(X_train) / len(X))
logger.info('Holdout : %d amostras (%.1f%%)', len(X_holdout), 100 * len(X_holdout) / len(X))
logger.info('Target no treino — média: %.0f | std: %.0f', y_train.mean(), y_train.std())
logger.info('Target no holdout — média: %.0f | std: %.0f', y_holdout.mean(), y_holdout.std())

# Verifica que o holdout reflete a distribuição original (sem grandes desvios)
logger.info('Distribuição por quantil (treino vs holdout):')
train_dist = pd.qcut(y_train, q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
holdout_dist = pd.qcut(y_holdout, q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
dist_df = pd.DataFrame({'treino': train_dist, 'holdout': holdout_dist})
logger.info('\n%s', dist_df.round(3).to_string())

# SEÇÃO 3 – Registro de Modelos e Configuração do Pipeline
#
# Os modelos são instanciados dinamicamente via importlib:
#   module: "sklearn.linear_model"  +  class: "Ridge"  → from sklearn.linear_model import Ridge
#
# Cada modelo é envolvido num sklearn Pipeline (_build_pipeline) que inclui:
#   1. GroupMedianImputer(s)       – imputa bedrooms_per_room por grupo
#   2. StandardScalerTransformer   – z-score nas colunas contínuas
#   3. FeatureReducer              – none | rfe | pca | kpca (config: feature_reduction)
#   4. estimador                   – o modelo em si

logger.info('_' * 60)
logger.info('SEÇÃO 3: Registro de Modelos e Configuração do Pipeline')

rows = []
for name, cfg in models_cfg.items():
    enabled = cfg.get('enabled', True)
    rows.append({
        'modelo'         : name,
        'habilitado'     : '√' if enabled else 'X',
        'classe'         : f"{cfg['module']}.{cfg['class']}",
        'optuna_trials'  : cfg.get('optuna_trials', _global_n_trials),
        'params_default' : len(cfg.get('default_params') or {}),
        'search_space'   : len(cfg.get('search_space') or {}),
    })

models_table = pd.DataFrame(rows)
logger.info('\n%s', models_table.to_string(index=False))

# Lê configuração da redução de features e prepara os parâmetros padrão
_red_method = feat_red_cfg.get('method', 'none')
_red_method_cfg = feat_red_cfg.get(_red_method, {})

def _default_reducer_params() -> dict:
    """
    Constrói o dict de parâmetros para FeatureReducer a partir do método ativo
    e seus valores padrão em modeling.yaml (feature_reduction.<method>).
    """
    params = {'method': _red_method}
    if _red_method == 'rfe':
        params['n_features_to_select'] = _red_method_cfg.get('n_features_to_select', 15)
        params['rfe_estimator'] = _red_method_cfg.get('rfe_estimator', 'ridge')
    elif _red_method == 'pca':
        params['n_components'] = _red_method_cfg.get('n_components', 15)
    elif _red_method == 'kpca':
        params['n_components'] = _red_method_cfg.get('n_components', 15)
        params['kernel'] = _red_method_cfg.get('kernel', 'rbf')
        params['gamma'] = _red_method_cfg.get('gamma', None)
        params['degree'] = _red_method_cfg.get('degree', 3)
        params['coef0'] = _red_method_cfg.get('coef0', 1.0)
    return params

logger.info('Feature reducer: method=%s params=%s', _red_method, _default_reducer_params())

# SEÇÃO 4 – Baseline: Cross-Validation com Parâmetros Padrão
#
# Para cada modelo habilitado:
# 1. Instancia com os parâmetros padrão do YAML
# 2. Executa KFold CV sobre os dados de treino
# 3. Registra no MLFlow:
#    • params: parâmetros padrão usados
#    • metrics: RMSE/MAE/R2/MAPE por fold (step = índice do fold)
#    • metrics: médias e desvios padrão agregados
#
# Por que fazer baseline ANTES do Optuna?
# • Serve como referência: o Optuna deve sempre superar o baseline
# • Identifica modelos claramente inadequados (ex: Linear para dados não-lineares)
# • Permite comparar o ganho real da otimizaçao de hiperparâmetros

n_splits = cv_cfg.get('n_splits', 5)
shuffle = cv_cfg.get('shuffle', True)

cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED)
logger.info('%' * 100)
logger.info('SEÇAO 4: Baseline CV (%s, %d folds)', cv_cfg.get('strategy', 'kfold'), n_splits)

all_results: dict[str, dict] = {}

# Loop de baseline — um MLFlow run por modelo
for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        logger.info('[SKIP] %s (desabilitado)', model_name)
        continue

    logger.info('[BASELINE] %-25s ...', model_name)
    pipeline = _build_pipeline(
        model_cfg   = model_cfg,
        model_params = None,
        reducer_params = _default_reducer_params(),
        pipe_cfg    = pipe_cfg,
    )
    t0 = time.time()

    with mlflow.start_run(
        run_name=f'baseline_{model_name}',
        tags={'stage': 'baseline', 'model': model_name}
    ):
        default_params = {
            str(k): (str(v) if v is None else v)
            for k, v in (model_cfg.get('default_params') or {}).items()
        }
        default_params['reducer_method'] = _red_method
        mlflow.log_params(default_params)
        mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")
        mlflow.set_tag('reducer_method', _red_method)

        fold_metrics = _run_cv(pipeline, X_train, y_train, cv)
        
        for fm in fold_metrics:
            step = fm['fold']
            mlflow.log_metric("fold_rmse", fm["rmse"], step=step)
            mlflow.log_metric("fold_mae",  fm["mae"],  step=step)
            mlflow.log_metric("fold_r2",   fm["r2"],   step=step)
            mlflow.log_metric("fold_mape", fm["mape"], step=step)

        # Agrega métricas
        agg = _aggregate_fold_metrics(fold_metrics)
        mlflow.log_metrics(agg)

        # Tempo de treino
        mlflow.log_metric("training_time_s", time.time() - t0)

        # mlflow.sklearn.log_model (
        #     sk_model=pipeline,
        #     name=model_name,
        #     registered_model_name="california-housing-model"
        # )

        # Guarda resultados
        all_results[model_name] = {
            **agg,
            "fold_metrics": fold_metrics,
            'model_cfg'     : model_cfg,
            'best_params'   : dict(model_cfg.get('default_params') or {}),
            'reducer_params': _default_reducer_params(),
            'tuned'         : False,
        }

        logger.info(
            '    CV RMSE: %8.2f ± %6.2f    |    R²: %.4f    |    %.1fs',
            agg['cv_rmse_mean'], agg['cv_rmse_std'], agg['cv_r2_mean'], time.time() - t0,
        )

baseline_df = pd.DataFrame([
    {
        'modelo'       : k,
        'cv_rmse_mean' : v['cv_rmse_mean'],
        'cv_rmse_std'  : v['cv_rmse_std'],
        'cv_mae_mean'  : v['cv_mae_mean'],
        'cv_r2_mean'   : v['cv_r2_mean']
    }
    for k, v in all_results.items()
]).sort_values('cv_rmse_mean')

logger.info('\n%s', baseline_df.round(2).to_string(index=False))

# SEÇÃO 5 - Otimização de Hiperparâmetros com Optuna
# 
# Para cada modelo com optuna_trials > 1 e search_space configurado:
# 1. Abre um MLFlow run "pai" para o modelo
# 2. Define uma função objetivo que:
#     a. Sugere hiperparâmetros via Optuna (Bayesian/TPE por padrão)
#     b. Instancia o modelo com os parâmetros sugeridos
#     c. Executa KFold CV e calcula RMSE médio
#     d. Abre um MLFlow run "filho" (aninhado) e registra params + métricas
# 3. Optuna minimiza o RMSE médio de CV ao longo dos trials
# 4. Plots do Optuna (histórico + importância de parâmetros) são logados como artefatos no run pai
# 
# Estrutura no MLFlow:
# baseline_<modelo>      (stage=baseline)
# optuna_<modelo>        (stage=optuna)  ← run pai
# trial_0                (stage=trial)   ← run filho
# trial_1
# ...

output_dir = ROOT_DIR / artifacts_cfg.get('output_dir', 'outputs/modeling')
output_dir.mkdir(parents=True, exist_ok=True)

logger.info('*' * 60)
logger.info('SEÇÃO 5: Otimização com Optuna')
logger.info('Diretório de artefatos: %s', output_dir)

for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        continue
    n_trials = model_cfg.get('optuna_trials', _global_n_trials)
    search_space = model_cfg.get('search_space') or {}

    # Modelos sem search_space (ex: LinearRegression) usam apenas o baseline
    if n_trials <= 1 or not search_space:
        logger.info('[ SKIP OPTUNA] %-20s - sem hiperparâmetros', model_name)
        continue

    logger.info('[ OPTUNA] %-22s (%d trials) ...', model_name, n_trials)
    t0 = time.time()

    # Subsampling para SVR (O(n^2) - muito lento sem isso)
    max_s = model_cfg.get('max_samples_for_tuning')
    if max_s and len(X_train) > max_s:
        rng = np.random.default_rng(SEED)
        tune_idx = rng.choice(len(X_train), max_s, replace=False)
        X_tune = X_train.iloc[tune_idx]
        y_tune = y_train.iloc[tune_idx]
        logger.info(' SVR: subsampling %d -> %d amostras para tuning', len(X_train), max_s)
    else:
        X_tune = X_train
        y_tune = y_train

    with mlflow.start_run(
        run_name=f'optuna_{model_name}',
        tags={'stage': 'optuna', 'model': model_name}
    ) as parent_run:
        # Espaço de busca global do reducer
        _red_global_ss = feat_red_cfg.get('search_space', {})

        # Funçao objetivo do Optuna
        def _objective(trial: optuna.Trial) -> float:
            params = {
                name: _suggest_param(trial, name, spec)
                for name, spec in search_space.items()
            }

            # Sugere o método de reduçao
            if 'method' in _red_global_ss:
                trial_method = _suggest_param(trial, 'reducer_method', _red_global_ss['method'])
            else:
                trial_method = _red_method
            
            # Sugere parâmetro específicos do método escolhido neste trial
            reducer_trial_params: dict = {'method': trial_method}
            method_ss = feat_red_cfg.get(trial_method, {}).get('search_space', {})
            for r_name, r_spec in method_ss.items():
                reducer_trial_params[r_name] = _suggest_param(
                    trial, f'reducer_{r_name}', r_spec
                )

            method_defaults = {
                k: v for k, v in feat_red_cfg.get(trial_method, {}).items()
                if k != 'search_space'
            }
            for k, v in method_defaults.items():
                if k not in reducer_trial_params:
                    reducer_trial_params[k] = v

            pipeline_trial = _build_pipeline(
                model_cfg = model_cfg,
                model_params = params,
                reducer_params = reducer_trial_params,
                pipe_cfg = pipe_cfg
            )

            # Executa CV
            fold_mets = _run_cv(pipeline_trial, X_tune, y_tune, cv)
            agg_mets = _aggregate_fold_metrics(fold_mets)

            # Registra trial como um run aninhado no MLFlow
            all_log_params = {k: (str(v) if v is None else v) for k, v in params.items()}
            all_log_params.update({
                f'reducer_{k}': (str(v) if v is None else v)
                for k, v in reducer_trial_params.items()
            })
            with mlflow.start_run(
                run_name=f'trial_{trial.number}',
                nested=True,
                tags= {'stage': 'trial', 'model': model_name, 'trial': str(trial.number)}
            ):
                mlflow.log_params(all_log_params)
                mlflow.log_metrics({
                    'cv_rmse_mean': agg_mets['cv_rmse_mean'],
                    'cv_mae_mean': agg_mets['cv_mae_mean'],
                    'cv_r2_mean': agg_mets['cv_r2_mean'],
                    'cv_mape_mean': agg_mets['cv_mape_mean']
                })

            return agg_mets['cv_rmse_mean']
        
        # Executa o estudo Optuna (TRE sampler por padrao)
        study = optuna.create_study(direction='minimize', study_name=model_name)
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

        # Verifica algum trial com sucesso
        try:
            _ = study.best_value
        except ValueError:
            logger.warning('    [SKIP OPTUNA] %s - Nenhum trial bem-sucedido', model_name)
            continue

        # Loga o melhor resultado no run pai
        best_log_params = {
            f'best_{k}': (str(v) if v is None else v)
            for k, v in study.best_params.items()
        }
        mlflow.log_params(best_log_params)
        mlflow.log_metrics({
            'best_cv_rmse': study.best_value,
            'n_trials': len(study.trials)
        })

        # Salva e loga plots do Optuna (histórico de otimizaçao)
        if len(study.trials) > 1:
            try:
                fig_hist = optuna.visualization.matplotlib.plot_optimization_history(study)
                fig_hist.figure.set_size_inches(10, 5)
                hist_path = output_dir / f'optuna_history_{model_name}.png'
                fig_hist.figure.savefig(hist_path, dpi=120, bbox_inches='tight')
                plt.close(fig_hist.figure)
                mlflow.log_artifact(str(hist_path), artifact_path='optuna')
            except Exception as exc:
                logger.warning('    Plot optuna_history falhou para %s: %s', model_name, exc)

            try:
                fig_imp = optuna.visualization.matplotlib.plot_param_importance(study)
                fig_imp.figure.set_size_inches(10, 5)
                imp_path = output_dir / f'optuna_params_{model_name}.png'
                fig_imp.figure.savefig(imp_path, dpi=120, bbox_inches='tight')
                plt.close(fig_imp.figure)
                mlflow.log_artifact(str(imp_path), name='optuna')
            except Exception as exc:
                logger.warning('    Plot optuna_params falhou para %s: %s', model_name, exc)

    # Separa parâmetros do estimador dos parâmetros do reducer
    best_params_all = study.best_params
    best_estimator_params = {
        k: v for k, v in best_params_all.items() if not k.startswith('reducer_')
    }

    # Reconstrói reducer params: extrai method + params específicos do trial
    trial_method = best_params_all.get('reducer_method', _red_method)
    best_reducer_params: dict = {'method': trial_method}
    for k, v in best_params_all.items():
        if k.startswith('reducer_') and k != 'reducer_method':
            best_reducer_params[k[len('reducer_'):]] = v
    
    # Preenche parâmetros fixos do método escolhido que nao foram tunados
    method_defaults = {
        k: v for k, v in feat_red_cfg.get(trial_method, {}).items()
        if k != 'search_space'
    }

    for k, v in method_defaults.items():
        if k not in best_reducer_params:
            best_reducer_params[k] = v

    pipeline_tuned = _build_pipeline(
        model_cfg=model_cfg,
        model_params=best_estimator_params,
        reducer_params=best_reducer_params,
        pipe_cfg=pipe_cfg
    )
    fold_mets_tuned = _run_cv(pipeline_tuned, X_train, y_train, cv)
    agg_tuned = _aggregate_fold_metrics(fold_mets_tuned)

    previous_rmse = all_results[model_name]['cv_rmse_mean']
    if agg_tuned['cv_rmse_mean'] < previous_rmse:
        all_results[model_name].update({
            **agg_tuned,
            'fold_metrics': fold_mets_tuned,
            'best_params': best_estimator_params,
            'reducer_params': best_reducer_params,
            'tuned': True
        })
        logger.info(
            '   Melhoria: %.2f -> %.2f (Δ=%.2f) %.1f',
            previous_rmse, agg_tuned['cv_rmse_mean'],
            previous_rmse - agg_tuned['cv_rmse_mean'],
            time.time() - t0
        )
    else:
        logger.info(
            '   Sem melhoria: baseline %.2f mantido %.1fs',
            previous_rmse, time.time() - t0
        )

tuning_df = pd.DataFrame([
    {
        'modelo': k,
        'cv_rmse_mean': v['cv_rmse_mean'],
        'cv_rmse_std': v['cv_rmse_std'],
        'cv_r2_mean': v['cv_r2_mean'],
        'tuned': '✅' if v['tuned'] else '🚫'
    }
    for k, v in all_results.items()
]).sort_values('cv_rmse_mean')

logger.info('\n%s', tuning_df.round(2).to_string(index=False))


# SEÇÃO 6 – Ensembles: StackingRegressor e VotingRegressor

# Após otimizar todos os modelos individuais, selecionamos os TOP-N por RMSE.
# Ensembles combinam as previsões desses modelos para reduzir variância.

# StackingRegressor:
# • Usa out-of-fold predictions dos modelos base como features para um meta-learner
# • O meta-learner aprende a combinar as previsões otimamente
# • Optuna busca o melhor alpha do Ridge (meta-learner) + flag passthrough
# • passthrough=True inclui as features originais junto às predições dos base

# VotingRegressor:
# • Média ponderada das previsções dos modelos base
# • Optuna busca os pesos ótimos (inteiros) para cada modelo base
# • Mais simples e interpretável que Stacking; frequentemente comparável

# %%
top_n = ensembles_cfg.get('top_n_base_models', 3)

# Seleciona top-N modelos por RMSE médio de CV
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['cv_rmse_mean'])
top_n_entries = sorted_results[:top_n]
top_n_names = [name for name, _ in top_n_entries]

logger.info('-' * 60)
logger.info('SEÇÃO 6: Ensembles – top-%d modelos base: %s', top_n, top_n_names)
for name, res in top_n_entries:
    logger.info('%-25s CV RMSE: %.2f', name, res['cv_rmse_mean'])

def _make_top_n_estimators(top_n_entries: list[tuple]) -> list[tuple]:
    """Cria pipelines nao treinados dos top-N modelos com os melhores params."""
    return [
        (name, _build_pipeline(
            model_cfg=result['model_cfg'],
            model_params=result['best_params'],
            reducer_params=result.get('reducer_params', _default_reducer_params()),
            pipe_cfg=pipe_cfg
        ))
        for name, result in top_n_entries
    ]

stacking_cfg = ensembles_cfg.get('stacking', {})

if stacking_cfg.get('enabled', True):
    n_stacking_trials=stacking_cfg.get('optuna_trials', _global_n_trials)
    inner_cv = stacking_cfg.get('inner_cv_folds', 3)
    stacking_ss = stacking_cfg.get('meta_learner_search_space', {})

    logger.info('   [OPTUNA] stacking (%d trials) ...', n_stacking_trials)
    t0_s = time.time()

    with mlflow.start_run(
        run_name='optuna_stacking',
        tags={'stage': 'optuna', 'model': 'stacking', 'base_models': str(top_n_names)}
    ):
        def _stacking_objective(trial: optuna.Trial) -> float:
            meta_alpha = trial.suggest_float('meta_alpha', 1e-4, 1e4, log=True)
            stacking = StackingRegressor(
                estimators=_make_top_n_estimators(top_n_entries),
                final_estimator=Ridge(alpha=meta_alpha),
                passthrough=False,
                cv=inner_cv,
                n_jobs=1
            )
            fold_mets = _run_cv(stacking, X_train, y_train, cv)
            agg_mets = _aggregate_fold_metrics(fold_mets)

            with mlflow.start_run(
                run_name=f'stacking_trial_{trial.number}',
                nested=True,
                tags={'stage': 'stacking_trial'}

            ):
                mlflow.log_params({'meta_alpha': meta_alpha, 'passthrough': False})
                mlflow.log_metrics({
                    'cv_rmse_mean': agg_mets['cv_rmse_mean'],
                    'cv_r2_mean': agg_mets['cv_r2_mean']
                })
            return agg_mets['cv_rmse_mean']
        
        stacking_study = optuna.create_study(direction='minimize', study_name='stacking')
        stacking_study.optimize(_stacking_objective, n_trials=n_stacking_trials, catch=(Exception,))

        try:
            mlflow.log_params({f'best_{k}': v for k, v in stacking_study.best_params.items()})
            mlflow.log_metrics('best_cv_rmse', stacking_study.best_value)
        except ValueError:
            logger.warning('    [SKIP] stacking - todos os %d trials falharam', n_stacking_trials)

        mlflow.log_param('base_models', str(top_n_names))

    try:
        best_meta_alpha = stacking_study.best_params['meta_alpha']
    except ValueError:
        logger.warning('    [SKIP] stacking - sem trials bem-sucedidos, ensemble ignorado')
        best_meta_alpha = None

    if best_meta_alpha is not None:
        best_stacking = StackingRegressor(
            estimators=_make_top_n_estimators(top_n_entries),
            final_estimator=Ridge(alpha=best_meta_alpha),
            passthrough=False,
            cv=inner_cv,
            n_jobs=1
        )
        fold_mets_stacking = _run_cv(best_stacking, X_train, y_train, cv)
        agg_stacking=_aggregate_fold_metrics(fold_mets_stacking)

        all_results['stacking']= {
            **agg_stacking,
            'fold_metrics': fold_mets_stacking,
            'model_cfg': {
                'module': 'sklearn.ensemble', 'class': 'StackingRegressor', 'default_params': {}
            },
            'best_params': stacking_study.best_params,
            'tuned': True,
            '_instance': best_stacking
        }
        logger.info(
            '   Stacking -> CV_RMSE: %.2f +/- %.2f  (meta_alpha=%.4f)   %.1f',
            agg_stacking['cv_rmse_mean'], agg_stacking['cv_rmse_std'],
            best_meta_alpha, time.time() - t0_s
        )

# Voting Regressor

voting_cfg = ensembles_cfg.get('voting', {})

if voting_cfg.get('enabled', True):
    n_voting_trials = voting_cfg.get('optuna_trials', _global_n_trials)
    w_low = voting_cfg.get('weight_low', 1)
    w_high = voting_cfg.get('weight_high', 10)

    logger.info('   [OPTUNA] voting     (%d trials) ...', n_voting_trials)
    t0_v = time.time()

    with mlflow.start_run(
        run_name='optuna_voting',
        tags={'stage': 'optuna', 'model': 'voting', 'base_model': str(top_n_names)},
    ):
        def _voting_objective(trial: optuna.Trial) -> float:
            weights = [
                trial.suggest_int(f'w_{name}', w_low, w_high)
                for name in top_n_names
            ]
            voting = VotingRegressor(
                estimators=_make_top_n_estimators(top_n_entries),
                weights=weights,
                n_jobs=1
            )
            fold_mets=_run_cv(voting, X_train, y_train, cv)
            agg_mets=_aggregate_fold_metrics(fold_mets)

            with mlflow.start_run(run_name=f'voting_trial_{trial.number}', nested=True, tags={'stage': 'voting_trial'}):
                mlflow.log_params({f'w_{n}': w for n, w in zip(top_n_names, weights)})
                mlflow.log_metrics({
                    'cv_rmse_mean': agg_mets['cv_rmse_mean'],
                    'cv_r2_mean': agg_mets['cv_r2_mean']
                })
            return agg_mets['cv_rmse_mean']
        
        voting_study = optuna.create_study(direction='minimize', study_name='voting')
        voting_study.optimize(_voting_objective, n_trials=n_voting_trials, catch=(Exception,))

        try:
            mlflow.log_params({f'best_{k}': v for k, v in voting_study.best_params.items()})
            mlflow.log_metric('best_cv_rmse', voting_study.best_value)
        except ValueError:
            logger.warning('    [SKIP] voting - todos os %d trials falharam', n_voting_trials)
        mlflow.log_param('base_models', str(top_n_names))

    try:
        best_weights = [voting_study.best_params[f'w_{name}'] for name in top_n_names]
    except ValueError:
        logger.warning('    [SKIP] voting - sem trials bem-sucedidos, ensemble ignorado')
        best_weights=None

    if best_weights is not None:
        best_voting = VotingRegressor(
            estimators=_make_top_n_estimators(top_n_entries),
            weights=best_weights,
            n_jobs=1
        )
        fold_mets_voting= _run_cv(best_voting, X_train, y_train, cv)
        agg_voting = _aggregate_fold_metrics(fold_mets_voting)

        all_results['voting'] = {
            **agg_voting,
            'fold_metrics': fold_mets_voting,
            'model_cfg': {'module': 'sklearn.ensemble', 'class': 'VotingRegressor', 'default_params': {}},
            'best_params': voting_study.best_params,
            'tuned': True,
            '_instance': best_voting
        }
        logger.info(
            '   Voting -> CV RMSE: %.2f +/- %.2f (pesos=%s) %.1fs',
            agg_voting['cv_rmse_mean'], agg_voting['cv_rmse_std'],
            best_weights, time.time() - t0_v,
        )

# SEÇAO 7 - Selecao do melhor modelo

logger.info('SEÇAO 7: Seleçao do Melhor Modelo')

full_ranking = pd.DataFrame([
    {
        'modelo': k,
        'cv_rmse_mean': v['cv_rmse_mean'],
        'cv_rmse_std': v['cv_rmse_std'],
        'cv_mae_mean': v['cv_mae_mean'],
        'cv_r2_mean': v['cv_r2_mean'],
        'cv_mape_mean': v['cv_mape_mean'],
        'tuned': '✅' if v['tuned'] else '🚫'
    }
    for k, v in all_results.items()
]).sort_values(['cv_rmse_mean', 'cv_rmse_std'], ascending=[True, True])

logger.info('\n%s', full_ranking.round(2).to_string(index=False))

best_model_name = full_ranking.iloc[0]['modelo']
best_result=all_results[best_model_name]
best_params=best_result['best_params']
best_red_params=best_result.get('reducer_params', _default_reducer_params())

logger.info('Melhor modelo: %s', best_model_name)
logger.info('   CV RMSE: %.2f +/- %.2f', best_result['cv_rmse_mean'], best_result['cv_rmse_std'])
logger.info('   CV R2: %.4f', best_result['cv_r2_mean'])
logger.info('   Params: %s', best_params)
logger.info('   Reducer: %s', best_red_params)