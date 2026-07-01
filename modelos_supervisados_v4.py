import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Clasificadores
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ==============================================================================
# CONFIGURACIÓN GLOBAL Y SEMILLAS
# ==============================================================================
SEED = 42
np.random.seed(SEED)

TRAIN_PATH = 'data/training.csv'
TEST_PATH = 'data/test.csv'
RESULTS_DIR = 'data'

# Variables redundantes identificadas en el EDA (alta correlación entre pares)
COLS_TO_DROP = ['bpsmt', 'csuhz', 'gwrec', 'glhls', 'bqwyz']

# Validación Cruzada
CV_FOLDS = 5
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
def load_data():
    print(">>> Cargando datos de entrenamiento y prueba...")
    train_df = pd.read_csv(TRAIN_PATH, index_col=0)
    test_df = pd.read_csv(TEST_PATH, index_col=0)
    
    # Separación de características y variable objetivo
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class'].astype(int)
    
    # En test.csv, la variable 'class' está vacía (NaN)
    X_test = test_df.drop(columns=['class'])
    
    # Eliminación de variables redundantes (alta correlación detectada en EDA)
    X_train = X_train.drop(columns=COLS_TO_DROP, errors='ignore')
    X_test = X_test.drop(columns=COLS_TO_DROP, errors='ignore')
    print(f"   Variables eliminadas por redundancia: {COLS_TO_DROP}")
    
    print(f"   [Train] Dimensiones: {X_train.shape} | Clases: {np.bincount(y_train)}")
    print(f"   [Test]  Dimensiones: {X_test.shape}")
    
    # Validaciones básicas de integridad
    assert X_train.isnull().sum().sum() == 0, "Error: Hay valores nulos en el training set."
    assert X_train.shape[1] == X_test.shape[1], "Error: El número de columnas de train y test difiere."
    
    return X_train, y_train, X_test

# ==============================================================================
# 2. DEFINICIÓN DE CONFIGURACIONES DE MODELOS (DISTRITO DUAL)
# ==============================================================================
def get_models_config(balanced=False):
    """
    Retorna la configuración de pipelines y rejillas de hiperparámetros.
    Si balanced=True, aplica class_weight='balanced' en los clasificadores que lo soportan.
    """
    configs = {}
    
    # Número de features tras eliminar redundantes (50 - 5 = 45)
    n_features = 45
    k_values = [15, 25, 35, n_features]
    
    # A. RANDOM FOREST (invariante a escala → sin StandardScaler)
    rf_params = {
        'class_weight': 'balanced' if balanced else None,
        'random_state': SEED
    }
    configs['Random Forest'] = {
        'pipeline': Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', RandomForestClassifier(**rf_params))
        ]),
        'param_grid': {
            'feature_selection__k': k_values,
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20],
            'classifier__max_features': ['sqrt', 'log2']
        }
    }
    
    # B. GRADIENT BOOSTING (invariante a escala → sin StandardScaler)
    configs['Gradient Boosting'] = {
        'pipeline': Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', GradientBoostingClassifier(random_state=SEED))
        ]),
        'param_grid': {
            'feature_selection__k': k_values,
            'classifier__n_estimators': [100, 150, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5]
        }
    }
    
    # C. ADABOOST (invariante a escala → sin StandardScaler)
    configs['AdaBoost'] = {
        'pipeline': Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                random_state=SEED
            ))
        ]),
        'param_grid': {
            'feature_selection__k': k_values,
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.05, 0.1, 1.0]
        }
    }
    
    # D. SUPPORT VECTOR MACHINE (sensible a escala → StandardScaler obligatorio)
    svm_params = {
        'kernel': 'rbf',
        'probability': True,
        'class_weight': 'balanced' if balanced else None,
        'random_state': SEED
    }
    configs['SVM (RBF)'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', SVC(**svm_params))
        ]),
        'param_grid': {
            'feature_selection__k': k_values,
            'classifier__C': [1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1]
        }
    }
    
    # E. BAGGING (invariante a escala → sin StandardScaler)
    tree_params = {'class_weight': 'balanced'} if balanced else {}
    configs['Bagging'] = {
        'pipeline': Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', BaggingClassifier(
                estimator=DecisionTreeClassifier(**tree_params),
                random_state=SEED
            ))
        ]),
        'param_grid': {
            'feature_selection__k': k_values,
            'classifier__n_estimators': [100, 150, 200],
            'classifier__max_samples': [0.7, 0.8, 1.0]
        }
    }
    
    return configs

# ==============================================================================
# 3. PROCESAMIENTO E INTERROGACIÓN DE REJILLAS (GRID SEARCH)
# ==============================================================================
def run_grid_searches(X, y, balanced=False):
    """
    Ejecuta GridSearchCV para cada modelo.
    balanced=False -> scoring='accuracy'
    balanced=True -> scoring='balanced_accuracy'
    """
    configs = get_models_config(balanced=balanced)
    scorer = 'balanced_accuracy' if balanced else 'accuracy'
    print(f"\n>>> Iniciando GridSearchCV (Modo Balanceado: {balanced} | Optimización por: {scorer})")
    
    results = {}
    for name, conf in configs.items():
        print(f"   Optimizando {name}...")
        grid = GridSearchCV(
            estimator=conf['pipeline'],
            param_grid=conf['param_grid'],
            cv=cv_strategy,
            scoring=scorer,
            n_jobs=-1,
            refit=True
        )
        
        start_time = time.time()
        grid.fit(X, y)
        elapsed = time.time() - start_time
        
        print(f"      [Listo] Mejor puntuación CV ({scorer}): {grid.best_score_:.4f} | Tiempo: {elapsed:.1f}s")
        print(f"      Mejores hiperparámetros: {grid.best_params_}")
        
        results[name] = {
            'grid_search': grid,
            'best_params': grid.best_params_,
            'best_score_cv': grid.best_score_
        }
    return results

# ==============================================================================
# 4. EVALUACIÓN Y CÁLCULO DE MÉTRICAS FUERA DE FOLD (OOF)
# ==============================================================================
def evaluate_models(results, X, y, balanced=False):
    """
    Genera predicciones fuera de fold utilizando validación cruzada y compila
    métricas detalladas por clase, exactitud y brechas de sobreajuste.
    """
    evaluation_records = []
    
    for name, info in results.items():
        grid = info['grid_search']
        best_pipe = grid.best_estimator_
        
        # 1. Predicciones en entrenamiento (memorización)
        train_preds = best_pipe.predict(X)
        train_acc = accuracy_score(y, train_preds)
        
        # 2. Predicciones OOF (Out-Of-Fold) mediante validación cruzada
        oof_preds = cross_val_predict(
            best_pipe, X, y,
            cv=cv_strategy,
            n_jobs=-1
        )
        
        # Métricas agregadas
        cv_acc = accuracy_score(y, oof_preds)
        cv_bal_acc = balanced_accuracy_score(y, oof_preds)
        cv_f1_macro = f1_score(y, oof_preds, average='macro', zero_division=0)
        
        # Reporte detallado para extraer recalls por clase
        report = classification_report(y, oof_preds, output_dict=True, zero_division=0)
        
        recall_c0 = report.get('0', report.get('Clase 0', {'recall': 0.0}))['recall']
        recall_c1 = report.get('1', report.get('Clase 1', {'recall': 0.0}))['recall']
        recall_c2 = report.get('2', report.get('Clase 2', {'recall': 0.0}))['recall']
        
        # Brecha de generalización
        brecha_overfit = train_acc - cv_acc
        
        # Desviación estándar del score obtenido en CV
        best_idx = grid.best_index_
        cv_std = grid.cv_results_['std_test_score'][best_idx]
        
        evaluation_records.append({
            'Modelo': name,
            'Mejores Hiperparámetros': str(grid.best_params_),
            'Train Accuracy': round(train_acc, 4),
            'CV Accuracy': round(cv_acc, 4),
            'CV Balanced Accuracy': round(cv_bal_acc, 4),
            'CV Macro F1': round(cv_f1_macro, 4),
            'Recall Clase 0': round(recall_c0, 4),
            'Recall Clase 1': round(recall_c1, 4),
            'Recall Clase 2 (Minoritaria)': round(recall_c2, 4),
            'CV Desviación Estándar': round(cv_std, 4),
            'Brecha Overfitting': round(brecha_overfit, 4),
            'best_estimator': best_pipe,
            'oof_predictions': oof_preds
        })
        
    df_eval = pd.DataFrame(evaluation_records)
    return df_eval


# ==============================================================================
# 5. PREDICCIÓN FINAL DEL CONJUNTO DE TEST VACÍO
# ==============================================================================
def predict_test(best_estimator, X_train, y_train, X_test):
    """
    Entrena el mejor modelo con TODO el conjunto de entrenamiento
    y genera predicciones sobre test.csv.
    """
    print(f"\n>>> Reentrenando el mejor modelo global sobre TODO el dataset de entrenamiento...")
    best_estimator.fit(X_train, y_train)
    y_test_pred = best_estimator.predict(X_test)
    
    # Distribución final de predicciones
    dist = pd.Series(y_test_pred).value_counts().sort_index()
    print("Distribución de las predicciones generadas sobre test.csv:")
    for cls, cnt in dist.items():
        print(f"   Clase {cls}: {cnt} muestras ({cnt / len(y_test_pred) * 100:.2f}%)")
        
    return y_test_pred

# ==============================================================================
# EJECUCIÓN ORQUESTADA (MAIN)
# ==============================================================================
if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    
    # EJECUTAR BUSCADAS DUALES
    # 1. Enfoque Estándar
    results_std = run_grid_searches(X_train, y_train, balanced=False)
    df_eval_std = evaluate_models(results_std, X_train, y_train, balanced=False)
    
    # 2. Enfoque Balanceado
    results_bal = run_grid_searches(X_train, y_train, balanced=True)
    df_eval_bal = evaluate_models(results_bal, X_train, y_train, balanced=True)
    
    # 3. Guardar tablas comparativas
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_eval_std.drop(columns=['best_estimator', 'oof_predictions']).to_csv(
        os.path.join(RESULTS_DIR, 'evaluacion_estandar.csv'), index=False
    )
    df_eval_bal.drop(columns=['best_estimator', 'oof_predictions']).to_csv(
        os.path.join(RESULTS_DIR, 'evaluacion_balanceada.csv'), index=False
    )
    
    print("\n" + "="*80)
    print("  RESULTADOS: ENFOQUE ESTÁNDAR (Optimizado para Accuracy Global)")
    print("="*80)
    print(df_eval_std[['Modelo', 'Train Accuracy', 'CV Accuracy', 'CV Balanced Accuracy', 'Recall Clase 2 (Minoritaria)']].to_string(index=False))
    
    print("\n" + "="*80)
    print("  RESULTADOS: ENFOQUE BALANCEADO (Optimizado para Balanced Accuracy / Pesos)")
    print("="*80)
    print(df_eval_bal[['Modelo', 'Train Accuracy', 'CV Accuracy', 'CV Balanced Accuracy', 'Recall Clase 2 (Minoritaria)']].to_string(index=False))
    
    # 4. Seleccionar el mejor modelo absoluto (según la métrica del test/enunciado: CV Accuracy)
    # Buscamos el mejor del enfoque estándar
    best_idx_std = df_eval_std['CV Accuracy'].idxmax()
    best_row_std = df_eval_std.loc[best_idx_std]
    
    # Buscamos el mejor del enfoque balanceado (por si acaso tuviera mayor CV accuracy, aunque no es lo habitual)
    best_idx_bal = df_eval_bal['CV Accuracy'].idxmax()
    best_row_bal = df_eval_bal.loc[best_idx_bal]
    
    print("\n>>> Selección del mejor modelo predictivo:")
    if best_row_std['CV Accuracy'] >= best_row_bal['CV Accuracy']:
        best_model_name = best_row_std['Modelo']
        best_estimator = best_row_std['best_estimator']
        best_acc = best_row_std['CV Accuracy']
        origen = "Enfoque Estándar"
    else:
        best_model_name = best_row_bal['Modelo']
        best_estimator = best_row_bal['best_estimator']
        best_acc = best_row_bal['CV Accuracy']
        origen = "Enfoque Balanceado"
        
    print(f"   Mejor Clasificador: {best_model_name} ({origen}) con CV Accuracy = {best_acc:.4f}")
    
    # 5. Generar predicciones finales del test y guardarlas en test.csv
    y_test_pred = predict_test(best_estimator, X_train, y_train, X_test)
    
    # Cargamos el test.csv original, rellenamos y guardamos
    test_original = pd.read_csv(TEST_PATH)
    test_original['class'] = y_test_pred
    
    output_path = os.path.join(RESULTS_DIR, 'test_completado.csv')
    test_original.to_csv(output_path, index=False)
    print(f"\n[Éxito] Archivo final guardado en: {output_path}")
    print(f"La columna 'class' ha sido completada con las predicciones de {best_model_name} ({origen}).")
