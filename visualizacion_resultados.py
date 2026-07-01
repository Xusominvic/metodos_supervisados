"""
visualizacion_resultados.py
Genera gráficos comparativos de los resultados de los 5 clasificadores
bajo ambos enfoques (Estándar vs Balanceado).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
RESULTS_DIR = 'data'
PLOTS_DIR = 'data/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Cargar resultados
df_std = pd.read_csv(os.path.join(RESULTS_DIR, 'evaluacion_estandar.csv'))
df_bal = pd.read_csv(os.path.join(RESULTS_DIR, 'evaluacion_balanceada.csv'))

# Nombres cortos para los ejes
MODELOS = ['RF', 'GB', 'AdaBoost', 'SVM', 'Bagging']
MODELOS_FULL = list(df_std['Modelo'])

# Paleta de colores
C_STD = '#1f77b4'   # Azul - Estándar
C_BAL = '#ff7f0e'   # Naranja - Balanceado

# ==============================================================================
# 1. COMPARATIVA ACCURACY: TRAIN vs CV (ambos enfoques)
# ==============================================================================
def plot_train_vs_cv():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for ax, df, title, color in [
        (axes[0], df_std, 'Enfoque Estándar (scoring=accuracy)', C_STD),
        (axes[1], df_bal, 'Enfoque Balanceado (scoring=balanced_accuracy)', C_BAL)
    ]:
        x = np.arange(len(MODELOS))
        w = 0.35
        
        bars1 = ax.bar(x - w/2, df['Train Accuracy'], w, label='Train Accuracy',
                       color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + w/2, df['CV Accuracy'], w, label='CV Accuracy',
                       color=color, alpha=0.4, edgecolor='black', linewidth=0.5)
        
        # Etiquetas de valor sobre las barras
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELOS, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Marcar la brecha de overfitting con flechas
        for i in range(len(MODELOS)):
            gap = df['Brecha Overfitting'].iloc[i]
            if gap > 0.05:  # Solo mostrar si la brecha es significativa
                ax.annotate(f'Δ={gap:.2f}',
                           xy=(x[i], df['CV Accuracy'].iloc[i]),
                           xytext=(x[i], df['CV Accuracy'].iloc[i] - 0.06),
                           fontsize=7, ha='center', color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    fig.suptitle('Brecha de Overfitting: Train Accuracy vs CV Accuracy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_train_vs_cv_accuracy.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[1/5] Gráfico de Train vs CV Accuracy generado.")

# ==============================================================================
# 2. COMPARATIVA DE MÉTRICAS: ACCURACY vs BALANCED ACCURACY vs MACRO F1
# ==============================================================================
def plot_metricas_comparativa():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for ax, df, title in [
        (axes[0], df_std, 'Enfoque Estándar'),
        (axes[1], df_bal, 'Enfoque Balanceado')
    ]:
        x = np.arange(len(MODELOS))
        w = 0.25
        
        ax.bar(x - w, df['CV Accuracy'], w, label='CV Accuracy', color='#2196F3', edgecolor='black', linewidth=0.5)
        ax.bar(x,     df['CV Balanced Accuracy'], w, label='CV Balanced Acc.', color='#FF9800', edgecolor='black', linewidth=0.5)
        ax.bar(x + w, df['CV Macro F1'], w, label='CV Macro F1', color='#4CAF50', edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELOS, fontsize=10)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel('Score', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.suptitle('Comparativa de Métricas: Accuracy vs Balanced Accuracy vs Macro F1', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_metricas_comparativa.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[2/5] Gráfico de métricas comparativas generado.")

# ==============================================================================
# 3. RECALL POR CLASE (el gráfico más importante para la discusión)
# ==============================================================================
def plot_recall_por_clase():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for ax, df, title in [
        (axes[0], df_std, 'Enfoque Estándar'),
        (axes[1], df_bal, 'Enfoque Balanceado')
    ]:
        x = np.arange(len(MODELOS))
        w = 0.25
        
        ax.bar(x - w, df['Recall Clase 0'], w, label='Recall Clase 0 (68.7%)', color='#1f77b4', edgecolor='black', linewidth=0.5)
        ax.bar(x,     df['Recall Clase 1'], w, label='Recall Clase 1 (30.7%)', color='#ff7f0e', edgecolor='black', linewidth=0.5)
        ax.bar(x + w, df['Recall Clase 2 (Minoritaria)'], w, label='Recall Clase 2 (0.6%)', color='#d62728', edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELOS, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Recall', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Línea horizontal de referencia en 0 para Clase 2
        ax.axhline(y=0.0, color='#d62728', linestyle=':', alpha=0.3)
    
    fig.suptitle('Recall por Clase: Impacto del Desbalanceo en la Detección de la Clase Minoritaria', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_recall_por_clase.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[3/5] Gráfico de Recall por clase generado.")

# ==============================================================================
# 4. COMPARATIVA ESTÁNDAR vs BALANCEADO (lado a lado por modelo)
# ==============================================================================
def plot_std_vs_bal():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(MODELOS))
    w = 0.35
    
    ax.bar(x - w/2, df_std['CV Accuracy'], w, label='Estándar (CV Accuracy)', 
           color=C_STD, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, df_bal['CV Accuracy'], w, label='Balanceado (CV Accuracy)', 
           color=C_BAL, edgecolor='black', linewidth=0.5)
    
    # Etiquetas de valor
    for i in range(len(MODELOS)):
        ax.text(x[i] - w/2, df_std['CV Accuracy'].iloc[i] + 0.008,
                f'{df_std["CV Accuracy"].iloc[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i] + w/2, df_bal['CV Accuracy'].iloc[i] + 0.008,
                f'{df_bal["CV Accuracy"].iloc[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Marcar el mejor modelo
    best_std_idx = df_std['CV Accuracy'].idxmax()
    best_bal_idx = df_bal['CV Accuracy'].idxmax()
    ax.annotate('★ Mejor', xy=(x[best_std_idx] - w/2, df_std['CV Accuracy'].iloc[best_std_idx]),
               xytext=(x[best_std_idx] - w/2, df_std['CV Accuracy'].iloc[best_std_idx] + 0.04),
               ha='center', fontsize=9, color=C_STD, fontweight='bold')
    
    ax.set_title('Comparativa Global: Enfoque Estándar vs Balanceado (CV Accuracy)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS_FULL, fontsize=10, rotation=15)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel('CV Accuracy', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_estandar_vs_balanceado.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[4/5] Gráfico de Estándar vs Balanceado generado.")

# ==============================================================================
# 5. TABLA RESUMEN DE TAMAÑOS DE MODELO Y PARÁMETROS CLAVE
# ==============================================================================
def plot_tabla_resumen():
    import ast
    
    rows = []
    for _, row in df_std.iterrows():
        params = ast.literal_eval(row['Mejores Hiperparámetros'])
        modelo = row['Modelo']
        
        # Extraer parámetros clave según el modelo
        n_est = params.get('classifier__n_estimators', '-')
        k_feat = params.get('feature_selection__k', '-')
        
        if modelo == 'Random Forest':
            depth = params.get('classifier__max_depth', 'None')
            feat = params.get('classifier__max_features', '-')
            extra = f"depth={depth}, feat={feat}"
        elif modelo == 'Gradient Boosting':
            lr = params.get('classifier__learning_rate', '-')
            depth = params.get('classifier__max_depth', '-')
            extra = f"lr={lr}, depth={depth}"
        elif modelo == 'AdaBoost':
            lr = params.get('classifier__learning_rate', '-')
            extra = f"lr={lr}, base=stump(d=1)"
        elif modelo == 'SVM (RBF)':
            C = params.get('classifier__C', '-')
            gamma = params.get('classifier__gamma', '-')
            extra = f"C={C}, γ={gamma}"
            n_est = '-'
        elif modelo == 'Bagging':
            ms = params.get('classifier__max_samples', '-')
            extra = f"max_samples={ms}"
        else:
            extra = '-'
        
        rows.append({
            'Clasificador': modelo,
            'n_estimators': str(n_est),
            'k (SelectKBest)': str(k_feat),
            'Params. Clave': extra,
            'Train Acc': f"{row['Train Accuracy']:.4f}",
            'CV Acc': f"{row['CV Accuracy']:.4f}",
            'Brecha': f"{row['Brecha Overfitting']:+.4f}"
        })
    
    df_table = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis('off')
    ax.set_title('Tabla Resumen: Mejores Hiperparámetros y Tamaño de Modelos (Enfoque Estándar)',
                fontsize=13, fontweight='bold', pad=20)
    
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df_table.columns))))
    
    # Estilizar encabezado
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Resaltar la mejor fila (mejor CV Accuracy)
    best_idx = df_std['CV Accuracy'].idxmax()
    for j in range(len(df_table.columns)):
        table[best_idx + 1, j].set_facecolor('#d4edda')
    
    plt.savefig(os.path.join(PLOTS_DIR, '05_tabla_resumen_parametros.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[5/5] Tabla resumen de parámetros generada.")


# ==============================================================================
# EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    print(">>> Generando gráficos de resultados...")
    plot_train_vs_cv()
    plot_metricas_comparativa()
    plot_recall_por_clase()
    plot_std_vs_bal()
    plot_tabla_resumen()
    print(f"\n[Éxito] Todos los gráficos guardados en: {PLOTS_DIR}/")
