import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Configuración
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Carga de datos
train_data = pd.read_csv('data/training.csv')
X_features = train_data.drop(columns=['class', 'Unnamed: 0'], errors='ignore')
y_target = train_data['class']

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# PCA
pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca_transformed = pca_model.fit_transform(X_scaled)
explained_variance = pca_model.explained_variance_ratio_

# DataFrame
pca_dataframe = pd.DataFrame(data=X_pca_transformed, columns=['PC1', 'PC2'])
pca_dataframe['class'] = y_target.values

# Clases únicas y color map
unique_classes = np.unique(y_target)
color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))

# === ÚLTIMA CELDA ===
try:
    test_data = pd.read_csv('data/test.csv')
    X_test_features = test_data.drop(columns=['class', 'Unnamed: 0'], errors='ignore')

    scaler_pca = StandardScaler()
    X_train_scaled = scaler_pca.fit_transform(X_features)
    X_test_scaled = scaler_pca.transform(X_test_features)

    pca_comp = PCA(n_components=2, random_state=RANDOM_SEED)
    X_train_pca = pca_comp.fit_transform(X_train_scaled)
    X_test_pca = pca_comp.transform(X_test_scaled)

    plt.figure(figsize=(12, 8))

    for target_class, color in zip(unique_classes, color_map):
        class_mask = (y_target == target_class)
        plt.scatter(
            X_train_pca[class_mask, 0], 
            X_train_pca[class_mask, 1], 
            label=f'Train - Clase {target_class}', 
            alpha=0.5, 
            c=[color],
            s=20
        )

    plt.scatter(
        X_test_pca[:, 0], 
        X_test_pca[:, 1], 
        c='#7f7f7f', 
        label='Test (Distribución)', 
        alpha=0.3, 
        edgecolors='black', 
        linewidths=0.2, 
        s=15
    )

    plt.title('Comparativa de Distribución del Espacio de Características (PCA 2D)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({pca_comp.explained_variance_ratio_[0]:.2%} var. explicada)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca_comp.explained_variance_ratio_[1]:.2%} var. explicada)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)

    os.makedirs('data', exist_ok=True)
    plt.savefig('data/comparativa_pca_train_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("SUCCESS: Última celda ejecutada correctamente.")
except Exception as e:
    print("ERROR:", e)
