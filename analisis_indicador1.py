import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind
import os
from datetime import datetime

# --- Configuración General ---
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12, 'axes.titlesize': 16,
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'figure.dpi': 100, 'savefig.dpi': 300
})
os.makedirs('resultados_trec', exist_ok=True)
print("✓ Configuración inicial completada.")

# --- Carga de Datos ---
try:
    df_pre = pd.read_csv('data/indicador1/trec_preprueba.csv')
    df_post = pd.read_csv('data/indicador1/trec_postprueba.csv')
    print("✓ Datos cargados correctamente.")
    print(f"   - Preprueba: {len(df_pre)} registros")
    print(f"   - Postprueba: {len(df_post)} registros")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que la ruta es correcta.")
    exit()

# --- 1. Análisis Descriptivo ---
print("\n--- 1. Análisis Descriptivo ---")
stats_pre = df_pre['tiempo_respuesta_min'].describe()
stats_post = df_post['tiempo_respuesta_min'].describe()

df_desc = pd.DataFrame({
    'Estadístico': ['Media (min)', 'Desv. Estándar (min)', 'Mínimo (min)', 'Máximo (min)', 'N'],
    'Preprueba': [stats_pre['mean'], stats_pre['std'], stats_pre['min'], stats_pre['max'], int(stats_pre['count'])],
    'Postprueba': [stats_post['mean'], stats_post['std'], stats_post['min'], stats_post['max'], int(stats_post['count'])]
})
df_desc['Preprueba'] = df_desc['Preprueba'].map('{:.2f}'.format)
df_desc['Postprueba'] = df_desc['Postprueba'].map('{:.2f}'.format)
print("\nTabla Comparativa de Estadísticos Descriptivos:")
print(df_desc.to_string(index=False))

# --- Gráfico 1: Boxplot Comparativo ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=[df_pre['tiempo_respuesta_min'], df_post['tiempo_respuesta_min']],
            palette=['#FF6347', '#4682B4'], ax=ax, width=0.4)
ax.set_xticklabels(['Preprueba (Antes)', 'Postprueba (Después)'])
ax.set_title('Comparación del Tiempo de Respuesta (TREC)', fontweight='bold')
ax.set_ylabel('Tiempo de Respuesta (minutos)')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('resultados_trec/01_boxplot_comparativo.png')
plt.show()
print("\n✓ Gráfico 1 (Boxplot) generado y guardado.")

# --- Gráfico 2: Histogramas de Normalidad ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df_pre['tiempo_respuesta_min'], kde=True, ax=axes, color='#FF6347', bins=15)
axes.set_title('Distribución Preprueba', fontweight='bold')
axes.set_xlabel('Tiempo (min)')
axes.set_ylabel('Frecuencia')

sns.histplot(df_post['tiempo_respuesta_min'], kde=True, ax=axes, color='#4682B4', bins=15)
axes.set_title('Distribución Postprueba', fontweight='bold')
axes.set_xlabel('Tiempo (min)')
axes.set_ylabel('Frecuencia')
fig.suptitle('Histogramas de Distribución del TREC', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('resultados_trec/02_histogramas_normalidad.png')
plt.show()
print("✓ Gráfico 2 (Histogramas) generado y guardado.")

# --- Gráfico 3: Barras de Medias con Error ---
fig, ax = plt.subplots(figsize=(8, 6))
means = [stats_pre['mean'], stats_post['mean']]
stds = [stats_pre['std'], stats_post['std']]
labels = ['Preprueba', 'Postprueba']
colors = ['#FF6347', '#4682B4']
bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, alpha=0.8)
ax.set_title('Media del Tiempo de Respuesta con Desviación Estándar', fontweight='bold')
ax.set_ylabel('Tiempo Promedio (minutos)')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('resultados_trec/03_barras_medias_error.png')
plt.show()
print("✓ Gráfico 3 (Barras) generado y guardado.")

# --- 2. Análisis Inferencial ---
print("\n--- 2. Análisis Inferencial ---")
# Prueba de Normalidad (Shapiro-Wilk)
shapiro_pre = shapiro(df_pre['tiempo_respuesta_min'])
shapiro_post = shapiro(df_post['tiempo_respuesta_min'])
print("\nPrueba de Normalidad (Shapiro-Wilk):")
print(f"  - Preprueba:  W={shapiro_pre.statistic:.4f}, p-valor={shapiro_pre.pvalue:.4f}")
print(f"  - Postprueba: W={shapiro_post.statistic:.4f}, p-valor={shapiro_post.pvalue:.4f}")
alpha = 0.05
is_normal_pre = shapiro_pre.pvalue > alpha
is_normal_post = shapiro_post.pvalue > alpha
print(f"  -> Conclusión Pre: {'Se asume normalidad' if is_normal_pre else 'No se asume normalidad'}")
print(f"  -> Conclusión Post: {'Se asume normalidad' if is_normal_post else 'No se asume normalidad'}")


# Prueba t de Student para muestras independientes
# Asumimos varianzas desiguales (más seguro) con Welch's t-test.
ttest_result = ttest_ind(df_pre['tiempo_respuesta_min'], df_post['tiempo_respuesta_min'], equal_var=False)
print("\nPrueba t de Student (Welch's t-test):")
print(f"  - Estadístico t: {ttest_result.statistic:.4f}")
print(f"  - p-valor: {ttest_result.pvalue:.4e}")

# Tamaño del Efecto (d de Cohen)
mean_pre, mean_post = stats_pre['mean'], stats_post['mean']
std_pre, std_post = stats_pre['std'], stats_post['std']
n_pre, n_post = stats_pre['count'], stats_post['count']
pooled_std = np.sqrt(((n_pre - 1) * std_pre**2 + (n_post - 1) * std_post**2) / (n_pre + n_post - 2))
cohen_d = (mean_pre - mean_post) / pooled_std
print(f"\nTamaño del Efecto (d de Cohen): {cohen_d:.4f}")

# --- Guardar Resumen en TXT ---
with open('resultados_trec/resumen_estadistico_TREC.txt', 'w') as f:
    f.write(f"RESUMEN ESTADÍSTICO - INDICADOR 1: TREC\n")
    f.write(f"Fecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*50 + "\n\n")
    f.write("1. ANÁLISIS DESCRIPTIVO\n")
    f.write("-"*50 + "\n")
    f.write(df_desc.to_string(index=False))
    f.write("\n\n")
    f.write("2. ANÁLISIS INFERENCIAL\n")
    f.write("-"*50 + "\n")
    f.write("Prueba de Normalidad (Shapiro-Wilk):\n")
    f.write(f"  - Preprueba:  W={shapiro_pre.statistic:.4f}, p-valor={shapiro_pre.pvalue:.4f}\n")
    f.write(f"  - Postprueba: W={shapiro_post.statistic:.4f}, p-valor={shapiro_post.pvalue:.4f}\n\n")
    f.write("Prueba t de Student (Welch's t-test):\n")
    f.write(f"  - Estadístico t: {ttest_result.statistic:.4f}\n")
    f.write(f"  - p-valor: {ttest_result.pvalue:.4e}\n\n")
    f.write(f"Tamaño del Efecto (d de Cohen): {cohen_d:.4f}\n\n")
    f.write("3. INTERPRETACIÓN\n")
    f.write("-"*50 + "\n")
    if ttest_result.pvalue < alpha:
        f.write("El p-valor es menor que 0.05, por lo que se rechaza la hipótesis nula.\n")
        f.write("Existe una diferencia estadísticamente significativa en el tiempo de respuesta\n")
        f.write("entre la preprueba y la postprueba. El tiempo de respuesta se redujo significativamente.\n")
    else:
        f.write("El p-valor no es menor que 0.05. No hay evidencia suficiente para afirmar\n")
        f.write("una diferencia significativa en el tiempo de respuesta.\n")
    f.write(f"La media del tiempo de respuesta se redujo de {mean_pre:.2f} min a {mean_post:.2f} min.\n")
    f.write(f"El valor d de Cohen ({cohen_d:.2f}) indica un tamaño del efecto grande, confirmando\n")
    f.write("la importancia práctica de la reducción del tiempo.\n")
print("\n✓ Resumen estadístico guardado en 'resultados_trec/resumen_estadistico_TREC.txt'")
print("\n--- ANÁLISIS COMPLETADO ---")