# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:46:24 2024

@author: Alberto Gallegos
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Button, Frame, Toplevel, PhotoImage, Text
from tkinter import ttk
from scipy.stats import pearsonr

# Configuración de colores de la interfaz
PRIMARY_COLOR = "#C8102E"   # Rojo Coca-Cola
SECONDARY_COLOR = "#1D1D1B" # Negro
BACKGROUND_COLOR = "#FFFFFF" # Blanco
TEXT_COLOR = "#FFFFFF"       # Blanco para texto

# Cargar datos
df_resultados = pd.read_csv('mineria/resultados_trimestre.csv')
df_volumen = pd.read_csv('mineria/volumen.csv')
df_ingresos = pd.read_csv('mineria/ingresos.csv')

# Preprocesamiento de datos
df_volumen = df_volumen.rename(columns={'Unnamed: 0': 'Pais', '1T 2024': 'Volumen'})
df_ingresos = df_ingresos.rename(columns={'1T 2024': 'Ingresos'})
df_volumen['Volumen'] = pd.to_numeric(df_volumen['Volumen'], errors='coerce')
df_ingresos['Ingresos'] = pd.to_numeric(df_ingresos['Ingresos'], errors='coerce')

df_combined = pd.concat([df_resultados[['Tiempo', 'Ingresos totales']], df_volumen[['Volumen']], df_ingresos[['Ingresos']]], axis=1).dropna()

# Análisis de regresión y clusters
X = df_combined[['Volumen']]
y = df_combined['Ingresos']

# Regresión Lineal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Regresión Polinómica
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)
y_pred_poly = poly_model.predict(X_test_poly)

# Análisis de Clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42).fit(df_combined[['Volumen', 'Ingresos']])
kmeans_3 = KMeans(n_clusters=3, random_state=42).fit(df_combined[['Volumen', 'Ingresos']])
df_combined['Cluster_2'] = kmeans_2.labels_
df_combined['Cluster_3'] = kmeans_3.labels_

# Análisis de Correlación
corr_vol_ing, _ = pearsonr(df_combined['Volumen'], df_combined['Ingresos'])

# Predicción del Producto Más Vendido por País
def predecir_producto_mas_vendido(df_volumen):
    df_volumen = df_volumen.dropna(subset=['Pais'])
    df_volumen = df_volumen[~df_volumen['Pais'].str.contains("En hoja", case=False, na=False)]

    trimestres_productos = ['2T 2024', '1T 2024', '4T 2023', '3T 2023']
    productos = ['Refrescos', 'Agua (1)', 'Garrafon (2)', 'Otros']
    data = []

    for index, row in df_volumen.iterrows():
        pais = row['Pais']
        if pd.notna(pais):
            for i, trimestre in enumerate(trimestres_productos):
                volumen = pd.to_numeric(row.get(trimestre, None), errors='coerce')
                if volumen is not None:
                    data.append({'Pais': pais, 'Producto': productos[i], 'Volumen': volumen})

    df_productos = pd.DataFrame(data)
    df_agrupado = df_productos.groupby(['Pais', 'Producto']).agg({'Volumen': 'sum'}).reset_index()
    df_agrupado['Producto_mas_vendido'] = df_agrupado.groupby('Pais')['Volumen'].transform(max) == df_agrupado['Volumen']
    return df_agrupado[df_agrupado['Producto_mas_vendido']].drop(columns=['Producto_mas_vendido'])

df_top_productos = predecir_producto_mas_vendido(df_volumen)

# Crear la ventana principal
root = Tk()
root.title("Análisis de Negocios de Coca-Cola")
root.geometry("1000x800")
root.configure(bg=BACKGROUND_COLOR)

header_frame = Frame(root, bg=PRIMARY_COLOR, height=100)
header_frame.pack(fill='x')

Label(header_frame, text="Análisis de Negocios de Coca-Cola", font=("Arial", 24, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR).pack()

notebook = ttk.Notebook(root)
notebook.pack(expand=True, padx=10, pady=10)

# Pestaña de Predicciones de Ingresos
frame_predicciones = Frame(notebook, bg=BACKGROUND_COLOR)
notebook.add(frame_predicciones, text="Predicciones de Ingresos")

pred_text_linear = "\n".join([f"Predicción Lineal {i+1}: {pred:.2f}" for i, pred in enumerate(y_pred_linear)])
pred_text_poly = "\n".join([f"Predicción Polinómica {i+1}: {pred:.2f}" for i, pred in enumerate(y_pred_poly)])
Label(frame_predicciones, text=f"Regresión Lineal:\n{pred_text_linear}\n\nRegresión Polinómica:\n{pred_text_poly}", font=("Arial", 12)).pack()

# Pestaña de Producto Más Vendido
frame_producto = Frame(notebook, bg=BACKGROUND_COLOR)
notebook.add(frame_producto, text="Producto Más Vendido")
text_box = Text(frame_producto, height=20, width=70, bg=BACKGROUND_COLOR, fg=SECONDARY_COLOR, font=("Arial", 12))
text_box.pack(pady=10)
for _, row in df_top_productos.iterrows():
    text_box.insert("end", f"{row['Pais']}: {row['Producto']} (Volumen: {row['Volumen']:.2f})\n")

# Pestaña de Análisis de Clusters
frame_clusters = Frame(notebook, bg=BACKGROUND_COLOR)
notebook.add(frame_clusters, text="Análisis de Clusters")

def show_kmeans_plot(n_clusters):
    plot_window = Toplevel(root)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_combined['Volumen'], df_combined['Ingresos'], c=df_combined[f'Cluster_{n_clusters}'], cmap='viridis')
    plt.colorbar(scatter)
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

Button(frame_clusters, text="Ver Clusters (2)", command=lambda: show_kmeans_plot(2)).pack()
Button(frame_clusters, text="Ver Clusters (3)", command=lambda: show_kmeans_plot(3)).pack()

root.mainloop()
