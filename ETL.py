# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:46:24 2024

@author: Alberto Gallegos
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Button, Frame, messagebox, Toplevel

# Cargar los datos desde los archivos CSV en la misma carpeta
df_resultados = pd.read_csv('resultados_trimestre.csv')
df_volumen = pd.read_csv('volumen.csv')
#df_ingresos = pd.read_csv('ingresos.csv')

# Seleccionar y limpiar columnas específicas
df_resultados = df_resultados[['Tiempo', 'Ingresos totales']]
df_volumen = df_volumen[['1T 2024']].rename(columns={'1T 2024': 'Volumen'})
#df_ingresos = df_ingresos[['1T 2024']].rename(columns={'1T 2024': 'Ingresos'})
print(df_resultados)
print(df_volumen)

#print(df_ingresos)

# Convertir a numérico y eliminar filas no numéricas
df_volumen['Volumen'] = pd.to_numeric(df_volumen['Volumen'], errors='coerce')
#df_ingresos['Ingresos'] = pd.to_numeric(df_ingresos['Ingresos'], errors='coerce')

# Unir los datos en un solo DataFrame y eliminar filas con valores no numéricos
#df_combined = pd.concat([df_resultados, df_volumen, df_ingresos], axis=1).dropna()

if not df_combined.empty:
    # Regresión lineal: usando 'Volumen' para predecir 'Ingresos'
    X = df_combined[['Volumen']]
    y = df_combined['Ingresos']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar modelo de regresión lineal
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # Predicción de ingresos
    y_pred = regression_model.predict(X_test)

    # Segmentación con K-means
    X_kmeans = df_combined[['Volumen', 'Ingresos']]
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_combined['Cluster'] = kmeans.fit_predict(X_kmeans)

    # Interfaz gráfica en Tkinter
    root = Tk()
    root.title("Análisis de Negocios de Coca-Cola")
    root.geometry("400x300")
    root.configure(bg="#e1f5fe")

    # Título de la interfaz
    title_label = Label(root, text="Resultados de Minería de Datos", 
                       font=("Arial", 16, "bold"), bg="#e1f5fe", fg="#0277bd")
    title_label.pack(pady=10)

    # Marco para organizar botones
    frame = Frame(root, bg="#e1f5fe")
    frame.pack(pady=20)

    def show_predictions():
        predictions = "\n".join([f"Predicción {i+1}: {pred:.2f}" for i, pred in enumerate(y_pred)])
        messagebox.showinfo("Predicciones de Ingresos", predictions)

    def show_kmeans_plot():
        # Crear una nueva ventana para la gráfica
        plot_window = Toplevel(root)
        plot_window.title("Gráfica de Clusters")
        plot_window.geometry("800x600")

        # Crear la figura y el lienzo
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df_combined['Volumen'], df_combined['Ingresos'], 
                           c=df_combined['Cluster'], cmap='viridis')
        ax.set_xlabel("Volumen")
        ax.set_ylabel("Ingresos")
        ax.set_title("Segmentación de Regiones por K-means")
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # Crear el canvas de Matplotlib
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        # Botón para cerrar la ventana de la gráfica
        close_button = Button(plot_window, text="Cerrar", 
                            command=plot_window.destroy,
                            font=("Arial", 12), bg="#b0bec5", fg="black")
        close_button.pack(pady=10)

    # Botones de la interfaz principal
    pred_button = Button(frame, text="Mostrar Predicciones de Ingresos", 
                        command=show_predictions,
                        font=("Arial", 12), bg="#0288d1", fg="white", 
                        padx=10, pady=5)
    pred_button.grid(row=0, column=0, padx=10, pady=10)

    kmeans_button = Button(frame, text="Ver Gráfica de Clusters", 
                          command=show_kmeans_plot,
                          font=("Arial", 12), bg="#0288d1", fg="white", 
                          padx=10, pady=5)
    kmeans_button.grid(row=0, column=1, padx=10, pady=10)

    # Botón para cerrar la aplicación
    close_button = Button(root, text="Cerrar", command=root.destroy,
                         font=("Arial", 12), bg="#b0bec5", fg="black",
                         padx=10, pady=5)
    close_button.pack(pady=10)

    root.mainloop()
else:
    messagebox.showerror("Error", "No se encontraron datos válidos para el análisis.")