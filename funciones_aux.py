import numpy as np

def propagar(patron, pesos):
    return np.dot(patron, pesos.T)


def obtener_neurona_ganadora(patron, pesos, salidas, umbral):
    arreglo_salidas = np.copy(salidas)
    arreglo_pesos = np.copy(pesos)
    
    while arreglo_salidas.size != 0:
        # Encontrar la neurona con mayor salida (potencial)
        indice_mayor = np.argmax(arreglo_salidas)
        
        # Comprobamos si la similitud coseno supera el umbral
        if calcular_similitud_cos(patron, arreglo_pesos[indice_mayor]) > umbral:
            # Actualizar la categoría del patrón basado en la neurona ganadora
            
            return indice_mayor
        else:
            # Eliminamos el peso y la salida correspondiente si no supera el umbral
            arreglo_salidas = np.delete(arreglo_salidas, indice_mayor)
            arreglo_pesos = np.delete(arreglo_pesos, indice_mayor, axis=0)
    
    # Si no encontramos una neurona ganadora que supere el umbral, agregamos a nueva categoría
  
    return -1 

def actualizar_pesos(patron, peso, tasa):
    # Ajustamos la fórmula para acercar el peso al patrón de entrada
    nuevo_peso = peso + tasa * (patron - peso)
    
    return nuevo_peso

def calcular_similitud_cos(P, W):
    numerador = np.dot(W, P)
    norma_W = np.linalg.norm(W)
    norma_P = np.linalg.norm(P)
    denominador = norma_W * norma_P

    # Similaridad
    similitud = numerador / denominador
    return similitud

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graficar_patrones_y_pesos(datos, pesos, categorias):
    # Colores para cada categoría de neuronas
    colores_neuronas = ['r', 'g', 'b', 'y', 'c', 'm']  # Puedes personalizar estos colores
    
    # Crear figura y ejes 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los patrones, coloreados según su categoría
    for i, patron in enumerate(datos):
        ax.scatter(patron[0], patron[1], patron[2], color=colores_neuronas[categorias[i]], alpha=0.6)

    # Graficar los pesos finales de las neuronas
    for i, peso in enumerate(pesos):
        ax.scatter(peso[0], peso[1], peso[2], color=colores_neuronas[i % len(colores_neuronas)], s=100, marker='o', label=f'Neurona {i}')

    # Configurar etiquetas de ejes
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_zlabel('Característica 3')

    # Título y leyenda
    ax.set_title('Visualización de Patrones y Pesos Finales de Neuronas')
    ax.legend()

    # Mostrar gráfico
    plt.show()
