import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import funciones_aux as faux
from data import datos

EPOCAS = 100
resonancia = 0.65
tasa_aprendizaje = 0.5
caracteristicas = 3
neuronas_salida = 2
pesos = np.random.rand(2,3)
neuronas_agregadas = 0
categorias = np.zeros(len(datos), dtype=int)  # Inicializar las categorías para cada patrón

for epoca in range(EPOCAS):
    print(f"\nEpoca: {epoca + 1}")
    
    for i, patron in enumerate(datos):
        salidas = faux.propagar(patron, pesos)
        
        # Obtener la neurona ganadora y asignar la categoría
        indice_neurona_ganadora = faux.obtener_neurona_ganadora(patron, pesos, salidas, resonancia)
        
        if indice_neurona_ganadora == -1:
            # Agregar nueva neurona cuando no se encuentra una neurona ganadora
            pesos = np.append(pesos, [patron], axis=0)
            neuronas_agregadas += 1
            categorias[i] = len(pesos) - 1  # Asigna la categoría de la nueva neurona # Categoriza según la neurona ganadora
            print(f"Patrón {i} asignado a nueva categoría 2 (neurona nueva)")
        else:
            # Actualizar los pesos de la neurona ganadora
            pesos[indice_neurona_ganadora] = faux.actualizar_pesos(patron, pesos[indice_neurona_ganadora], tasa_aprendizaje)
            categorias[i] = indice_neurona_ganadora  # Categoriza según la neurona ganadora
           
           

# Información final
print("\nPesos finales:", pesos)
print(f"Neuronas agregadas en total: {neuronas_agregadas}")
pesos_finales = pesos

faux.graficar_patrones_y_pesos(datos, pesos, categorias)
