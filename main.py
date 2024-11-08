import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import funciones_aux as faux
from data import datos

EPOCAS = 10
resonancia = 0.6
tasa_aprendizaje = 0.1
caracteristicas = 3
neuronas_salida = 2
pesos = np.array([[0.1, 0.1, 0.1], # Pesos para la neurona A1
                  [0.5, 0.5, 0.5] # Pesos para la neurona A2
                  ])
neuronas_agregadas = 0

for epoca in range(EPOCAS):
    print(f"Epoca: {epoca}")
    salidas = np.zeros((neuronas_salida))
    
    for patron in datos:
        salidas = faux.propagar(patron, pesos)
        
        indice_neurona_ganadora = faux.obtener_neurona_ganadora(patron, pesos, salidas, resonancia)
        # Ninguna neurona supera el umbral de resonancia
        if (indice_neurona_ganadora == -1):
            # Agregar pesos de una neurona nueva
            pesos = np.append(pesos,[patron],axis=0)
            categorias.append(len(pesos) - 1)  
            neuronas_agregadas += 1
        
        # Hay una neurona seleccionada (actualizar pesos)
        else:
            pesos[indice_neurona_ganadora] = faux.actualizar_pesos(patron,pesos[indice_neurona_ganadora],tasa_aprendizaje)
            categorias.append(indice_neurona_ganadora)
print(pesos)
print(f"Neuronas agregadas en total: {neuronas_agregadas}")

pesos_finales = pesos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D para características tridimensionales

# Visualización de las neuronas (centroides de cada categoría)
for i, peso in enumerate(pesos_finales):
    ax.scatter(peso[0], peso[1], peso[2], marker='o', s=100, label=f'Neurona {i}')

# Visualización de los patrones asignados a cada categoría
for i, patron in enumerate(datos):
    categoria = categorias[i]
    ax.scatter(patron[0], patron[1], patron[2], marker='x', label=f'Patron {i} - Categoría {categoria}')

ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.set_zlabel('Característica 3')
plt.legend()
plt.show()
