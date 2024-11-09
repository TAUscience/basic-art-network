import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import funciones_aux as faux
from data import datos

EPOCAS = 3
resonancia = 0.6
tasa_aprendizaje = 0.4
caracteristicas = 3
neuronas_salida = 2
pesos = np.array([[0.1, 0.1, 0.1], # Pesos para la neurona A1
                  [0.5, 0.5, 0.5] # Pesos para la neurona A2
                  ])
neuronas_agregadas = 0
categorias = np.zeros(len(datos), dtype=int)  


for epoca in range(EPOCAS):
    print(f"\nEpoca: {epoca + 1}")
    
    for i, patron in enumerate(datos):
        salidas = faux.propagar(patron, pesos)
        
        indice_neurona_ganadora = faux.obtener_neurona_ganadora(patron, pesos, salidas, resonancia)
        
        if indice_neurona_ganadora == -1:
       
            pesos = np.append(pesos, [patron], axis=0)
            categorias[i] = 2  # Categoría para nueva neurona
            neuronas_agregadas += 1
            print(f"Patrón {i} asignado a nueva categoría 2 (neurona nueva)")
        else:
            # Actualiza los pesos de la neurona ganadora y asigna su categoría (0 o 1)
            pesos[indice_neurona_ganadora] = faux.actualizar_pesos(patron, pesos[indice_neurona_ganadora], tasa_aprendizaje)
            categorias[i] = indice_neurona_ganadora  # Categoría según neurona ganadora
            print(f"Patrón {i} asignado a categoría {categorias[i]} (neurona {indice_neurona_ganadora})")

# Información final
print("\nPesos finales:", pesos)
print(f"Neuronas agregadas en total: {neuronas_agregadas}")
print("Categorías finales asignadas a cada patrón:", categorias)
pesos_finales = pesos

# Visualización 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colores_neuronas = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm'}  

# Graficar los patrones con los colores según su categoría
for i, patron in enumerate(datos):
    ax.scatter(patron[0], patron[1], patron[2], color=colores_neuronas[categorias[i]], label=f"Patrón {i} - Categoría {categorias[i]}" if i == 0 else "")

# Graficar las neuronas con un marcador más grande y específico
for i, peso in enumerate(pesos):
    ax.scatter(peso[0], peso[1], peso[2], color=colores_neuronas[i], s=100, marker='o', label=f"Neurona {i}")

# Etiquetas de los ejes
ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.set_zlabel('Característica 3')

# Título y leyenda
ax.set_title('Posición de Patrones y Neuronas')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())  

# Mostrar el gráfico
plt.show()
