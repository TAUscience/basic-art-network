import numpy as np
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
            neuronas_agregadas += 1
        
        # Hay una neurona seleccionada (actualizar pesos)
        else:
            pesos[indice_neurona_ganadora] = faux.actualizar_pesos(patron,pesos[indice_neurona_ganadora],tasa_aprendizaje)

print(pesos)
print(f"Neuronas agregadas en total: {neuronas_agregadas}")

