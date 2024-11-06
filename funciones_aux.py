import numpy as np

def propagar(patron, pesos):
    return np.dot(patron,pesos.T)

def obtener_neurona_ganadora(patron, pesos, salidas, umbral):
    arreglo_salidas = np.copy(salidas)
    while(arreglo_salidas.size != 0):
        indice_mayor = np.argmax(arreglo_salidas)
        if calcular_similitud_cos(patron, pesos[indice_mayor]) > umbral:
            return indice_mayor
        else:
            arreglo_salidas = np.delete(arreglo_salidas,indice_mayor)

    return -1

def actualizar_pesos(patron, peso, tasa):
    nuevo_peso = np.add(peso,(tasa*(peso-patron)))
    return nuevo_peso

def calcular_similitud_cos(P, W):
    numerador = np.dot(W, P)

    norma_W = np.sqrt(np.sum(W**2))
    norma_P = np.sqrt(np.sum(P**2))
    denominador = norma_W * norma_P

    # Similaridad
    similitud = numerador / denominador
    return similitud