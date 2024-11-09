from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import numpy as np

wine_data = load_wine()
primeras_tres_caracteristicas = wine_data.data[:, :3]

scaler = MinMaxScaler(feature_range=(0, 1))
datos_normalizados = scaler.fit_transform(primeras_tres_caracteristicas)

datos = np.array([row for row in datos_normalizados])
