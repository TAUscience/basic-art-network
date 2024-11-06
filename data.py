from sklearn.datasets import load_wine
wine_data = load_wine()
import numpy as np

primeras_tres_caracteristicas = wine_data.data[:, :3]

datos = np.array([
    [row[0], row[1], row[2]] for row in primeras_tres_caracteristicas
])