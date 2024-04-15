import numpy as np
import pandas as pd
import meshpy.test as tet

#creamos un dominio de 10 matrices de 0s de 10x10
domain = np.zeros((10,10, 10))
print(domain)

#creamos un csv con las coordenadas x, y,z y la presion
df = pd.DataFrame({'x':[0,1], 'y':[0,1], 'z':[0,1], 'pressure':[100,150]})
df.to_csv('data.csv', index=False)

#calculo de gradientes
def tensor_deformacion(desplazamientos):
    # Implementa el cálculo del gradiente de desplazamientos
    # Este es un ejemplo simplificado
    return np.gradient(desplazamientos)

points, facets = [ ... ], [ ... ]  # Define puntos y caras
mesh_info = tet.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets)
mesh = tet.build(mesh_info)