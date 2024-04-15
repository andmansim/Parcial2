import numpy as np
import pandas as pd
import meshpy.tet as tet
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import pyvtk

#Parte 1
#definimos las dimansiones del dominio
nx = 10 #número de nodos en x
ny = 10 #número de nodos en y
nz = 10 #número de nodos en z
#creamos un dominio de 10x10x10
domain = np.zeros((nx,ny, nz))

#generamos coordenadas 
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)

#creamos cuadricula
X, Y, Z = np.meshgrid(x, y, z)

#visualizamos la cuadricula
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Dominio Estructural')
plt.show()

#Parte 2
def exportar_paraview(nombre, presion, desplazamineto):
    #creamos una malla
    puntos = np.column_stack((np.arange(len(presion)), np.zeros(len(presion)), np.zeros(len(presion))))
    mesh = pyvtk.UnstructuredGrid(puntos, point_data={"Presion": presion, "Desplazamiento": desplazamineto})

    #lo pasamos a vtk
    mesh.tofile(nombre)


