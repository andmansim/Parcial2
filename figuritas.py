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

#Parte3
#definimos los datos
def calcular_tensor_deformaciones(desplazamiento):
    # Derivadas de las funciones de forma respecto a las coordenadas naturales del elemento
    B = np.array([
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1]
    ])

    # Calcular el tensor de deformaciones
    strain_tensor= np.dot(B, desplazamiento)
    return strain_tensor



# Ejemplo de uso
desplazamientos_manual = np.array([
    [0.1, 0.2, 0.3],  
    [0.2, 0.3, 0.4],  
    [0.3, 0.4, 0.5],  
    [0.4, 0.5, 0.6],  
    [0.5, 0.6, 0.7]   
])
'''
tensor_deformaciones = calcular_tensor_deformaciones(desplazamientos)
print("Tensor de deformaciones:")
print(tensor_deformaciones)'''

#Parte 4

def generar_mallado():
    # Lista de coordenadas de nodos
    coordenadas_nodos = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]
    
    # Lista de índices de nodos que forman los tetraedros
    indices_tetraedros = [
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ]

    # Crear una lista vacía para almacenar los nodos
    nodos = []
    # Crear una lista vacía para almacenar los tetraedros
    tetraedros = []

    # Agregar los nodos al mallado
    for coordenada in coordenadas_nodos:
        nodos.append(coordenada)

    # Agregar los tetraedros al mallado
    for tetraedro in indices_tetraedros:
        tetraedros.append(tetraedro)

    return nodos, tetraedros

# Ejemplo de uso
nodos, tetraedros = generar_mallado()

print("Nodos:")
print(nodos)
print("Tetraedros:")
print(tetraedros)
