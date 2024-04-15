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

#Parte 5

def funcion_forma_tetraedro(xi, eta, zeta):
    N = np.array([
        [1 - xi - eta - zeta],
        [xi],
        [eta],
        [zeta]
    ])
    return N

# Ejemplo de uso
xi = 0.1
eta = 0.2
zeta = 0.3

N = funcion_forma_tetraedro(xi, eta, zeta)
print("Función de forma en el punto (xi, eta, zeta):")
print(N)

#Parte 6
def ensamblar_matriz_rigidez_global_sparse(nodos, tetraedros, propiedades):
    #inicializamos la matriz de rigidez global como una matriz dispersa
    num_nodos = len(nodos)
    K_global = lil_matrix((3*num_nodos, 3*num_nodos))
    
    #iteramos sobre cada tetraedro para ensamblar su matriz de rigidez en la matriz global
    for t in tetraedros:
        #calcular matriz de rigidez local para el tetraedro actual
        matriz_rigidez_local = calcular_matriz_rigidez_local(nodos, t, propiedades)
        #ensamblar matriz de rigidez local en matriz global
        ensmablar_matriz_local (K_global, matriz_rigidez_local, t)
    return K_global
def calcular_matriz_rigidez_local(nodos, tetraedro, propiedades):
    #propiedades del material
    E = propiedades["E"]
    nu = propiedades["nu"]
    
    #coordenadas de los nodos del tetraedro
    x0, y0, z0 = nodos[tetraedro[0]]
    x1, y1, z1 = nodos[tetraedro[1]]
    x2, y2, z2 = nodos[tetraedro[2]]
    x3, y3, z3 = nodos[tetraedro[3]]
    
    #calculo de las derivadas de las funciones de forma
    dn_dxi = np.array([-1,1, 0, 0])
    dn_deta = np.array([-1, 0, 1, 0])
    dn_dzeta = np.array([-1, 0, 0, 1])
    
    #jacoviano
    J = np.array([
        [x1-x0, x2-x0, x3-x0],
        [y1-y0, y2-y0, y3-y0],
        [z1-z0, z2-z0, z3-z0]
    ])
    detJ = np.linalg.det(J)
    #matriz del gradiente de las funciones de forma
    B = np.array([
        [dn_dxi[0], 0, 0, dn_dxi[1], 0, 0, dn_dxi[2], 0, 0, dn_dxi[3], 0, 0],
        [0, dn_deta[0], 0, 0, dn_deta[1], 0, 0, dn_deta[2], 0, 0, dn_deta[3], 0],
        [0, 0, dn_dzeta[0], 0, 0, dn_dzeta[1], 0, 0, dn_dzeta[2], 0, 0, dn_dzeta[3]],
        [dn_deta[0], dn_dxi[0], 0, dn_deta[1], dn_dxi[1], 0, dn_deta[2], dn_dxi[2], 0, dn_deta[3], dn_dxi[3], 0],
        [0, dn_dzeta[0], dn_deta[0], 0, dn_dzeta[1], dn_deta[1], 0, dn_dzeta[2], dn_deta[2], 0, dn_dzeta[3], dn_deta[3]],
        [dn_dzeta[0], 0, dn_dxi[0], dn_dzeta[1], 0, dn_dxi[1], dn_dzeta[2], 0, dn_dxi[2], dn_dzeta[3], 0, dn_dxi[3]]
    ]) / detJ
 