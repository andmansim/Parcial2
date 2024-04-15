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
def ensamablar_matriz_local(K_global, K_local, tetraedro):
    #tamaño de la matriz de rigidez local
    n = K_local.shape[0]
    
    #convertir matriz local a matriz dispersa
    K_local = lil_matrix(K_local)
    
    #ínidices de los nodos del tetraedro
    i, j, k, l = tetraedro
    
    #ensamblar matriz local en matriz global
    for i_local, j_local in enumerate([i, j, k, l]):
        for k_local, l_local in enumerate([i, j, k, l]):
            K_global[3 * i_local:3 * i_local + 3, 3 * k_local:3 * k_local + 3] += K_local[3 * i_local:3 * i_local + 3, 3 * k_local:3 * k_local + 3]
    
          
def ensamblar_matriz_rigidez_global(nodos, tetraedros, propiedades):
    #inicializamos la matriz de rigidez global como una matriz dispersa
    num_nodos = len(nodos)
    K_global = lil_matrix((3*num_nodos, 3*num_nodos))
    
    #iteramos sobre cada tetraedro para ensamblar su matriz de rigidez en la matriz global
    for t in tetraedros:
        #calcular matriz de rigidez local para el tetraedro actual
        matriz_rigidez_local = calcular_matriz_rigidez_local(nodos, t, propiedades)
        #ensamblar matriz de rigidez local en matriz global
        ensamablar_matriz_local(K_global, matriz_rigidez_local, t)
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
    
    #matriz de elasticidad
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C = factor * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    
    #matriz de rigidez local
    K_local = np.dot(np.dot(B.T, C), B) * detJ
    return K_local

#ejemplo 
nodos, tetraedros = generar_mallado()
propiedades = {"E": 1, "nu": 0.3}
K_global = ensamblar_matriz_rigidez_global(nodos, tetraedros, propiedades)
print("Matriz de rigidez global:")
print(K_global.toarray())  

  
#Parte 7
def resolver_sistema(K_global, f):

    #resolver sistema de ecuaciones
    u = np.linalg.solve(K_global, f)
    return u

'''#ejemplo
fuerzas = np.array([0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1])
desplazamientos = resolver_sistema(K_global, fuerzas)
print("Desplazamientos:")
print(desplazamientos)
'''
#Parte 8
def calcular_tensor_deformaciones(K_global, desplazamiento, nodos, tetraedros, propiedades):
    #inicializamos deformaciones y tensores
    tensores = []
    deformaciones = []
    
    #iteramos sobre cada tetraedro para calcular las deformaciones y tensores
    for t in tetraedros:
        #coordenadas
        coord = np.array([nodos[i] for i in t])
        
        #matriz de deformacion para el tetraedro actual
        B = calcular_matriz_deformacion(coord)
        
        #desplazamientos del tetraedro actual
        despl = np.array([desplazamiento[3 * i:3 * i + 3] for i in t]).flatten()
        
        #desplazamientos del tetraedro actual en coordenadas globales
        despl_glob = np.dot(B, despl)
        
        #calcular tensor de deformaciones
        deformacion = calcular_deform_elem(B, despl)
        
        #calcular tensor de esfuerzos
        tensor = calcular_tensor_esfuerzos(deformacion, propiedades)
        
        #almacenar tensor y deformacion
        tensores.append(tensor)
        deformaciones.append(deformacion)
    return tensores, deformaciones

def calcular_matriz_deformacion(coord):
    #matriz de deformacion
    B = np.zeros((6, 12))
    
    #calcular derivadas de las funciones de forma
    dn_dxi = np.array([-1, 1, 0, 0])
    dn_deta = np.array([-1, 0, 1, 0])
    dn_dzeta = np.array([-1, 0, 0, 1])
    
    #iterar sobre cada nodo del tetraedro
    for i in range(4):
        #coordenadas del nodo
        x, y, z = coord[i]
        
        #calcular matriz de deformacion
        B[0, 3 * i] = dn_dxi[i]
        B[1, 3 * i + 1] = dn_deta[i]
        B[2, 3 * i + 2] = dn_dzeta[i]
        B[3, 3 * i] = dn_deta[i]
        B[3, 3 * i + 1] = dn_dxi[i]
        B[4, 3 * i + 1] = dn_dzeta[i]
        B[4, 3 * i + 2] = dn_deta[i]
        B[5, 3 * i] = dn_dzeta[i]
        B[5, 3 * i + 2] = dn_dxi[i]
    return B

def calcular_deform_elem(B, despl):
    #calcular tensor de deformaciones
    deformacion = np.dot(B, despl)
    return deformacion

def calcular_tensor_esfuerzos(deformacion, propiedades):
    #propiedades del material
    E = propiedades["E"]
    nu = propiedades["nu"]
    
    #calcular tensor de esfuerzos
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C = factor * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    tensor = np.dot(C, deformacion)
    return tensor

def visualizar_solucion(tensiores, deformaciones, nodos, tetraedros):
    #creamos un objeto vtkUnstructuredGrid para almacenar la malla
    grid =pyvtk.UnstructuredGrid()
    
    #objeto vtkpoints para almacenar los nodos
    points = pyvtk.Points()
    
    #almacenar nodos
    for nodo in nodos:
        points.InserNextPoint(nodo)
        
    #almacenar puntos en la malla
    grid.SetPoints(points)
    
    #crear un array para almacenar las tensiones
    tensiones_array = pyvtk.DoubleArray()
    tensiones_array.SetNumberOfComponents(1)
    tensiones_array.SetName("Tensiones")
    
    #almacenar tensiones
    for tension in tensiores:
        tensiones_array.InsertNextValue(tension)
        
    #añadir tensiones a la malla
    grid.GetPointData().AddArray(tensiones_array)
    
    #crear un array para almacenar las deformaciones
    deformaciones_array = pyvtk.DoubleArray()
    deformaciones_array.SetNumberOfComponents(1)
    deformaciones_array.SetName("Deformaciones")
    
    #almacenar deformaciones
    for deformacion in deformaciones:
        deformaciones_array.InsertNextValue(deformacion)
    
    #añadir deformaciones a la malla
    grid.GetPointData().AddArray(deformaciones_array)
    
    #crear un array para almacenar los tetraedros
    tetraedros_array = pyvtk.CellArray()
    
    #almacenar tetraedros
    for tetraedro in tetraedros:
        cell = pyvtk.Tetra()
        for i, a in enumerate(tetraedro):
            cell.GetPointIds().SetId(i, a)
        tetraedros_array.InsertNextCell(cell)
    
    #añadir tetraedros a la malla
    grid.SetCells(pyvtk.VTK_TETRA, tetraedros_array)
    
    #crear un objeto vtkXMLUnstructuredGridWriter para exportar la malla
    writer = pyvtk.XMLUnstructuredGridWriter()
    writer.SetFileName("solucion.vtu")
    writer.SetInputData(grid)
    writer.Write()

#ejemplo


#visualizar_solucion(tensiones, deformaciones, nodos, tetraedros)
