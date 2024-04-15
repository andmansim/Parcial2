import numpy as np
import pandas as pd
import meshpy.tet as tet
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from pyvtk import VtkData, UnstructuredGrid, PointData, CellData, Scalars


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

#presiones y desplazamientos
lx, ly, lz = 1.0, 1.0, 1.0
dx, dy, dz = lx/nx, ly/ny, lz/nz

def exportar_resultados_a_vtk(presiones, desplazamientos, lx, ly, lz, nx, ny, nz):
    # Crear un objeto VTK UnstructuredGrid
    grid = VtkData(UnstructuredGrid([nx, ny, nz]), PointData(Scalars(presiones.ravel(), name="Pressure")), CellData(Scalars(np.arange(nx*ny*nz), name="CellID"))

    # Crear un arreglo de puntos para almacenar los desplazamientos
    points = np.zeros((nx*ny*nz, 3))

    # Agregar los puntos al arreglo
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points[i*ny*nz + j*nz + k] = [i*dx, j*dy, k*dz]

    # Asignar los puntos al objeto grid
    grid.point_data = PointData(Scalars(presiones.ravel(), name="Pressure"), Scalars(points, name="Displacement"))

    # Escribir el archivo VTK
    grid.tofile("results.vtk")

# Ejemplo de uso
pressure_data = np.random.rand(nx, ny, nz)
displacement_data = np.random.rand(nx+1, ny+1, nz+1, 3)
exportar_resultados_a_vtk(pressure_data, displacement_data, lx, ly, lz, nx, ny, nz)



'''def export_to_vtk(filename, pressures, displacements):
    # Crear un objeto VTK UnstructuredGrid
    grid = vtk.vtkUnstructuredGrid()

    # Crear un arreglo de puntos para almacenar los desplazamientos
    points = vtk.vtkPoints()

    # Agregar los puntos al arreglo
    nx, ny, nz = pressures.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.InsertNextPoint(i, j, k)

    # Asignar los puntos al objeto grid
    grid.SetPoints(points)

    # Crear un arreglo de datos escalares para las presiones
    pressure_data = vtk.vtkFloatArray()
    pressure_data.SetNumberOfComponents(1)
    pressure_data.SetName("Pressure")

    # Crear un arreglo de datos vectoriales para los desplazamientos
    displacement_data = vtk.vtkFloatArray()
    displacement_data.SetNumberOfComponents(3)
    displacement_data.SetName("Displacement")

    # Agregar los datos al arreglo
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pressure_data.InsertNextValue(pressures[i, j, k])
                displacement_data.InsertNextTuple3(displacements[i, j, k, 0], displacements[i, j, k, 1], displacements[i, j, k, 2])

    # Agregar los arreglos de datos al objeto grid
    grid.GetPointData().AddArray(pressure_data)
    grid.GetPointData().AddArray(displacement_data)

    # Escribir el archivo VTK
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

# Ejemplo de uso
pressures = np.random.rand(5, 5, 5)  # Presiones aleatorias como ejemplo
displacements = np.random.rand(5, 5, 5, 3)  # Desplazamientos aleatorios como ejemplo

export_to_vtk("results.vtu", pressures, displacements)
#Paraview
#creamos un csv con las coordenadas x, y,z y la presion
df = pd.DataFrame({'x':[0,1], 'y':[0,1], 'z':[0,1], 'pressure':[100,150]})
df.to_csv('data.csv', index=False)

#Funciones auxiliares 
def tensor_deformacion(desplazamientos):
    # Implementa el cálculo del gradiente de desplazamientos
    # Este es un ejemplo simplificado
    return np.gradient(desplazamientos)

#Preprocesamiento de datos de entrada
points, facets = [ ... ], [ ... ]  # Define puntos y caras
mesh_info = tet.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets)
mesh = tet.build(mesh_info)

#implementación de la función de forma
def funcion_forma(punto, nodos):
    # Implementa la función de forma
    return np.array([ ... ])

K_global = lil_matrix((nodos_totales, nodos_totales))
# Suma las matrices locales de rigidez al K_global en los índices correctos'''