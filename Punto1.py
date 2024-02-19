import numpy as np

# Generar aleatoriamente el número de filas y columnas
filas = 6  
columnas = 5  

# Simular una matriz rectangular aleatoria A
A = np.random.rand(filas, columnas)

print("Matriz A:")
print(A)

# Calcular el Rango y la Traza de A
rango_A = np.linalg.matrix_rank(A)
traza_A = np.trace(A)

print("\nRango de A:", rango_A)
print("Traza de A:", traza_A)


# Calcular la transpuesta de A
ATransp=A.transpose()
print("A Transpose: \n",ATransp)
print("Original A Matrix: \n",A)

# AA' A'A
AAT=np.dot(A,ATransp)
ATA=np.dot(ATransp,A)

print("AA': \n",np.dot(A,ATransp))
print("A'A: \n",np.dot(ATransp,A))


eigenvalues_AAT, U = np.linalg.eigh(np.dot(A,ATransp))
print("eigenvalues AA':",eigenvalues_AAT)
print("eigenvectors:\n",U)

eigenvalues_ATA, V = np.linalg.eigh(np.dot(ATransp,A))
print("eigenvalues A'A:",eigenvalues_ATA)
print("eigenvectors:\n",V)
