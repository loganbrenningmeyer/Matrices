import numpy as np

'''
Matrix Operations
'''

'''
Addition

Given matrices A and B, the sum C is given by C_ij = A_ij + B_ij
In other words, you add each respective element of the matrices
'''
def matrix_add(A, B):
    # Shapes must match
    if A.shape != B.shape:
        return -1

    C = np.zeros(shape=A.shape)

    # Sum element-wise
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] = A[i][j] + B[i][j]

    return C

'''
Multiplication

Given matrix A of size m*n and matrix B of size n*p, the product C_ij = sum(k=1 --> n) A_ik * B_kj
A's columns must match the length of B's rows, and the element C_ij equals the sum of A's elements in the same row times B's elements in the same column
'''
def matrix_mult(A, B):
    # A rows must equal B cols
    if A.shape[0] != B.shape[1]:
        return -1

'''
Matrix transposition

The transpose of a matrix A, denoted A^T, is obtained by sqapping rows with columns, A^T_ij = A_ji
'''


'''
Determinants

The determinant of a square matrix A, denoted det(A), is a scalar value that determines if a matrix is invertible
For a 2x2 matrix A, det(A) = ad - bc

'''

'''
Inverse of a matrix

The inverse of a matrix A, denoted A^-1, is a matrix such that A*A^-1 = A^-1*A = I (identity matrix)
'''

'''
Eigenvalues and Eigenvectors

For a square matrix A, a scalar λ is an eigenvalue and a non-zero vector v is an eigenvector if Av = λv
'''

'''
Matrix decomposition
'''

'''
Special matrices
'''

'''
Matrix transformations
'''

'''
Solving linear systems
'''

'''
Matrix norms
'''