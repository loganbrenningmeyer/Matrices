import numpy as np

def main():
    # A = np.random.normal(size=(2,3))
    # B = np.random.normal(size=(3,4))
    # C = multiply(A, B)
    # C_test = np.matmul(A, B)

    # print(C)
    # print(C_test)

    # print("Transpose Test\n")
    # A = np.random.randint(0, 10 + 1, size=(2,3))
    # print(A)
    # print(transpose(A))
    # print(A.T)

    # A = np.arange(9).reshape((3,3))
    # print(A)
    # print(submatrix(A, 0, 0))

    # print(det(A))

    A = np.random.randint(0, 10, size=(3,3))

    A_inverse = inverse(A)

    print(multiply(A, A_inverse))

'''
Matrix Operations
'''

'''
Scalar Multiplication

Given a matrix A and a scalar value k, multiply all of its elements by k
'''
def scalar_mult(A, k):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = k * A[i][j]
    
    return A

'''
Addition

Given matrices A and B, the sum C is given by C_ij = A_ij + B_ij
In other words, you add each respective element of the matrices
'''
def add(A, B):
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
def multiply(A, B):
    # A columns must equal B rows
    if A.shape[1] != B.shape[0]:
        return -1
    
    # Product is of shape (A rows, B cols)
    C = np.zeros(shape=(A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            # Dot product of respective A row and B column
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * transpose(B)[j][k]
            
    return C

'''
Matrix transposition

The transpose of a matrix A, denoted A^T, is obtained by sqapping rows with columns, A^T_ij = A_ji
'''
def transpose(A):
    A_T = np.zeros(shape=(A.shape[1], A.shape[0]))

    # Columns become rows
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            A_T[i][j] = A[j][i]

    return A_T

'''
Returns the submatrix of A excluding row i and column j
'''
def submatrix(A, row, col):
    sub_m = A.shape[0] - 1
    sub_n = A.shape[1] - 1

    sub = np.zeros(shape=(sub_m, sub_n))

    count = 0

    for i in [x for x in range(A.shape[0]) if x != row]:
        for j in [x for x in range(A.shape[1]) if x != col]:
            sub[count // sub_n][count % sub_n] = A[i][j]
            count += 1

    return sub


'''
Determinants

The determinant of a square matrix A, denoted det(A), is a scalar value that determines if a matrix is invertible
For a 2x2 matrix A, det(A) = ad - bc

'''
def determinant(A):
    # Verify that A is a square 2x2 matrix or larger
    if A.shape[0] != A.shape[1] or A.shape[0] == 1:
        return None
    
    # Base case, if 2x2 matrix, return determinant
    if A.shape[0] == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    # Get submatrices if A shape larger than 2x2
    else:

        det = 0

        # For each element in the first row
        for k in range(A.shape[1]): 

            sub = submatrix(A, 0, k)

            # Add determinants of submatrices
            if (k % 2 == 0):
                det += A[0][k] * determinant(sub)
            else:
                det -= A[0][k] * determinant(sub)

        return det

'''
Inverse of a matrix

The inverse of a matrix A, denoted A^-1, is a matrix such that A*A^-1 = A^-1*A = I (identity matrix)
'''
def inverse(A):

    # Check if det(A) == 0
    if determinant(A) == 0:
        return None

    m = A.shape[0]
    n = A.shape[1]

    # Calculate the matrix of Minors
    minors = np.zeros(shape=(m, n))

    for i in range(m):
        for j in range(n):
            # Find the submatrix
            sub = submatrix(A, i, j)

            minors[i][j] = determinant(sub)

    # Calculate the matrix of Cofactors
    count = 0

    for i in range(m):
        for j in range(n):
            if count % 2 != 0:
                minors[i][j] *= -1
            count += 1

    # Calculate the Adjugate
    adjugate = transpose(minors)

    # Divide by det(A)
    inverse = scalar_mult(adjugate, 1/determinant(A))

    return inverse

'''
Eigenvalues and Eigenvectors

For a square matrix A, a scalar λ is an eigenvalue and a non-zero vector v is an eigenvector if Av = λv
'''
def eigenvalue(A):
    return None
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

if __name__ == "__main__":
    main()