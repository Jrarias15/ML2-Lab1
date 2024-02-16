import numpy as np

class SVDApproximation:
    def __init__(self, k):
        """
        Inicializa la clase con el número de componentes principales k.

        Parameters:
        - k: Número de componentes principales a retener para la aproximación
        """
        self.k = k
        self.U = None
        self.Sigma = None
        self.Vt = None

    def fit(self, A):
        """
        Realiza la descomposición SVD de la matriz A.

        Parameters:
        - A: Matriz original
        """
        self.U, self.Sigma, self.Vt = np.linalg.svd(A, full_matrices=False)

    def transform(self, A):
        """
        Aproximación de la matriz A utilizando SVD.

        Parameters:
        - A: Matriz original

        Returns:
        - A_approx: Matriz aproximada
        """
        # Reducir las matrices U, Sigma, Vt a las primeras k componentes
        U_k = self.U[:, :self.k]
        Sigma_k = np.diag(self.Sigma[:self.k])
        Vt_k = self.Vt[:self.k, :]

        # Aproximación de la matriz original A
        A_approx = np.dot(U_k, np.dot(Sigma_k, Vt_k))

        return A_approx
