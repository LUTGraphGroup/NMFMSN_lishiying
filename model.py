import numpy as np


class MatrixFactorization:
    def __init__(self):
        pass

    def pearson_similarity(self, x, y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((x - mean_x) * (y - mean_y))
        denominator = ((sum((x - mean_x) ** 2) * sum((y - mean_y) ** 2)) ** 0.5)
        if denominator == 0:
            return 0.99998
        else:
            return numerator / denominator

    def svd_init(self, A, k):
        U, S, V = np.linalg.svd(A)
        C = np.dot(U[:, :k], np.sqrt(np.diag(S[:k])))
        D = np.dot(V.T[:, :k], np.sqrt(np.diag(S[:k])))
        return C, D

    def random_init(self, m, n, k):
        # 随机生成非负的C和D矩阵作为初始值
        C = np.random.uniform(low=1, high=2, size=(m, k))
        D = np.random.uniform(low=1, high=2, size=(n, k))
        return C, D

    # 计算S的拉普拉斯矩阵L
    def laplacian(self, S):
        D = np.diag(np.sum(S, axis=1))
        L = D - S
        return L

    def loss_func(self, A, C, D, SC, SD, alpha, beta, gamma, GS, GD):
        p, q = A.shape

        loss = np.sum((A - np.dot(C, D.T)) ** 2) + alpha * np.sum(np.square(C)) + alpha * np.sum(np.square(D)) \
               + gamma * (np.trace(np.dot(np.dot(C.T, GS), C)) + np.trace(np.dot(np.dot(D.T, GD), D)))
        for i in range(p):
            for j in range(i + 1, p):
                corr = self.pearson_similarity(C[i], C[j])
                loss += beta * ((SC[i, j] - corr) ** 2)
        for i in range(q):
            for j in range(i + 1, q):
                corr = self.pearson_similarity(D[i], D[j])
                loss += beta * ((SD[i, j] - corr) ** 2)
        return loss / 2

    def grad_C(self, A, C, D, SC, alpha, beta, gamma, i, GS):
        p, q = A.shape
        len = C.shape[1]
        grad = np.zeros(len)

        for ii in range(C.shape[0]):
            grad+=gamma*(C[i,:]-C[ii,:])

        for j in range(q):
            grad += (np.dot(C[i], D[j]) - A[i, j]) * D[j]
        grad += alpha * C[i]
        for k in range(i + 1, p):
            corr = self.pearson_similarity(C[i], C[k])
            if np.isnan(corr):
                corr = 0
                grad += beta * (SC[i, k] - corr) * (((C[k] - np.mean(C[k])) / (
                            np.linalg.norm(C[i] - np.mean(C[i])) * np.linalg.norm(C[k] - np.mean(C[k])))) - (((np.dot(
                    (C[i] - np.mean(C[i])), (C[k] - np.mean(C[k])))) / ((np.linalg.norm(
                    C[i] - np.mean(C[i])) ** 3) * np.linalg.norm(C[k] - np.mean(C[k])))) * (C[i] - np.mean(C[i]))))
        return grad

    def grad_D(self, A, C, D, SD, alpha, beta, gamma, j, GD):
        p, q = A.shape
        len = D.shape[1]
        grad = np.zeros(len)

        for ii in range(D.shape[0]):
            grad+=gamma*(D[j,:]-D[ii,:])

        for i in range(p):
            grad += (np.dot(C[i], D[j]) - A[i, j]) * C[i]
        grad += alpha * D[j]
        for k in range(j + 1, q):
            corr = self.pearson_similarity(D[j], D[k])
            if np.isnan(corr):
                corr = 0
                grad += beta * (SD[j, k] - corr) * (((D[k] - np.mean(D[k])) / (
                            np.linalg.norm(D[j] - np.mean(D[j])) * np.linalg.norm(D[k] - np.mean(D[k])))) - (((np.dot(
                    (D[j] - np.mean(D[j])), (D[k] - np.mean(D[k])))) / ((np.linalg.norm(
                    D[j] - np.mean(D[j])) ** 3) * np.linalg.norm(D[k] - np.mean(D[k])))) * (D[j] - np.mean(D[j]))))
        return grad

    def nonnegative_matrix_factorization(self, A, SC, SD, r, alpha, beta, gamma, lr, max_iter, tol, diff_threshold):
        p, q = A.shape
        # C, D = self.random_init(p, q, r)
        C, D = self.svd_init(A, r)
        print(C.shape, D.shape)

        GS = self.laplacian(SC)
        GD = self.laplacian(SD)

        loss_list = []
        for iter in range(max_iter):
            loss = self.loss_func(A, C, D, SC, SD, alpha, beta, gamma, GS, GD)
            if iter % 10 == 0:
                loss_list.append(loss)
                if len(loss_list) >= 3:
                    diff = abs(loss_list[-3] - loss_list[-2]) + abs(loss_list[-2] - loss_list[-1])
                    if diff < diff_threshold:
                        break
            if tol > loss > 0:
                break
            for i in range(p):
                grad = self.grad_C(A, C, D, SC, alpha, beta, gamma, i, GS)
                C[i] = np.maximum(C[i] - lr * grad, 0)
            for j in range(q):
                grad = self.grad_D(A, C, D, SD, alpha, beta, gamma, j, GD)
                D[j] = np.maximum(D[j] - lr * grad, 0)

        return C, D


class Re_Matix:
    def __init__(self):
        pass

    def find_k_largest_elements(self, matrix, i, k):
        row_without_i = np.delete(matrix[i, :], i)
        largest_indices = np.argsort(row_without_i)[-k:]
        largest_elements = row_without_i[largest_indices]
        largest_indices = largest_indices + (largest_indices >= i)
        return largest_elements, largest_indices

    def process_matrix(self, new_circrna_disease_matrix, circ_sim_matrix, dis_sim_matrix, k):
        if k != 0:
            # 重构邻接矩阵
            # 水平重构
            A_R = np.zeros_like(new_circrna_disease_matrix)
            for i in range(new_circrna_disease_matrix.shape[0]):
                values_r, indices_r = self.find_k_largest_elements(circ_sim_matrix, i, k)
                sum_R = 0
                sum_v_r = 0
                for j in range(k):
                    # 最大的元素与他们对应的行向量乘积的和
                    sum_R = sum_R + values_r[j] * new_circrna_disease_matrix[indices_r[j], :]
                for v in range(k):
                    sum_v_r += values_r[v]
                A_R[i, :] = sum_R / sum_v_r
            # 垂直重构
            A_D = np.zeros_like(new_circrna_disease_matrix)
            for i in range(new_circrna_disease_matrix.shape[1]):
                values_d, indices_d = self.find_k_largest_elements(dis_sim_matrix, i, k)
                sum_D = 0
                sum_v_d = 0
                for j in range(k):
                    # 最大的元素与他们对应的列向量乘积的和
                    sum_D = sum_D + values_d[j] * new_circrna_disease_matrix[:, indices_d[j]]
                for v in range(k):
                    sum_v_d += values_d[v]
                A_D[:, i] = sum_D / sum_v_d

            A_RD = (A_D + A_R) / 2

            re_circrna_disease_matrix = np.zeros_like(new_circrna_disease_matrix)
            for i in range(new_circrna_disease_matrix.shape[0]):
                for j in range(new_circrna_disease_matrix.shape[1]):
                    re_circrna_disease_matrix[i, j] = max(A_RD[i][j], new_circrna_disease_matrix[i][j])
        else:
            re_circrna_disease_matrix = new_circrna_disease_matrix

        return re_circrna_disease_matrix
