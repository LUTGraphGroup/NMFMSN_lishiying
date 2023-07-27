import numpy as np

class MatrixFactorization:
    def __init__(self):
        pass
    def pearson_similarity(self,x, y):
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

    def random_init(self,m, n, k):
        # 随机生成非负的C和D矩阵作为初始值
        C = np.random.uniform(low=1, high=2, size=(m, k))
        D = np.random.uniform(low=1, high=2, size=(n, k))
        return C, D

    # 计算S的拉普拉斯矩阵L
    def laplacian(self,S):
        D = np.diag(np.sum(S, axis=1))
        L = D - S
        return L

    def loss_func(self,A, C, D, S, T, lambda1, lambda2, alpha, beta,gamma,GS,GD):
        p, q = A.shape

        #第二行是图正则
        loss = np.sum((A - np.dot(C, D.T)) ** 2) + lambda1 * np.sum(np.square(C)) + lambda2 * np.sum(np.square(D))\
             +gamma * (np.trace(np.dot(np.dot(C.T, GS), C)) + np.trace(np.dot(np.dot(D.T, GD), D)))
        for i in range(p):
            for j in range(i + 1, p):
                corr = self.pearson_similarity(C[i], C[j])
                loss += alpha * ((S[i, j] - corr) ** 2)
        for i in range(q):
            for j in range(i + 1, q):
                corr = self.pearson_similarity(D[i], D[j])
                loss += beta * ((T[i, j] - corr) ** 2)
        return loss / 2

    def grad_C(self,A, C, D, S, lambda1, alpha, i,gamma,GS):
        p, q = A.shape
        len = C.shape[1]
        grad = np.zeros(len)


        #加入了图正则项
        for ii in range(C.shape[0]):
            grad+=0.5*gamma*C[ii,:]*GS[i,ii]
        grad+=0.5*gamma*C[i,:]*GS[i,i]


        for j in range(q):
            grad += (np.dot(C[i], D[j]) - A[i, j]) * D[j]
        grad += lambda1 * C[i]
        for k in range(i + 1, p):
            corr = self.pearson_similarity(C[i], C[k])
            if np.isnan(corr):
                corr = 0
                # if np.linalg.norm(C[i] - np.mean(C[i])) != 0 and np.linalg.norm(C[i] - np.mean(C[i])) != 0:
                grad += alpha * (S[i, k] - corr) * (1 / (np.linalg.norm(C[k] - np.mean(C[k])))*(np.linalg.norm(C[i] - np.mean(C[i])))) * ((
                        (np.dot((C[i] - np.mean(C[i])), (C[k] - np.mean(C[k])))) / (
                        np.linalg.norm(C[i] - np.mean(C[i])) ** 2)) * (C[i] - np.mean(C[i]))-
                        (C[k] - np.mean(C[k])))
        return grad

    def grad_D(self,A, C, D, T, lambda2, beta, j,gamma,GD):
        p, q = A.shape
        len = D.shape[1]
        grad = np.zeros(len)



        #图正则化
        for ii in range(D.shape[0]):
            grad+=0.5*gamma*D[ii,:]*GD[j,ii]
        grad+=0.5*gamma*D[j,:]*GD[j,j]


        for i in range(p):
            grad += (np.dot(C[i], D[j]) - A[i, j]) * C[i]
        grad += lambda2 * D[j]
        for k in range(j + 1, q):
            corr = self.pearson_similarity(D[j], D[k])
            if np.isnan(corr):
                corr = 0
                # if np.linalg.norm(D[j] - np.mean(D[j])) != 0 and np.linalg.norm(D[j] - np.mean(D[j])) != 0:
                grad += beta * (T[j, k] - corr) * (1 / (np.linalg.norm(D[k] - np.mean(D[k])))*(np.linalg.norm(D[j] - np.mean(D[j])))) * ((
                        (np.dot((D[j] - np.mean(D[j])), (D[k] - np.mean(D[k])))) / (
                        np.linalg.norm(D[j] - np.mean(D[j])) ** 2)) * (D[j] - np.mean(D[j]))-
                        (D[k] - np.mean(D[k])))
        return grad

    def nonnegative_matrix_factorization(self,A, S, T, r, lambda1, lambda2, alpha, beta, lr, max_iter, tol,diff_threshold,gamma):
        p, q = A.shape
        # C, D = self.random_init(p, q, r)
        C, D = self.svd_init(A, r)
        print(C.shape, D.shape)


        GS=self.laplacian(S)
        GD=self.laplacian(T)

        loss_list = []
        for iter in range(max_iter):
            loss = self.loss_func(A, C, D, S, T, lambda1, lambda2, alpha, beta,gamma,GS,GD)
            if iter % 10 == 0:
                print("Iteration: {} Loss: {}".format(iter, loss))
                loss_list.append(loss)
                if len(loss_list) >= 3:
                    diff = abs(loss_list[-3] - loss_list[-2]) + abs(loss_list[-2] - loss_list[-1])
                    if diff < diff_threshold:
                        break
            if tol > loss > 0:
                break
            for i in range(p):
                grad = self.grad_C(A, C, D, S, lambda1, alpha, i,gamma,GS)
                C[i] = np.maximum(C[i] - lr * grad, 0)
            for j in range(q):
                grad = self.grad_D(A, C, D, T, lambda2, beta, j,gamma,GD)
                D[j] = np.maximum(D[j] - lr * grad, 0)

        return C, D
