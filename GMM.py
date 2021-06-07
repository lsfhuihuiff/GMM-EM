import numpy as np
def Gussian(ndim, alpha,miu,sigma):#生成样本点
    seed = np.random.seed(2021)
    Y = []
    #miu = np.random.randint(1,10,(ndim))
    #sigma = np.random.randint(1,5,(ndim))
    for i in range(ndim):
        y = np.random.normal(miu[i],sigma[i],(int(alpha[i]*100),1))
        if i == 0:
            Y = y
        else:
            Y = np.append(Y,y)
    
    return Y

def phi(y, miu, sigma):
    
    p = (1./(np.sqrt(2*np.pi)*sigma))*(np.exp(-(y-miu)**2/(2*sigma**2)))
    return p

def GMM(Y, K, iter):
    N = Y.shape[0]
    seed = np.random.seed(2021)
    miu = np.random.rand(K)
    sigma = np.array([np.eye(1)] * K)
    alphas = np.array([1.0 / K] * K)#均分概率
    gamma = np.zeros((K,N))
    
    for s in range(iter):
        for i in range(K):
            for j in range(N):
                Sum = sum([alphas[t]*phi(Y[j],miu[t],sigma[t]) for t in range(K)])
                gamma[i][j] = alphas[i]*phi(Y[j],miu[i],sigma[i])/float(Sum)
                #print(Sum)
        oldmiu = miu
        oldsigma = sigma
        oldalphas = alphas
        # 更新 mu
        for i in range(K):
            miu[i] = np.sum(gamma[i]*Y)/np.sum(gamma[i])
        # 更新 sigma
        for i in range(K):
            sigma[i] = np.sqrt(np.sum(gamma[i]*(Y - miu[i])**2)/np.sum(gamma[i]))
        # 更新系数k
        for i in range(K):
            alphas[i] = np.sum(gamma[i])/N
            
#         if ((abs(miu - oldmiu)).sum() < EPS) and \
#         ((abs(alphas - oldalphas)).sum() < EPS) and \
#         ((abs(sigma - oldsigma)).sum() < EPS):
#             #print(Miu,sigma,alpha,it)
#             print(s)
#             break
    
    return alphas,miu,sigma

alpha = [0.3,0.3,0.4]
miu = [1,2,3]
sigma = [3,2,1]
print('真值alpha：',alpha)
print('真值miu：',miu)
print('真值sigma：',sigma)
Y = Gussian(3,alpha,miu,sigma)

alphas,mius,sigmas = GMM(Y, 3, 15)

print('模型alpha:',alphas)
print('模型miu:',mius)
print('模型sigma:',sigmas)