import numpy as np
import matplotlib.pyplot as plt

# function


def f(x):
    ans = x**2
    ans *= np.sin(np.pi * x)
    ans += 0.7
    return(ans)

# generate data


def GenerateData():
    data = np.random.uniform(0, 1, (100, 2))
    class0 = []
    class1 = []
    for pt in data:
        if(f(pt[0]) > pt[1]):
            class0 += [pt]
        else:
            class1 += [pt]
    return(class0, class1)


def plot(class0, class1):
    plt.scatter([pt[0] for pt in class0], [pt[1]
                for pt in class0], s=2, c='pink')
    plt.scatter([pt[0] for pt in class1], [pt[1]
                for pt in class1], s=2, c='blue')
    plt.show()


def sig(x):
    ans = 1
    ans /= (1 + np.exp(-x))
    return(ans)


def forwardProp(T1, T2, x,numofnodes):
    a1 = np.array(x)
    a2 = sig(np.matmul(np.reshape(T1, (numofnodes,2) ) , a1 ))
    a3 = sig(np.matmul(np.reshape(T2, (1, numofnodes)), a2))
    return(a1, a2, a3)


def backwardProp(a2, a3, T2, y):
    g3 = a3 - y
    g2 = T2 * g3 * a2 * (1-a2)
    return(g2, g3)


def incrementGrad(grad, a1, a2, g2, g3,numofnodes):
    # TODO g2*a1 needs to resolve to a scalar

    grad[0] = grad[0] + \
        np.reshape(g2, (numofnodes,1)).dot(np.reshape(a1, (1, 2)))
    grad[1] = grad[1] + g3*a2

    return(grad)


def updateWeight(T1, T2, grad, alpha, numofnodes):

    T1 = T1 - (alpha*np.reshape(grad[0] , (2,numofnodes)  ))  # [0][0][0[]0[0][0]0[0[0]0[0]0[]]]
    T2 = T2 - (alpha*grad[1])
    return(T1, T2)


def epoch(x, T1, T2, deltas, y,numofnodes):
    a1, a2, a3 = forwardProp(T1, T2, x,numofnodes)

    g2, g3 = backwardProp(a2, a3, T2, y)
    grads = incrementGrad(deltas, a1, a2, g2, g3, numofnodes)
    return grads, g3


def formatData(class0, class1):
    data = []

    for pt in class0:
        data += [[pt[0], pt[1], 0]]

    for pt in class1:
        data += [[pt[0], pt[1], 1]]

    return data


if __name__ == '__main__':
    class0, class1 = GenerateData()
    # plot(class0,class1)

    numofnodes = 15



    # set weights
    T1 = np.random.rand(2, numofnodes)
    T2 = np.random.rand(1, numofnodes)
    data = formatData(class0, class1)

    grad = [0, 0]
    # training model
    eps = 0.05

    theta_prev = T2
    train_error = 0

    alpha = 0.1

    for i in range(1000):
        # 1epoch

        for inp in data:
            grad, train_error = epoch([inp[0], inp[1]], T1, T2, grad, inp[2],numofnodes)

        # update weights
        grad[0] *= 1/100
        grad[1] *= 1/100

        T1, T2 = updateWeight(T1, T2, grad, alpha,numofnodes)

        if (np.linalg.norm(T2 - theta_prev) < eps):
            break

        theta_prev = T2

    valid0, valid1 = GenerateData()
    validData = formatData(valid0, valid1)
    valid_error = 0

    valclass0 = []
    valclass1 = []

    for inp in validData:
        a1, a2, a3 = forwardProp(T1, T2, [inp[0], inp[1]],numofnodes)
        tmp, valid_error = backwardProp(a2, a3, T2, inp[2])

        if (a3 < 0.5):
            valclass0 += [a3]
        else:
            valclass1 += [a3]

    test0, test1 = GenerateData()
    testData = formatData(test0, test1)
    test_error = 0

    testclass0 = []
    testclass1 = []

    matrix = np.array([ [0,0], [0,0]])


    for inp in testData:
        a1, a2, a3 = forwardProp(T1, T2, [inp[0], inp[1]],numofnodes)
        tmp, test_error = backwardProp(a2, a3, T2, inp[2])

        if (a3 < 0.5):
            testclass0 += [a3]
            pred = 0
            actual = inp[2]
            matrix[pred][actual]+=1
        else:
            testclass1 += [a3]
            pred = 1
            actual = inp[2]
            matrix[pred][actual]+=1


    print("confusion matrix")
    print("\n")
    print(matrix)
    print("\n")
    print((train_error[0] + 1) -  (test_error[0]+1))