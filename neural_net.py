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


def forwardProp(T1, T2, x):
    a1 = np.insert(x, 0, 1)
    a2 = sig(np.insert(np.matmul(T1, a1), 0, 1))
    a3 = sig(np.matmul(np.reshape(T2, (1, 3)), a2))
    return(a1, a2, a3)


def backwardProp(a2, a3, T2, y):
    g3 = a3 - y
    g2 = T2 * g3 * a2 * (1-a2)
    return(g2, g3)


def incrementGrad(grad, a1, a2, g2, g3):
    # TODO g2*a1 needs to resolve to a scalar
    grad[0] = grad[0] + \
        g2.dot(np.reshape(a1, (3, 1)))
    grad[1] = grad[1] + g3*a2

    return(grad)


def updateWeight(T1, T2, grad, alpha):
    T1 = T1 - (alpha*grad[0])  # [0][0][0[]0[0][0]0[0[0]0[0]0[]]]
    T2 = T2 - (alpha*grad[1])
    return(T1, T2)


def epoch(x, T1, T2, deltas, y):
    a1, a2, a3 = forwardProp(T1, T2, x)

    g2, g3 = backwardProp(a2, a3, T2, y)
    grads = incrementGrad(deltas, a1, a2, g2, g3)
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
    # question 2
    #plot(class0, class1)

    # set weights
    T1 = np.random.rand(2, 3)
    T2 = np.random.rand(1, 3)
    data = formatData(class0, class1)

    # 3b
    a1, a2, a3 = forwardProp(T1, T2, [data[4][0], data[4][1]])
    print('Forward Propagation:')
    print('a1: ' + str(a1))
    print('a2: ' + str(a2))
    print('a3: ' + str(a3))

    # 3c
    g2, g3 = backwardProp(a2, a3, T2, data[4][2])
    print('\n')
    print('Backward Propagation:')
    print('g3: ' + str(g3))
    print('g2: ' + str(g2[0]))

    grad = [0, 0]

    # 3e
    grad = incrementGrad(grad, a1, a2, g2, g3)
    print('\n')
    print('Gradient Cumputation:')
    print('grad: ' + str(grad))

    # 3f
    print('\n')
    print('Original Weights:')
    print('Θ1: ' + str(T1))
    print('Θ2: ' + str(T2))
    T1, T2 = updateWeight(T1, T2, grad, 0.1)
    print('Updated Weights:')
    print('Θ1: ' + str(T1))
    print('Θ2: ' + str(T2))

    # training model
    # 3g
    eps = 0.05

    theta_prev = T2
    train_error = 0

    alpha = 0.1

    for i in range(1000):
        # 1epoch

        for inp in data:
            grad, train_error = epoch([inp[0], inp[1]], T1, T2, grad, inp[2])

        # update weights
        grad[0] *= 1/100
        grad[1] *= 1/100

        T1, T2 = updateWeight(T1, T2, grad, alpha)

        if (np.linalg.norm(T2 - theta_prev) < eps):
            break

        theta_prev = T2

    print('\n')
    print('Training Error: ' + str(train_error[0]+1))

    # 3h
    valid0, valid1 = GenerateData()
    validData = formatData(valid0, valid1)
    valid_error = 0

    valclass0 = []
    valclass1 = []

    # confusion matrix
    matr = np.array([[0, 0], [0, 0]])

    for inp in validData:
        a1, a2, a3 = forwardProp(T1, T2, [inp[0], inp[1]])
        tmp, valid_error = backwardProp(a2, a3, T2, inp[2])

        if (a3 < 0.5):
            valclass0 += [a3]
            # predicted a 0
            pred = 0
            actual = inp[2]
            matr[pred][actual] += 1
        else:
            valclass1 += [a3]
            pred = 1
            actual = inp[2]
            matr[pred][actual] += 1

    print('\n')
    print('Confusion Matrix:')
    print(matr)
    print('Training Error: ' + str(train_error[0]+1))
    print('Validation Error: ' + str(valid_error[0] + 1))
