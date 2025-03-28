import numpy as np
import scipy as sp


def GaussSeidel(A, b, es=1.e-7, maxit=50):
    """
    Implements the Gauss-Seidel method
    to solve a set of linear algebraic equations
    without relaxation
    Input:
    A = coefficient matris
    b = constant vector
    es = stopping criterion (default = 1.e-7)
    maxit = maximum number of iterations (default=50)
    Output:
    x = solution vector
    """
    n, m = np.shape(A)
    if n != m:
        return 'Coefficient matrix must be square'
    C = np.zeros((n, n))
    x = np.zeros((n, 1))
    for i in range(n):  # set up C matrix with zeros on the diagonal
        for j in range(n):
            if i != j:
                C[i, j] = A[i, j]
    d = np.zeros((n, 1))
    for i in range(n):  # divide C elements by A pivots
        C[i, 0:n] = C[i, 0:n] / A[i, i]
        d[i] = b[i] / A[i, i]
    ea = np.zeros((n, 1))
    xold = np.zeros((n, 1))
    for it in range(maxit):  # Gauss-Seidel method
        for i in range(n):
            xold[i] = x[i]  # save the x's for convergence test
        for i in range(n):
            x[i] = d[i] - C[i, :].dot(x)  # update the x's 1-by-1
            if x[i] != 0:
                ea[i] = abs((x[i] - xold[i]) / x[i])  # compute change error
        if np.max(ea) < es:  # exit for loop if stopping criterion met
            break
    if it == maxit:  # check for maximum iteration exit
        return 'maximum iterations reached'
    else:
        return x


def wegstein(g, x0, x1, Ea=1.e-7, maxit=30):
    """
    This function solves x=g(x) using the Wegstein method.
    The method is repeated until either the relative error
    falls below Ea (default 1.e-7) or reaches maxit (default 30).
    Input:
        g = name of the function for g(x)
        x0 = first initial guess for x
        x1 = second initial guess for x
        Ea = relative error threshold
        maxit = maximum number of iterations
    Output:
        x2 = solution estimate
        ea = relative error
        i+1 = number of iterations
    """
    for i in range(maxit):
        x2 = (x1 * g(x0) - x0 * g(x1)) / (x1 - x0 - g(x1) + g(x0))
        ea = abs((x1 - x0) / x1)
        if ea < Ea:  break
        x0 = x1
        x1 = x2
    return x2, ea, i + 1


def newtraph(f, fp, x0, Ea=1.e-7, maxit=30):
    """
    This function solves f(x)=0 using the Newton-Raphson method.
    The method is repeated until either the relative error
    falls below Ea (default 1.e-7) or reaches maxit (default 30).
    Input:
        f = name of the function for f(x)
        fp = name of the function for f'(x)
        x0 = initial guess for x
        Ea = relative error threshold
        maxit = maximum number of iterations
    Output:
        x1 = solution estimate
        f(x1) = equation error at solution estimate
        ea = relative error
        i+1 = number of iterations
    """
    for i in range(maxit):
        x1 = x0 - f(x0) / fp(x0)
        ea = abs((x1 - x0) / x1)
        if ea < Ea:  break
        x0 = x1
    return x1, f(x1), ea, i + 1


def strlinregr(x,y):
    n = len(x)
    if len(y) != n: return 'x and y must be of same length'
    sumx = np.sum(x)
    xbar = sumx/n
    sumy = np.sum(y)
    ybar = sumy/n
    sumsqx = 0
    sumxy = 0

    for i in range(n):
        sumsqx = sumsqx + x[i]**2
        sumxy = sumxy + x[i]*y[i]

    a1 = (n*sumxy - sumx*sumy)/(n*sumsqx - sumx**2)
    a0 = ybar - a1*xbar
    e = np.zeros(n)
    SST = 0
    SSE = 0

    for i in range(n):
        e[i] = y[i] - (a0+a1*x[i])
        SST = SST + (y[i] - ybar)**2
        SSE = SSE + e[i]**2

    SSR = SST - SSE
    Rsq = SSR/SST
    SE = np.sqrt(SSE/(n-2))

    return a0, a1, Rsq, SE


def question_12_3():
    print('Question 12.3')
    A = np.matrix([[10, 2, -1], [-3, -6, 2], [1, 1, 5]])
    b = np.matrix([[27], [-61.5], [-21.5]])

    sol = GaussSeidel(A, b, 0.05)
    print('Solution is\n', sol)

def cool_func(x):
    return -x**2 - x**2/(1+5*x) + x + 0.75
def cool_func_dx(x):
    return -2*x + 1 + (-2*x*(1+5*x)+5*x**2)/(1+5*x)**2
def question_12_15():
    print('\nQuestion 12.4')

    # Applying fixed-point iteration method
    print('Running fixed-point iteration method...')
    (xsoln, ea, n) = wegstein(cool_func, 1.2, 1.4, Ea=1.e-7, maxit=50)
    print('Root at x = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying Newton-Raphson method
    print('Running Newton-Raphson method...')
    (xsoln, f_x, ea, n) = newtraph(cool_func, cool_func_dx, 1.2, Ea=1.e-7, maxit=50)
    print('Root at x = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying SciPy.optimize method
    print('Running SciPy.optimize method...')
    result = sp.optimize.minimize_scalar(cool_func, bracket=(1.2, 1.3))
    print('Root at x = {0:8.5g}'.format(result.x))
    print('Number of iterations = {0:5d}\n'.format(result.nit))


def question_14_5():
    print('\nQuestion 14.5')
    x_vals = [0, 2, 4, 6, 9, 11, 12, 15, 17, 19]
    y_vals = [5, 6, 7, 6, 9, 8, 8, 10, 12, 12]
    a0, a1, Rsq, SE = strlinregr(x_vals, y_vals)
    print(f'Slope = {a1:.2f}')
    print(f'Intercept = {a0:.2f}')
    print(f'R^2 = {Rsq:.2f}')
    print(f's_e = {SE:.2f}')
    print('\nSwitching the two variables...')
    a0, a1, Rsq, SE = strlinregr(y_vals, x_vals)
    print(f'Slope = {a1:.2f}')
    print(f'Intercept = {a0:.2f}')
    print(f'R^2 = {Rsq:.2f}')
    print(f's_e = {SE:.2f}')


def question_14_9():
    print('\nQuestion 14.9a')
    x_vals = [4, 8, 12, 16, 20, 24]
    y_vals = [1600, 1320, 1000, 890, 650, 560]
    log_y_vals = np.log(y_vals)

    a0, a1, Rsq, SE = strlinregr(x_vals, log_y_vals)
    beta = a1
    alpha = np.exp(a0)
    print('Backcomputed alpha = {0:.2f}'.format(alpha))
    print('Backcomputed beta = {0:.5f}'.format(beta))
    print(f'Model is: y = {alpha:.2f}*exp({beta:.5f}x)')

    print('\nQuestion 14.9b')
    t = (np.log(200) - a0) / beta
    print(f'Time at CFU=200 will be t={t:.2f}')


def question_14_19():
    print('\nQuestion 14.19')
    x_vals = np.array([-50, -30, 0, 60, 90, 110])
    y_vals = np.array([1250, 1280, 1350, 1480, 1580, 1700])
    log_y_vals = np.log(y_vals)
    a0, a1, Rsq, SE = strlinregr(x_vals, log_y_vals)

    beta = a1
    alpha = np.exp(a0)
    print('Backcomputed alpha = {0:.2f}'.format(alpha))
    print('Backcomputed beta = {0:.5f}'.format(beta))
    print(f'Model is: y = {alpha:.2f}*exp({beta:.5f}x)')

    cap_at_30 = alpha*np.exp(beta*30)
    print(f'Model at 30C = {cap_at_30:.2f}')

def main():
    question_12_3()
    question_12_15()
    question_14_5()
    question_14_9()
    question_14_19()

if __name__ == '__main__':
    main()