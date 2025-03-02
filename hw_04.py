# "As a member of the Kennesaw State University community of scholars, I understand that my
# actions are not only a reflection on myself, but also a reflection on the University and the larger
# body of scholars of which it is a part. Acting unethically, no matter how minor the offense, will
# be detrimental to my academic progress and self-image. It will also adversely affect all students,
# faculty, staff, the reputation of this University, and the value of the degrees it awards. Whether
# on campus or online, I understand that it is not only my personal responsibility, but also a duty
# to the entire KSU community that I act in a manner consistent with the highest level of academic
# integrity. Therefore, I promise that as a member of the Kennesaw State University community, I
# will not participate in any form of academic misconduct."

import math

import numpy as np

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
        return x,it

def GaussSeidelR(A, b, lam=1, es=1.e-7, maxit=50):
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
            x[i] = lam*x[i] + (1-lam)*xold[i] # adjust for any added relaxation
            if x[i] != 0:
                ea[i] = abs((x[i] - xold[i]) / x[i])  # compute change error
        if np.max(ea) < es:  # exit for loop if stopping criterion met
            break
    if it == maxit:  # check for maximum iteration exit
        return 'maximum iterations reached'
    else:
        return x,it, np.max(ea)


def question_1():
    print('Question 1')
    for n in [10, 100, 1000, 10000, 10000]:
        print('n = ', n)
        a = np.zeros((n, n))
        b = np.ones((n, 1))
        a = np.matrix(a)
        b = np.matrix(b)
        b[0, 0] = 2
        b[n-1, 0] = 2
        actual_sol = np.matrix(np.ones((n, 1)))

        # We construct the tridiagonal matrix
        a[0, 0] = 3
        a[0, 1] = -1
        for row in range(1, n-1):
            col = row - 1
            a[row, col] = -1
            a[row, col + 1] = 3
            a[row, col + 2] = -1
        a[n - 1, n - 2] = -1
        a[n - 1, n - 1] = 3
        print(f'A: \n{a[:3]}\n...\n{a[-3:]}')
        print(f'b: \n{b[:3]}\n...\n{b[-3:]}')

        (x, num_iters) = GaussSeidel(a, b, es=1.e-6, maxit=200)
        rel_error = np.linalg.norm(x - actual_sol, np.inf) / np.linalg.norm(actual_sol, np.inf)
        print(f'solution matrix: \n{x[:3]}\n...\n{x[-3:]}')
        print('Number of iterations = ', num_iters)
        print('Relative error: ', str(rel_error) + '\n')


def question_2_a():
    print('Question 2.a')
    for n in [10, 100, 200, 300, 400, 500]:
        print('n = ', n)
        a = np.zeros((n, n))
        x = np.ones((n, 1))
        a = np.matrix(a)
        x = np.matrix(x)

        for row in range(n):
            for col in range(n):
                entry = abs((row+1) - (col+1)) + 1
                a[row, col] = entry

        b = a*x

        sol = np.linalg.solve(a, b)
        forward_error = sol - x
        inf_norm_forward_error = np.linalg.norm(forward_error, np.inf)
        mag_factor = inf_norm_forward_error / np.linalg.norm(b, np.inf)
        cond_num = np.linalg.cond(a)

        print('Forward error infinity norm is\n', inf_norm_forward_error)
        print('Condition num is\n', cond_num)
        print(f'error magnification factor is\n{mag_factor}\n')


def question_2_b():
    print('Question 2.b')
    for n in [100, 200, 300, 400, 500]:
        print('n = ', n)
        a = np.zeros((n, n))
        x = np.ones((n, 1))
        a = np.matrix(a)
        x = np.matrix(x)

        for row in range(n):
            for col in range(n):
                entry = math.sqrt(((row + 1) - (col + 1))**2 + n/10)
                a[row, col] = entry

        b = a * x

        sol = np.linalg.solve(a, b)
        forward_error = sol - x
        inf_norm_forward_error = np.linalg.norm(forward_error, np.inf)
        mag_factor = inf_norm_forward_error / np.linalg.norm(b, np.inf)
        cond_num = np.linalg.cond(a)

        # print('Solution is\n', sol)
        print('Forward error infinity norm is\n', inf_norm_forward_error)
        print('Condition num is\n', cond_num)
        print(f'error magnification factor is\n{mag_factor}\n')


def question_3():
    a = np.matrix([[0.8, -0.4, 0, 0],
                   [-0.4, 0.8, -0.4, 0],
                   [0, -0.4, 0.8, -0.4],
                   [0, 0, 0.4, 0.8]])
    b = [[44],
         [27],
         [110],
         [84]]

    for l in [1, 0.8, 1.2]:
        print('relaxation = ', l)
        (x, num_iters, rel_error) = GaussSeidelR(a, b, l, es=1.e-1, maxit=50)
        print('x = ', x)
        print('num_iters = ', num_iters)
        print('rel_error = ', str(rel_error) + '\n')


def main():
    question_1()
    question_2_a()
    question_2_b()
    question_3()


if __name__ == '__main__':
    main()