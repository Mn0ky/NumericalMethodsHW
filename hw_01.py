"""
“As a member of the Kennesaw State University community of scholars, I
understand that my actions are not only a reflection on myself, but also a
reflection on the University and the larger body of scholars of which it is a part.
Acting unethically, no matter how minor the offense, will be detrimental to my
academic progress and self-image. It will also adversely affect all students, faculty,
staff, the reputation of this University, and the value of the degrees it awards.
Whether on campus or online, I understand that it is not only my personal
responsibility, but also a duty to the entire KSU community that I act in a manner
consistent with the highest level of academic integrity. Therefore, I promise that
as a member of the Kennesaw State University community, I will not participate in
any form of academic misconduct.”
"""
import math
import numpy as np


def secant(f, x0, x1, Ea=1.e-6, maxit=30):
    for i in range(maxit):
        x2 = x1 - f(x1)*(x1 - x0) / (f(x1) - f(x0))

        ea = abs((x2 - x1) / x2)
        if ea < Ea:
            break

        x0 = x1
        x1 = x2
    return x2, f(x2), ea, i + 1


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


def g_1(x):  # g(x) for use in the wegstein method (fixed-point iteration)
    return 2 / x

def g_2(x):  # g(x) for use in the wegstein method (fixed-point iteration)
    return x ** 2 + x - 2

def g_3(x):  # g(x) for use in the wegstein method (fixed-point iteration)
    return (x + 2) / (x + 1)


def bisect1(func, xl, xu, Ea=1.e-6, maxit=20):
    """
    Uses the bisection method to estimate a root of func(x).
    The method is iterated maxit (default = 20) times.
    Input:
        func = name of the function
        xl = lower guess
        xu = upper guess
    Output:
        xm = root estimate
        or
        error message if initial guesses do not bracket solution
    """
    xm_old = 0  # Default values to make interpreter happy
    ea = 0
    i = -1
    if func(xl) * func(xu) > 0:
        return 'initial estimates do not bracket solution'
    for i in range(maxit):
        xm = (xl + xu) / 2
        if func(xm) * func(xl) > 0:
            xl = xm
        else:
            xu = xm

        if i != 0:  # Don't want approximate error to be 0% because of the first iteration
            ea = abs((xm - xm_old) / xm)
            if ea < Ea:  break
        xm_old = xm

    return xm, ea, i + 1


# QUESTION 1.)
def matrix_func(x):
    A = np.matrix([[1, 2, 3, x], [4, 5, x, 6], [7, x, 8, 9], [x, 10, 11, 12]])
    # Sets up the root equation |A(x)| - 5000 = 0
    return np.linalg.det(A) - 5000.0


def question_1():
    print('Question #1:')
    # solution 1 from bisection method
    (xsoln, ea, n) = bisect1(matrix_func, 11, 12, Ea=1.e-09, maxit=200)
    A = np.matrix([[1, 2, 3, xsoln], [4, 5, xsoln, 6], [7, xsoln, 8, 9], [xsoln, 10, 11, 12]])
    det_1 = np.linalg.det(A)
    print(f'root1 value: {xsoln}\nComputed det(A) when x=root1: {det_1}')

    # solution 2 from bisection method
    (xsoln, ea, n) = bisect1(matrix_func, -17, -18, Ea=1.e-09, maxit=200)
    A = np.matrix([[1, 2, 3, xsoln], [4, 5, xsoln, 6], [7, xsoln, 8, 9], [xsoln, 10, 11, 12]])
    det_2 = np.linalg.det(A)
    print(f'root2 value: {xsoln}\nComputed det(A) when x=root1: {det_2}\n')


# QUESTION 2.)
def question_2():
    print('Question #2:')
    print('Going in order of equations a, b, c respectively...')
    (xsoln, ea, n) = wegstein(g_1, 1, 2)  # Note that x_1 != x_0
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    (xsoln, ea, n) = wegstein(g_2, 1, 2)
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    (xsoln, ea, n) = wegstein(g_3, 1, 2)
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))


# QUESTION 3.)
def bisect_test_func(x):
    return x ** 4 - 2
def secant_test_func(x):
    return x ** 4 - 2
def fixed_point_test_func(x):
    return (x / 2) + (1 / x ** 3)
def newton_test_func(x):
    return x ** 4 - 2


def newtons_test_func_dx(x):
    return 4 * x ** 3


def question_3():
    # Applying bisection method
    print('Running bisection method...')
    (xsoln, ea, n) = bisect1(bisect_test_func, 1, 2, Ea=1.e-6, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying secant method
    print('Running secant method...')
    (xsoln, fxsoln, ea, n) = secant(secant_test_func, 1, 2, Ea=1.e-6, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying fixed-point iteration method
    print('Running fixed-point iteration method...')
    (xsoln, ea, n) = wegstein(fixed_point_test_func, 1, 2, Ea=1.e-6, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying Newton's method
    print("Running Newton's method...")
    (xsoln, fxsoln, ea, n) = newtraph(newton_test_func, newtons_test_func_dx, 1, Ea=1.e-6, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))


# Question 4.)
def g4_func(x):
    return math.log(x) ** 2 - x - 1
def g4_func_dx(x):
    return (2 * math.log(x) - x) / x

def g4_func_dx_2(x):
    return (2 - 2*math.log(x)) / x**2

def largest_b_func(x):
    # x0 - g4_func(x0) / fp(x0) = 0
    return x - g4_func(x) / g4_func_dx(x)

def largest_b_func_dx(x):
    return 1 - (g4_func_dx(x)*g4_func_dx(x) - g4_func_dx_2(x)*g4_func(x)) / g4_func_dx(x)**2

def question_4():
    (xsoln, fxsoln, ea, n) = newtraph(largest_b_func, largest_b_func_dx, .5)
    print(f'Solution = {xsoln}')  # 0.6608598014068282


def main():
    question_1()
    question_2()
    question_3()
    question_4()


if __name__ == '__main__':
    main()
