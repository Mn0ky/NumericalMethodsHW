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

from scipy.optimize import minimize_scalar


def secant(f, x0, x1, Ea=1.e-6, maxit=30):
    for i in range(maxit):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

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
def easily_diff_func(x):
    return math.e ** (x / 2) - 5 * (1 - x)
def easily_diff_func_dx(x):
    return .5 * math.e ** (x / 2) + 5


def question_1():
    print('question 1...')
    (xsoln, ea, n) = bisect1(easily_diff_func, 0, 2, Ea=1.e-2, maxit=200)
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    (xsoln, fxsoln, ea, n) = newtraph(easily_diff_func, easily_diff_func_dx, .7, Ea=1.e-2, maxit=200)
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    (xsoln, fxsoln, ea, n) = secant(easily_diff_func, 0, 2, Ea=1.e-2, maxit=200)
    print(f'Solution = {xsoln}')
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))


# QUESTION 2.)
def parabolic_interpolation(f, x1, x2, x3, maxit=10):
    x4 = None
    i = -1

    for i in range(maxit):
        x4 = (x2 - (.5 * (x2 - x1) ** 2 * (f(x2) - f(x3)) - (x2 - x3) ** 2 * (f(x2) - f(x1)))
              / ((x2 - x1) * (f(x2) - f(x3)) - (x2 - x3) * (f(x2) - f(x1))))

        if (x3 > x4 > x2) or (x2 > x4 > x3):  # Check if x4 is between x3 and x2
            if f(x4) < f(x2):
                x1 = x2
                x2 = x4
            else:
                x3 = x4
        else:
            if f(x4) < f(x2):
                x3 = x2
                x2 = x4
            else:
                x1 = x4
    return x4, i+1

def find_max_func(x):
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4

def question_2():
    print('question 2...')
    (xsoln, numiter) = parabolic_interpolation(find_max_func, 1.75, 2, 2.5, maxit=10)
    print(f'Maximum at x = {xsoln}')
    print(f'Maximum is {find_max_func(xsoln)}')
    print(f'Number of iterations = {numiter}')


# QUESTION 3.)
def norm_dist(x):
    return math.e**(-x**2)
def norm_dist_neg_abs_dx(x):
    return -abs(math.e**(-x**2) * -2*x)

def question_3():
    print('question 3...')
    xmin = minimize_scalar(norm_dist_neg_abs_dx, bracket=(0, 2), method='golden')
    print(f'Inflection point at x={xmin.x}')
    print(f'Inflection point at y={norm_dist(xmin.x)}')

def main():
    question_1()
    question_2()
    question_3()

if __name__ == '__main__':
    main()
