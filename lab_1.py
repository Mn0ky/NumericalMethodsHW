import math


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


def modified_secant(f, x0, delta, Ea=1.e-6, maxit=30):
    for i in range(maxit):
        x1 = x0 - f(x0)*(delta*x0) / (f(x0+delta) - f(x0))

        ea = abs((x1 - x0) / x1)
        if ea < Ea:
            break

        x0 = x1
    return x1, f(x1), ea, i + 1


# Q5.11a)
def tank_func(h):
    return (math.pi*h**2*(15-h))/3 - 300

def question_5_11_a():
    print('Question 5.11a')
    (xsoln, ea, n) = bisect1(tank_func, 0, 10, 1.e-13, 200)
    print(f'h = {xsoln}')


# Q5.11b)
def question_5_11_b():
    print('Question 5.11b')

    for V in range(0, 300, 10):
        variable_vol_tank_func = lambda h: (math.pi * h**2 * (15 - h))/3 - V
        (xsoln, ea, n) = bisect1(variable_vol_tank_func, 0, 10, 1.e-13, 200)
        print(f'h = {xsoln}\t at V = {V}')


# Q5.16)
def loan_formula(i):
    return 71991 * i*(i+1)**84/((1+i)**84 - 1) - 1000

def question_5_16():
    print('Question 5.16')
    (xsoln, ea, n) = bisect1(loan_formula, 0.03/12, 0.09/12, 1.e-13, 200)
    print(f'APR = {xsoln*1200}%')


# Q6.2)
def quad_func(x):
    return -0.9*x**2 + 1.7*x +2.5
def quad_func_dx(x):
    return -1.8*x + 1.7

def question_6_2():
    print('Question 6.2')
    # Applying fixed-point iteration method
    print('Running fixed-point iteration method...')
    (xsoln, ea, n) = wegstein(quad_func, 5, 6, Ea=1.e-2, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying the Newton-Raphson method
    print("Running Newton's method...")
    (xsoln, fxsoln, ea, n) = newtraph(quad_func, quad_func_dx, 5, Ea=1.e-2, maxit=100)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))


# Q6.4)
def small_root_func(x):
    return 7*math.sin(x) * math.e**(-x) - 1
def small_root_func_dx(x):
    return 7*math.cos(x)*math.e**(-x) + -math.e**(-x)*7*math.sin(x)

def question_6_4():
    print('Question 6.4')
    print('Running fixed-point iteration method...')
    (xsoln, ea, n) = wegstein(small_root_func, 0.3, 0.5, Ea=1.e-20, maxit=4)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying the Newton-Raphson method
    print("Running Newton's method...")
    (xsoln, fxsoln, ea, n) = newtraph(small_root_func, small_root_func_dx, 0.3, Ea=1.e-20, maxit=4)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))

    # Applying secant method
    print('Running modified-secant method...')
    (xsoln, fxsoln, ea, n) = modified_secant(small_root_func, 0.3, 0.001, Ea=1.e-20, maxit=4)
    print('Solution = {0:8.5g}'.format(xsoln))
    print('Relative error = {0:8.3e}'.format(ea))
    print('Number of iterations = {0:5d}\n'.format(n))


def main():
    question_5_11_a()
    question_5_11_b()
    question_5_16()
    question_6_2()
    question_6_4()

if __name__ == '__main__':
    main()

