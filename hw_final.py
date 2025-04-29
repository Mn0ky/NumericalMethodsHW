# "As a member of the Kennesaw State University community of scholars, I understand that my
# actions are not only a reflection on myself, but also a reflection on the University and the larger
# body of scholars of which it is a part. Acting unethically, no matter how minor the offense, will
# be detrimental to my academic progress and self-image. It will also adversely affect all students,
# faculty, staff, the reputation of this University, and the value of the degrees it awards. Whether
# on campus or online, I understand that it is not only my personal responsibility, but also a duty
# to the entire KSU community that I act in a manner consistent with the highest level of academic
# integrity. Therefore, I promise that as a member of the Kennesaw State University community, I
# will not participate in any form of academic misconduct."

import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
        sumxy = sumxy + x[i]*y[i]# "As a member of the Kennesaw State University community of scholars, I understand that my
# actions are not only a reflection on myself, but also a reflection on the University and the larger
# body of scholars of which it is a part. Acting unethically, no matter how minor the offense, will
# be detrimental to my academic progress and self-image. It will also adversely affect all students,
# faculty, staff, the reputation of this University, and the value of the degrees it awards. Whether
# on campus or online, I understand that it is not only my personal responsibility, but also a duty
# to the entire KSU community that I act in a manner consistent with the highest level of academic
# integrity. Therefore, I promise that as a member of the Kennesaw State University community, I
# will not participate in any form of academic misconduct."

import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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


def trap(func,a,b,n=100):
    """Composite trapezoidal rule quadrature
    Input:
    func = name of function to be integrated
    a,b = integration limits
    n = number of segments (default = 100)
    Output:
    I = estimate of integral
    """
    if b <= a: return 'upper bound must be greater than lower bound'
    x = a
    h = (b-a)/n
    s = func(a)

    for i in range(n-1):
        x = x + h
        s = s + 2*func(x)

    s = s + func(b)
    I = (b-a)*s/2/n
    return I


def problem_1():
    print('Problem 1.')
    temperature = np.array([80, 44.5, 30.0, 24.1, 21.7, 20.7])

    # Part a.
    temp_dt = np.gradient(temperature)
    print(f'dT/dt: {temp_dt}')

    # Part b.
    temp_diff = temperature - 20
    plt.plot(temp_diff, temp_dt) # dT/dt on the Y-axis and T-T_a on the X-axis
    plt.ylabel('dT/dt')
    plt.xlabel('T - T_a')
    plt.title('dT/dt versus T-T_a')
    plt.show()

    a0, a1, Rsq, SE = strlinregr(temp_diff, temp_dt)
    print(f'Slope (k) = {a1:.2f}')


def prob_2_func(t):
    return 1 / (3*(1 + t**(4/3)))
def problem_2():
    print('\nProblem 2.')
    auc = trap(prob_2_func, 0, 1, 10)
    print(f'Area under the curve: {auc:.5f}')


def problem_3():
    print('\nProblem 3.')
    a = 400
    alpha_deg = np.array([54.80, 54.06, 53.34])
    beta_deg = np.array([65.59, 64.59,  63.62])

    alpha = alpha_deg * (math.pi/180)
    beta = beta_deg * (math.pi/180)
    alpha_tan = np.tan(alpha)
    beta_tan = np.tan(beta)

    # Compute position for t at 9, 10, and 11 seconds
    x = a*beta_tan / (beta_tan - alpha_tan)
    y = a*beta_tan*alpha_tan / (beta_tan - alpha_tan)
    print(f'Horizontal distances: {x}')
    print(f'Vertical distances: {y}')

    # Approximate instantaneous speed at t=10 using central-finite difference.
    step_size = 2 # h must equal 2 so that the position at 9 and 11 can be used, i.e. h*0.5=1
    speed_x = (x[2] - x[0]) / step_size
    speed_y = (y[2] - y[0]) / step_size

    # Compute magnitude of the horz. and vertical speeds to get overall speed.
    total_speed = np.sqrt(speed_x**2 + speed_y**2)
    climbing_angle_rad = np.arctan2(speed_y, speed_x) # Use np.arctan2() for single arguments
    climbing_angle = climbing_angle_rad * (180/math.pi)
    print(f'Overall instantaneous speed at t=10s is approximately {total_speed:.2f} m/s')
    print(f'Climbing angle at t=10s is approximately {climbing_angle:.2f}°')


def prob_4_func(params):
    x_1, y_1, x_2, y_2 = params
    P = (x_1, y_1)
    Q = (x_2, y_2)
    point_a = (0, 0)
    point_b = (0.3, 1.6)
    point_c = (1.5, 1)
    point_d = (1.8, 0)
    const_dist = math.dist(point_a, point_b) + math.dist(point_b, point_c) + math.dist(point_c, point_d)

    # Objective function we're trying to minimize.
    return (const_dist + math.dist(P, point_a) +
            math.dist(P, point_b) +
            math.dist(P, Q) +
            math.dist(Q, point_c) +
            math.dist(Q, point_d))
def problem_4():
    print('\nProblem 4.')
    init_guess = np.array([.3, 1, 1.5, .7]) # Initial guess inside the area enclosed by the given region.
    result = minimize(prob_4_func, init_guess) # Minimize
    x_1_opt, y_1_opt, x_2_opt, y_2_opt = result.x
    print(f'x_1 = {x_1_opt:.2f}, y_1 = {y_1_opt:.2f}, x_2 = {x_2_opt:.2f}, y_2 = {y_2_opt:.2f}')


def main():
    problem_1()
    problem_2()
    problem_3()
    problem_4()


if __name__ == '__main__':
    main()

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


def trap(func,a,b,n=100):
    """Composite trapezoidal rule quadrature
    Input:
    func = name of function to be integrated
    a,b = integration limits
    n = number of segments (default = 100)
    Output:
    I = estimate of integral
    """
    if b <= a: return 'upper bound must be greater than lower bound'
    x = a
    h = (b-a)/n
    s = func(a)

    for i in range(n-1):
        x = x + h
        s = s + 2*func(x)

    s = s + func(b)
    I = (b-a)*s/2/n
    return I


def problem_1():
    print('Problem 1.')
    temperature = np.array([80, 44.5, 30.0, 24.1, 21.7, 20.7])

    # Part a.
    temp_dt = np.gradient(temperature)
    print(f'dT/dt: {temp_dt}')

    # Part b.
    temp_diff = temperature - 20
    plt.plot(temp_diff, temp_dt) # dT/dt on the Y-axis and T-T_a on the X-axis
    plt.ylabel('dT/dt')
    plt.xlabel('T - T_a')
    plt.title('dT/dt versus T-T_a')
    plt.show()

    a0, a1, Rsq, SE = strlinregr(temp_diff, temp_dt)
    print(f'Slope (k) = {a1:.2f}')


def prob_2_func(t):
    return 1 / (3*(1 + t**(4/3)))
def problem_2():
    print('\nProblem 2.')
    auc = trap(prob_2_func, 0, 1, 10)
    print(f'Area under the curve: {auc:.5f}')


def problem_3():
    print('\nProblem 3.')
    a = 400
    alpha_deg = np.array([54.80, 54.06, 53.34])
    beta_deg = np.array([65.59, 64.59,  63.62])

    alpha = alpha_deg * (math.pi/180)
    beta = beta_deg * (math.pi/180)
    alpha_tan = np.tan(alpha)
    beta_tan = np.tan(beta)

    # Compute position for t at 9, 10, and 11 seconds
    x = a*beta_tan / (beta_tan - alpha_tan)
    y = a*beta_tan*alpha_tan / (beta_tan - alpha_tan)
    print(f'Horizontal distances: {x}')
    print(f'Vertical distances: {y}')

    # Approximate instantaneous speed at t=10 using central-finite difference.
    step_size = 2 # h must equal 2 so that the position at 9 and 11 can be used, i.e. h*0.5=1
    speed_x = (x[2] - x[0]) / step_size
    speed_y = (y[2] - y[0]) / step_size

    # Compute magnitude of the horz. and vertical speeds to get overall speed.
    total_speed = np.sqrt(speed_x**2 + speed_y**2)
    climbing_angle_rad = np.arctan2(speed_y, speed_x) # Use np.arctan2() for single arguments
    climbing_angle = climbing_angle_rad * (180/math.pi)
    print(f'Overall instantaneous speed at t=10s is approximately {total_speed:.2f} m/s')
    print(f'Climbing angle at t=10s is approximately {climbing_angle:.2f}°')


def prob_4_func(params):
    x_1, y_1, x_2, y_2 = params
    P = (x_1, y_1)
    Q = (x_2, y_2)
    point_a = (0, 0)
    point_b = (0.3, 1.6)
    point_c = (1.5, 1)
    point_d = (1.8, 0)
    const_dist = math.dist(point_a, point_b) + math.dist(point_b, point_c) + math.dist(point_c, point_d)

    # Objective function we're trying to minimize.
    return (const_dist + math.dist(P, point_a) +
            math.dist(P, point_b) +
            math.dist(P, Q) +
            math.dist(Q, point_c) +
            math.dist(Q, point_d))
def problem_4():
    print('\nProblem 4.')
    init_guess = np.array([.3, 1, 1.5, .7]) # Initial guess inside the area enclosed by the given region.
    result = minimize(prob_4_func, init_guess) # Minimize
    x_1_opt, y_1_opt, x_2_opt, y_2_opt = result.x
    print(f'x_1 = {x_1_opt:.2f}, y_1 = {y_1_opt:.2f}, x_2 = {x_2_opt:.2f}, y_2 = {y_2_opt:.2f}')


def main():
    problem_1()
    problem_2()
    problem_3()
    problem_4()


if __name__ == '__main__':
    main()