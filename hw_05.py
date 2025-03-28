import math
import numpy as np
import matplotlib.pyplot as plt

# Code from lecture slides
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

def question_1():
    print('Question 1.)')
    x = np.array([5000, 5200, 6000, 6538, 7109, 7556, 8005, 8207, 8210, 8600, 9026, 9197, 9926, 10813, 13800, 14311])
    y = np.array(
        [2596.8, 3328.0, 3181.1, 3198.4, 4779.9, 5905.6, 5769.2, 8089.5, 4813.1, 5618.7, 7736.0, 6788.3, 7840.8,
         8882.5, 10489.5, 12506.6])

    # Part a.)
    a0, a1, Rsq, SE = strlinregr(x, y)
    print('Intercept = {0:7.2f}'.format(a0))
    print('Slope = {0:7.3f}'.format(a1))
    print('R-squared = {0:5.3f}'.format(Rsq))
    print('Standard error = {0:7.2f}'.format(SE))

    print(f'y = {a1:.3f}x + {a0:.3f}')

    # Part b.)
    extra_cost = a1*1000
    print('Extra cost = {0:7.2f}'.format(extra_cost))

    # Part c.)
    cost_10000 = a1*10000 + a0
    print('Depth of 10000 ft Cost = {0:7.2f}'.format(cost_10000))

    # Part d.)
    cost_20000 = a1 * 20000 + a0
    print('Depth of 20000 ft Cost = {0:7.2f}'.format(cost_20000))


def question_2():
    print('\nQuestion 2.)')
    x = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
    y = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])

    trans_y = np.log(y) - np.log(x)
    a0, a1, Rsq, SE = strlinregr(x, trans_y)
    #a = np.exp(a0)
    print('Intercept = {0:7.2f}'.format(a0))
    print('Slope = {0:7.3f}'.format(a1))
    print('R-squared = {0:5.3f}'.format(Rsq))
    print('Standard error = {0:7.2f}'.format(SE))

    B = a1
    a = np.exp(a0)
    y_est_model = a * x * np.exp(B * x)
    print(f'Fitted model after backtransforming: y = {a:.3f}*x*exp({B:.3f}*x)')

    plt.figure()
    plt.scatter(x, y, c='k', marker='s')
    plt.plot(x, y_est_model, c='k')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitted Model')
    plt.show()

    plt.figure()
    residuals = y - y_est_model
    plt.scatter(x, residuals, color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('Residuals Versus Predicted Y Values')
    plt.show()


def question_3():
    print('\nQuestion 3.)')
    x = np.array([0, 5, 10, 15, 20, 25, 30])
    y = np.array([14.6, 12.8, 11.3, 10.1, 9.09, 8.26, 7.56])

    p_1 = np.polyfit(x, y, 3)
    p_2 = np.polyfit(x, y, 6)

    results_1 = np.polyval(p_1,x)
    results_2 = np.polyval(p_2,x)
    print('Results from cubic polynomial fit:')
    print(results_1)
    print('Results from sextic polynomial fit:')
    print(results_2)


def question_4():
    # Question 4.)
    print('\nQuestion 4.)')
    x_1 = np.array([0.3, 0.6, 0.9, 0.3, 0.6, 0.9, 0.3, 0.6, 0.9])
    x_2 = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    y = np.array([0.04, 0.24, 0.69, 0.13, 0.82, 2.38, 0.31, 1.95, 5.66])

    # Define normal equations
    n = len(x_1)
    x_1_squared = x_1**2
    x_2_squared = x_2**2
    x_1_prod_x_2 = x_1*x_2

    x_1_prod_y = x_1*y
    x_2_prod_y = x_2*y

    x_1_sum = np.sum(x_1)
    x_2_sum = np.sum(x_2)
    x_1_sq_sum = np.sum(x_1_squared)
    x_2_sq_sum = np.sum(x_2_squared)
    x_1_prod_x_2_sum = np.sum(x_1_prod_x_2)

    y_sum = np.sum(y)
    x_1_prod_y_sum = np.sum(x_1_prod_y)
    x_2_prod_y_sum = np.sum(x_2_prod_y)

    # Construct matrix equation
    A = np.matrix([[n, x_1_sum, x_2_sum], [x_1_sum, x_1_sq_sum, x_1_prod_x_2_sum], [x_2_sum, x_1_prod_x_2_sum, x_2_sq_sum]])
    b = np.matrix([[y_sum], [x_1_prod_y_sum], [x_2_prod_y_sum]])    

    model_results = np.linalg.solve(A, b)
    b_0 = model_results[0,0]
    b_1 = model_results[1,0]
    b_2 = model_results[2,0]

    print(f'b_0 = {b_0}\nb_1 = {b_1}\nb_2 = {b_2}')
    print(f'y = {b_0:.2f} + {b_1:.2f}x_1 + {b_2:.2f}x_2')


def main():
    question_1()
    question_2()
    question_3()
    question_4()


if __name__ == '__main__':
    main()