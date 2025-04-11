import numpy as np

# Code is from textbook/lecture slides
def Newtint(x,y,xx):
    """
    Newtint: Newton interpolating polynomial
    Uses an (n−1)th−order Newton interpolating polynomial
    based on n data pairs to return a value of the
    dependent variable, yint, at a given value of the
    independent variable, xx.
    Input:
    x = array of independent variable values
    y = array of dependent variable values
    xx = value of independent variable at which
    the interpolation is calculated
    Output:
    yint = interpolated value of the dependent variable
    """

# compute the finite divided differences in the
# form of a difference table
    n = len(x)
    if len(y) != n:
        return 'x and y must be of same length'
    b = np.zeros((n,n))
    # assign the dependent variables to the first column of b
    b[:,0] = np.transpose(y)
    for j in range(1,n):
        for i in range(n-j):
            b[i,j] = (b[i+1,j-1]-b[i,j-1])/(x[i+j]-x[i])
    # use the finite divided differences to interpolate
    xt = 1
    yint = b[0,0]
    for j in range(n-1):
        xt = xt * (xx - x[j])
        yint = yint + b[0,j+1]*xt
    return yint


def question_17_5():
    print('Question 17.5')
    x_vals = np.array([1,2,3,5,6])
    y_vals = np.array([4.75, 4, 5.25, 19.75, 36])

    order_1 = Newtint(x_vals[2:4],y_vals[2:4],4)
    order_2 = Newtint(x_vals[1:4],y_vals[1:4],4)
    order_3 = Newtint(x_vals[1:], y_vals[1:], 4)
    order_4 = Newtint(x_vals, y_vals, 4)

    print(f'order_1 = {order_1}')
    print(f'order_2 = {order_2}')
    print(f'order_3 = {order_3}')
    print(f'order_4 = {order_4}')


def question_17_15():
    print('\nQuestion 17.15')
    x_vals = np.array([370, 382, 394, 406, 418])
    y_vals = np.array([5.9313, 7.5838, 8.8428, 9.796, 10.5311])

    coef = np.polyfit(x_vals, y_vals, 4)
    volume = np.polyval(coef, 400)
    print(f'volume = {volume:.2f} at 400 degrees celsius')


def question_21_12():
    print('\nQuestion 21.12')
    t_vals = np.array([0, 0.52, 1.04, 1.75, 2.37, 3.25, 3.83])
    y_vals = np.array([153, 185, 208, 249, 261, 271, 273])

    velocity_vals = np.gradient(y_vals, t_vals)
    print(f'Velocity Values = {velocity_vals}')
    acc_vals = np.gradient(velocity_vals)
    print(f'Acceleration Values = {acc_vals}')


def question_21_20():
    print('\nQuestion 21.20')
    x = np.array([0, 1, 5, 8])
    y = np.array([0, 1, 8, 16.4])

    coef = np.polyfit(x, y, 3)
    derivative_coeffs = np.polyder(coef)
    output_at_7 = np.polyval(derivative_coeffs, 7)
    print(f'flow rate at t=7s is approximately {output_at_7:.2f}cm^3/s')


def main():
    question_17_5()
    question_17_15()
    question_21_12()
    question_21_20()


if __name__ == '__main__':
    main()
