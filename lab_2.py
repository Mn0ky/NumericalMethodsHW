import math
import numpy as np
import scipy as sp


# Q8.5
def question_8_5():
    A = np.matrix([[complex(3, 2), 4], [complex(0, -1), 1]])
    b = np.matrix([[complex(2, 1)], [3]])

    x = np.linalg.inv(A) * b
    print(f'x: {x}')


def question_9_10():
    A = np.matrix([[.55, .25, .25], [.30, .45, .20], [.15, .30, .55]])
    b = np.matrix([[4800], [5800], [5700]])

    x = np.linalg.solve(A, b)
    print(f'x: {x}')


def question_10_10():
    A = np.matrix([[-1, -2, 5], [3, -2, 1], [2, 6, -4]])
    b = np.matrix([[-26], [-10], [44]])

    P, L, U = sp.linalg.lu(A)
    d = np.linalg.solve(L, b)
    x = np.linalg.solve(U, d)
    print(f'P: {P}')
    print(f'x: {x}')


def question_10_13():
    A = np.matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 1]])
    U = sp.linalg.cholesky(A)
    print(f'U: {U}')
    print(f'A: {U.transpose().dot(U)}')


def main():
    question_8_5()
    question_9_10()
    question_10_10()
    question_10_13()


if __name__ == '__main__':
    main()