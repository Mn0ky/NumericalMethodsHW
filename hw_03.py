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

import numpy as np

def question_1():
    print('Question 1.')
    # coefficient matrix
    A = np.matrix([
        [-9, 0, 3, 0, 0],
        [1, -1, 0, 0, 0],
        [0, 2, -9, 0, 0],
        [0, 1, 6, -9, 2],
        [5, 1, 0, 0, -6]
        ])

    # solution matrix
    b = np.matrix([
        [-120], # c_1
        [0],    # c_2
        [-350], # c_3
        [0],    # c_4
        [0]     # c_5
        ])

    print('Concentration values c_1 through c_5:')
    sol = np.linalg.solve(A, b)
    print(sol)

    # Verify mass balances by taking the abs difference of the LHS and RHS, which should result in 0.
    epsilon = 1e-13  # Account for floating point imprecision, use a small epsilon instead of actual 0.
    assert abs(120 + 3*sol[2,0] - (5*sol[0,0] + 4*sol[0,0])) < epsilon          # reactor 1
    assert abs(4*sol[0,0] - (1*sol[1,0] + 1*sol[1,0] + 2*1*sol[1,0])) < epsilon # reactor 2
    assert abs(350 + 2*sol[1,0] - (3*sol[2,0] + 6*sol[2,0])) < epsilon          # reactor 3
    assert abs(1*sol[1,0] + 6*sol[2,0] + 2*sol[4,0] - 9*sol[3,0]) < epsilon     # reactor 4
    assert abs(5*sol[0,0] + 1*sol[1,0] - (2*sol[4,0] + 4*sol[4,0])) < epsilon   # reactor 5

    print('All mass balances verified.')


def question_2():
    print('Question 2.')

    # coefficient matrix
    A = np.matrix([
        [-4, 1, 1, 0],
        [1, -4, 0, 1],
        [1, 0, -4, 1],
        [0, 1, 1, -4]
    ])

    # solution matrix
    b = np.matrix([
        [-175],
        [-125],
        [-75],
        [-25],
    ])

    print('Node temperature values T_11, T_12, T_21, T_22, respectively')
    sol = np.linalg.solve(A, b)
    print(sol)

    # Verify steady-state distributions by taking the abs difference of the LHS and RHS, which should result in 0.
    epsilon = 1e-13  # Account for floating point imprecision, use a small epsilon instead of actual 0.
    assert abs(100 - 4*sol[0,0] + sol[2,0] + 75 + sol[1,0]) < epsilon  # Node T_11
    assert abs(100 - 4*sol[1,0] + sol[3,0] + sol[0,0] + 25) < epsilon  # Node T_12
    assert abs(sol[0,0] - 4*sol[2,0] + 0 + 75 + sol[3,0]) < epsilon  # Node T_21
    assert abs(sol[1,0] - 4*sol[3,0] + 0 + sol[2,0] + 25) < epsilon  # Node T_22

    print('All steady-state distributions verified.')

def main():
    question_1()
    question_2()


if __name__ == '__main__':
    main()