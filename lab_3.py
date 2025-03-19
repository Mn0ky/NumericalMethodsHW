import numpy as np


def question_11_1():
    print('\nQuestion 11.1')
    A = np.matrix([[10, 2, -1], [-3, -6, 2], [1, 1, 5]])
    A_inv = np.linalg.inv(A)

    print(f'Matrix inverse:\n{A_inv}')
    identity_mat = A_inv*A
    print(f'Identity matrix verification:\n{identity_mat}')


def question_11_2():
    print('\nQuestion 11.2')
    A = np.matrix([[-8, 1, -2], [2, -6, -1], [-3, -1, 7]])
    A_inv = np.linalg.inv(A)
    print(f'Matrix inverse:\n{A_inv}')


def question_11_6():
    print('\nQuestion 11.6')
    A = np.matrix([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
    for row in A:
        max_val_index = np.argmax(row)
        min_val_index = np.argmin(row)

        # This gets the index of the abs max element
        if abs(row[0, max_val_index]) > abs(row[0, min_val_index]):
            abs_max_index = max_val_index
        else:
            abs_max_index = min_val_index

        # Scale to 1 based on sign of the row max
        if row[0, abs_max_index] < 0:
            row[0, abs_max_index] = -1
        else:
            row[0, abs_max_index] = 1
    print(f'Scaled matrix:\n{A}')

    A_fro = np.linalg.norm(A, 'fro')
    A_p_1 = np.linalg.norm(A, 1)
    A_p_inf = np.linalg.norm(A, np.inf)

    print(f'frobenius norm: {A_fro}')
    print(f'p=1 norm: {A_p_1}')
    print(f'p=inf norm: {A_p_inf}')


def question_11_8():
    print('\nQuestion 11.8')
    A = np.matrix([[1, 4, 9, 16, 25], [4, 9, 16, 25, 36], [9, 16, 25, 36, 49], [16, 25, 36, 49, 64], [25, 36, 49, 64, 81]])
    print(A)
    spectral_cond_num = np.linalg.cond(A, 2)
    print(f'spectral condition number: {spectral_cond_num}')

def question_11_9():
    print('\nQuestion 11.9a')
    A = np.matrix([[16, 4, 1], [4, 2, 1], [49, 7, 1]])
    row_sum_cond_num = np.linalg.cond(A, np.inf)
    print(f'row sum condition number: {row_sum_cond_num}')
    print('\nQuestion 11.9b')
    spectral_cond_num = np.linalg.cond(A, 2)
    fro_cond_num = np.linalg.cond(A, 'fro')
    print(f'spectral condition number: {spectral_cond_num}')
    print(f'frobenius condition number: {fro_cond_num}')


def question_11_13():
    print('\nQuestion 11.13a')
    A = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_inv = np.linalg.inv(A)
    A_cond = np.linalg.cond(A)

    print(f'Circular matrix inverse:\n{A_inv}')
    print(f'Circular matrix condition number:\n{A_cond}')

    print('\nQuestion 11.13b')
    A = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9.1]])
    A_inv = np.linalg.inv(A)
    A_cond = np.linalg.cond(A)
    print(f'Circular matrix inverse:\n{A_inv}')
    print(f'Circular matrix condition number:\n{A_cond}')

def main():
    question_11_1()
    question_11_2()
    question_11_6()
    question_11_8()
    question_11_9()
    question_11_13()



if __name__ == '__main__':
    main()