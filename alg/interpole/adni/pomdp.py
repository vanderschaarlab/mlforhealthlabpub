import numpy as np
import os

horizon = None

def write(S, A, Z, b0, T, O, R):

    with open('temp/temp.pomdp', 'w') as f:
        f.write('discount: 0.9\n')
        f.write('values: reward\n')

        f.write('states: {}\n'.format(S))
        f.write('actions: {}\n'.format(A))
        f.write('observations: {}\n'.format(Z))

        for s in range(S):
            for a in range(A):
                for s1 in range(S):
                    f.write('T: {} : {} : {} {}\n'.format(a, s, s1, T[s,a,s1]))
        for a in range(A):
            for s1 in range(S):
                for z in range(Z):
                    f.write('O: {} : {} : {} {}\n'.format(a, s1, z, O[a,s1,z]))
        for s in range(S):
            for a in range(A):
                f.write('R: {} : {} : * : * {}\n'.format(a, s, R[s,a]))

def solve(S, A, Z, b0, T, O, R):

    os.system('rm -f temp/*.alpha')
    os.system('rm -f temp/*.pg')
    write(S, A, Z, b0, T, O, R)
    cmd = './pomdp/src/pomdp-solve -stdout temp/temp.out -pomdp temp/temp.pomdp'
    cmd += ' -horizon {}'.format(horizon) if horizon is not None else ''
    os.system(cmd)

    alp = dict()
    for a in range(A):
        alp[a] = np.zeros((0,S))

    fs = [f for f in os.listdir('temp') if f[-6:] == '.alpha']
    with open('temp/{}'.format(fs[0]), 'r') as f:

        while True:
            try: a = int(f.readline())
            except ValueError: break
            val = np.array([float(x) for x in f.readline().split(' ')[:-1]])
            alp[a] = np.concatenate((alp[a], val[None,...]))
            f.readline()

    return alp
