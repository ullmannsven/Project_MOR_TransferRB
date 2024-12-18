from pymor.basic import * 
import numpy as np 
from pymor.models.examples import penzl_example
from pymor.models.iosys import LTIModel
from pymor.reductors.basic import StationaryRBReductor
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.algorithms.pod import pod


fom_penzl = penzl_example()
id_operator = NumpyMatrixOperator(np.eye(fom_penzl.order), source_id='STATE', range_id='STATE')

s_param = ProjectionParameterFunctional('s', 1)
new_op = LincombOperator([id_operator, fom_penzl.A], [s_param, -1])
fom = StationaryModel(operator=new_op, rhs=fom_penzl.B)

parameter_space = fom.parameters.space(1,10)
training_parameters = parameter_space.sample_randomly(10)
snapshots = fom.solution_space.empty()

for mu in training_parameters: 
    #print(mu['s'])
    #v_test = (mu['s'][0] * id_operator - fom_penzl.A).apply_inverse(fom_penzl.B)
    v = fom.solve(mu)

    #print(v - v_test.norm())

    snapshots.append(v)

basis, sv = pod(snapshots, modes=10)

reductor = StationaryRBReductor(fom, basis)
rom = reductor.reduce()

test_set = parameter_space.sample_randomly(100)
errors = np.zeros(100)
comp_times_fom = np.zeros(100)
comp_times_rom = np.zeros(100)

import time

for idx, mu_test in enumerate(test_set): 
    tic = time.time()
    U_h = fom.solve(mu_test)
    comp_times_fom[idx] = time.time() - tic
    tac = time.time()
    U_N = reductor.reconstruct(rom.solve(mu_test))
    comp_times_rom[idx] = time.time() - tac
    errors[idx] = (U_h - U_N).norm()[0]

print(np.mean(errors))
print(np.sum(comp_times_fom))
print(np.sum(comp_times_rom))








