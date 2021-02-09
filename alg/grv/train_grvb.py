from helper import *
from GRVB import GRV_B

exp_index, d_t_T, prob_matrix = 1, None, None
n_samples, data_name  = 400, 'syn4'

x_list, a_list, y_list = data_load(n_samples=n_samples)
data_test = data_load(n_samples=20000)
n_steps = np.shape(x_list)[1]

for time in range(1,n_steps+1)[::-1]:
    model = GRV_B(x_list, a_list, time=time)
    for _ in range(50):
        model.fit(rnn_X=x_list,treatments= a_list,rnn_Y=y_list,d_t_T=d_t_T, num_d=1)
        data_test, result_step = step_test(data_test, model, time, print_=True)

    prob_t, d_t = model.predict(rnn_X=x_list, treatments=a_list)
    d_t_T = d_concat(d_t, d_t_T, time,n_steps)
    model.save_network(data_name, exp_index, time, destroy=True)

data_test = data_load(n_samples=20000)
result = final_test(data_test,data_name, exp_index, print_=True)
