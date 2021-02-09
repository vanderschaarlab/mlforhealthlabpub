from helper import *
from GRVS import GRV_S

n_samples, data_name  = 400, 'syn4'

x_list, a_list, y_list = data_load(n_samples=n_samples)
model = GRV_S(x_list, a_list)
data_test = data_load(n_samples=20000)

for _ in range(50):
    model.fit(rnn_X=x_list,treatments= a_list,rnn_Y=y_list, num_d=1)
    y_list_test_grvs = GRVS_final(data_test,data_name,model,print_=True)

model.destroy_graph()