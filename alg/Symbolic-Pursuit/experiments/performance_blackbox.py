import numpy as np
import sys
import pickle
import os
sys.path.append(os.path.abspath('..'))
from symbolic_pursuit.models import SymbolicRegressor
from datasets.data_loader_UCI import data_loader, mixup
from sklearn.metrics import mean_squared_error
from experiments.train_model import train_model


black_box = sys.argv[1]
data_name = sys.argv[2]
n_model = 5
data_list = []  # List containing the data used in each experiment
random_list = []
print("Welcome to this experiment evaluating the performance of symbolic modeling. \n"
      + "This experiment uses the black-box " + black_box + " on the dataset " + data_name + ".")
print(100*"%" + "\n" + 100*"%" + "\n" + "Now building the black-box models.\n" + 100*"%" + "\n" + 100*"%")
for n in range(n_model):
    X_train, y_train, X_test, y_test = data_loader(data_name)
    data_list.append([X_train, y_train, X_test, y_test])
    X_random = mixup(X_train)
    random_list.append(X_random)

model_list = []  # List containing each model
for n in range(n_model):
    print("Now working on model", n + 1, "/", n_model)
    X_train, y_train, _, _ = data_list[n]
    model = train_model(X_train, y_train, black_box=black_box)
    model_list.append(model)

symbolic_list = []  # List containing each symbolic model
print(100*"%" + "\n" + 100*"%" + "\n" + "Now building the symbolic models.\n" + 100*"%" + "\n" + 100*"%")
for n in range(n_model):
    print("Now working on model", n+1, "/", n_model)
    X_random = random_list[n]
    model = model_list[n]
    symbolic_model = SymbolicRegressor(verbosity=False)
    symbolic_model.fit(model.predict, X_random)
    symbolic_list.append(symbolic_model)

print(100*"%" + "\n" + 100*"%" + "\n" + "Now computing the statistics.\n" + 100*"%" + "\n" + 100*"%")
model_errors = []  # Generalization MSE of the Black-Box
symbolic_errors = []  # Generalization MSE of the symbolic-Model
distance_symbolicBB = []  # Generalization Distance between the symbolic-Model and the BB
symbolic_nterms = []  # Number of terms of the symbolic Model


for n in range(n_model):
    _, _, X_test, y_test = data_list[n]
    model_errors.append(mean_squared_error(y_test, model_list[n].predict(X_test)))
    symbolic_errors.append(mean_squared_error(y_test, symbolic_list[n].predict(X_test)))
    symbolic_nterms.append(len(symbolic_list[n].terms_list))
    distance_symbolicBB.append(mean_squared_error(model_list[n].predict(X_test),
                                                  symbolic_list[n].predict(X_test)))

model_avgMSE, model_stdMSE = np.average(model_errors), np.std(model_errors)
symbolic_avgMSE, symbolic_stdMSE = np.average(symbolic_errors), np.std(symbolic_errors)
symbolic_avgNterms, symbolic_stdNterms = np.average(symbolic_nterms), np.std(symbolic_nterms)
symbolic_avgDist, symbolic_stdDist = np.average(distance_symbolicBB), np.std(distance_symbolicBB)

# Print and save the results

output_file = open("experiments/" + black_box + "_" + data_name + ".txt", "w")
output_file.write(100*"=" + "\n")
print("Black-Box generalization MSE", model_avgMSE, "+/-", model_stdMSE)
output_file.write("Black-Box generalization MSE: " + str(model_avgMSE) + " +/- " + str(model_stdMSE) + "\n")
print("Symbolic generalization MSE", symbolic_avgMSE, "+/-", symbolic_stdMSE)
output_file.write("Symbolic generalization MSE: " + str(symbolic_avgMSE)
                  + " +/- " + str(symbolic_stdMSE) + "\n")
print("Generalization distance between the Symbolic model and the Black-Box: ", symbolic_avgDist,
      "+/-", symbolic_stdDist)
output_file.write("Generalization distance between the Symbolic model and the Black-Box: "
                  + str(symbolic_avgDist) + " +/- " + str(symbolic_stdDist) + "\n")
print("Training Symbolic Model number of terms", symbolic_avgNterms, "+/-", symbolic_stdNterms)
output_file.write("Training Symbolic Model number of terms: " + str(symbolic_avgNterms) + " +/- "
                  + str(symbolic_stdNterms) + "\n")

symbolic_bestID = int(np.argmin(symbolic_errors))
symbolic_worstID = int(np.argmax(symbolic_errors))
print(100*'%')
print("Best Symbolic Model:", symbolic_list[symbolic_bestID].get_expression())
symbolic_list[symbolic_bestID].print_projections()
output_file.write("Best Symbolic Model: " + str(symbolic_list[symbolic_bestID]) + "\n"
                  + str(symbolic_list[symbolic_bestID].string_projections()))
print("Associated generalization loss: ", symbolic_errors[symbolic_bestID])
output_file.write("Associated loss: " + str(symbolic_errors[symbolic_bestID]) + "\n")
print("Worst Symbolic Model:", symbolic_list[symbolic_worstID].get_expression())
symbolic_list[symbolic_worstID].print_projections()
output_file.write("Worst Symbolic Model: " + str(symbolic_list[symbolic_worstID]) + "\n"
                  + str(symbolic_list[symbolic_worstID].string_projections()))
print("Associated generalization loss: ", symbolic_errors[symbolic_worstID])
output_file.write("Associated loss: " + str(symbolic_errors[symbolic_worstID]) + "\n")
print(100*'%')
output_file.close()

# Save everything

with open("experiments/" + black_box + "_" + data_name + ".pickle", 'wb') as filename:
    save_tuple = (model_list, symbolic_list, model_errors, symbolic_errors, symbolic_nterms,
                  data_list, random_list)
    pickle.dump(save_tuple, filename)

