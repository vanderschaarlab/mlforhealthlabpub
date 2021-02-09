import numpy as np
from sklearn.metrics import mean_squared_error



def compute_test_statistic_all_timesteps(treatment_replica, treatment_probability, max_sequence_length, predicted_mask):
    test_statistic_sequence = np.zeros(shape=(max_sequence_length,))

    for timestep in range(max_sequence_length):
        treatment_replica_timestep = np.squeeze(treatment_replica[:, timestep, :])
        mask_timestep = predicted_mask[:, timestep]

        no_treatment = np.where(treatment_replica_timestep == 0)
        treatment_probability_timestep = np.squeeze(treatment_probability[:, timestep, :])
        treatment_probability_timestep[no_treatment] = 1 - treatment_probability_timestep[no_treatment]
        treatment_log_probability_timestep = np.log(treatment_probability_timestep + 1e-5)

        treatment_log_probability_timestep = np.sum(treatment_log_probability_timestep, axis=1)
        treatment_log_probability_timestep = np.sum(treatment_log_probability_timestep * mask_timestep)

        if (np.sum(mask_timestep) == 0):
            test_statistic_sample_timestep = 0
        else:
            test_statistic_sample_timestep = treatment_log_probability_timestep / np.sum(mask_timestep)

        test_statistic_sequence[timestep] = test_statistic_sample_timestep
    return test_statistic_sequence



def compute_predictive_checks_eval_metric(p_values_over_time):
    max_timestep = np.max(np.where(p_values_over_time != 0)[0]) + 1
    p_values_over_time = p_values_over_time[:max_timestep]
    ideal_p_values = 0.5 * np.ones_like(p_values_over_time)
    mse = mean_squared_error(p_values_over_time, ideal_p_values)

    return mse, p_values_over_time
