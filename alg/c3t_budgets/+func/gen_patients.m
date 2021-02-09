function [patients] = gen_patients(T, arrive_rate)

arrive_dist = arrive_rate / sum(arrive_rate);

arrive_dist = [0, arrive_dist];
arrive_dist_cum = cumsum(arrive_dist);
patients_gen = rand(1, T);
pat_idx = 1:length(arrive_dist) - 1;
[~, inds] = histc(patients_gen, arrive_dist_cum);

patients = pat_idx(inds);
