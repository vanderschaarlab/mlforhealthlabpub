%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of C3T-Budgets
% 'Contextual constrained learning for dose-finding clinical trials'
% AISTATS 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

rng('default') % for reproducibility

rep = 500; % number of simulated trials
K = 6; % number of doses
B = 400; % budget
T = 1200; % time-horizon

tox_thre = 0.35; % toxicity threshold
eff_thre = 0.2; % efficacy threshold

S = 3; % num of subgroups
arr_rate = [5, 4, 3];

% p q true values
p_true = [0.01, 0.01, 0.05, 0.15, 0.20, 0.45; ...
          0.01, 0.05, 0.15, 0.20, 0.45, 0.60; ...
          0.01, 0.05, 0.15, 0.20, 0.45, 0.60];
q_true = [0.01, 0.02, 0.05, 0.10, 0.10, 0.10; ...
          0.10, 0.20, 0.30, 0.50, 0.60, 0.65; ...
          0.20, 0.50, 0.60, 0.80, 0.84, 0.85];

opt = [7; 4; 4]; % optimal doses

p_rec = zeros(S, T, rep);

alg_no = 2;
names = char('Algorithm', 'C3T-Budget', 'C3T-Budget-E');
out = struct('names', names, ...
    'rec', zeros(alg_no, S, K+1), ...
    'cum_eff', zeros(alg_no, B), ...
    'cum_tox', zeros(alg_no, B), ...
    'cum_s', zeros(alg_no, S, T, rep), ...
    'typeI', zeros(alg_no, S), ...
    'typeII', zeros(alg_no, S), ...
    'q_mse', zeros(alg_no, S, K, T));

for i = 1:rep
    % patients arrival generation
    pats = func.gen_patients(T, arr_rate);
    for tau = 1:T
        p_rec(pats(tau), tau:end, i) = p_rec(pats(tau), tau:end, i) + 1;
    end

    % C3T-Budget
    [rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse] = ...
        func.C3T_Budget(T, B, S, K, pats, arr_rate, tox_thre, eff_thre, p_true, q_true, opt);
    out.rec(1, :, :) = squeeze(out.rec(1, :, :)) + rec;
    out.cum_eff(1, :) = out.cum_eff(1, :) + cum_eff;
    out.cum_tox(1, :) = out.cum_tox(1, :) + cum_tox;
    out.cum_s(1, :, :, i) = squeeze(out.cum_s(1, :, :, i)) + cum_s;
    out.typeI(1, :) = out.typeI(1, :) + typeI;
    out.typeII(1, :) = out.typeII(1, :) + typeII;
    out.q_mse(1, :, :, :) = squeeze(out.q_mse(1, :, :, :)) + q_mse;

    % C3T-Budget-E
    [rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse] = ...
        func.C3T_Budget_E(T, B, S, K, pats, arr_rate, tox_thre, eff_thre, p_true, q_true, opt);
    out.rec(2, :, :) = squeeze(out.rec(2, :, :)) + rec;
    out.cum_eff(2, :) = out.cum_eff(2, :) + cum_eff;
    out.cum_tox(2, :) = out.cum_tox(2, :) + cum_tox;
    out.cum_s(2, :, :, i) = squeeze(out.cum_s(2, :, :, i)) + cum_s;
    out.typeI(2, :) = out.typeI(2, :) + typeI;
    out.typeII(2, :) = out.typeII(2, :) + typeII;
    out.q_mse(2, :, :, :) = squeeze(out.q_mse(2, :, :, :)) + q_mse;
end


% print results
rec_error = zeros(alg_no, S);
for i = 1:alg_no
    for s = 1:S
        k = 1:K + 1;
        k(opt(s)) = [];
        rec_error(i, s) = sum(out.rec(i, s, k)/rep);
    end
end
fprintf("== Recommended dose error rates ==============\n");
fprintf("%s |  SG1  |  SG2  |  SG3  | Total |\n", out.names(1, :));
for i = 1:alg_no
    fprintf("%s | %.3f | %.3f | %.3f | %.3f |\n", ...
        out.names(i+1, :), rec_error(i, 1), rec_error(i, 2), rec_error(i, 3), mean(rec_error(i, :)));
end
fprintf("\n");

typeI = mean(out.typeI/rep, 2);
typeII = mean(out.typeII/rep, 2);
fprintf("== Safe dose estimation error rates ===========\n");
fprintf("%s |  Type-I  |  Type-II |  Total   |\n", out.names(1, :));
for i = 1:alg_no
    fprintf("%s |  %.4f  |  %.4f  |  %.4f  |\n", ...
        out.names(i+1, :), typeI(i), typeII(i), (typeI(i) + typeII(i))/2);
end
fprintf("\n");

efficacy = out.cum_eff(:, end) / B / rep;
toxicity = out.cum_tox(:, end) / B / rep;
fprintf("== Efficacy and toxicity per patient =======\n");
fprintf("%s |   Efficacy   |   Toxicity   |\n", out.names(1, :));
for i = 1:alg_no
    fprintf("%s |    %.4f    |    %.4f    |\n", ...
        out.names(i+1, :), efficacy(i), toxicity(i));
end
fprintf("\n");

% plot
figure();
subplot(2, 3, 1);
plot(squeeze(mean(out.cum_s(1, 1, :, :), 4)), '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(mean(out.cum_s(2, 1, :, :), 4)), '-b', 'linewidth', 1.5);
plot(1:B, mean(p_rec(1, 1:B, :), 3)', '-k', 'linewidth', 1.5);
plot(B+1:T, mean(p_rec(1, B+1:T, :), 3)', '--k', 'linewidth', 1.5);
hold off;
legend({'C3T-Budget', 'C3T-Budget-E', '\pi_s'}, 'fontsize', 10); title('Subgroup 1')
set(gca, 'Box', 'on');
ylim([0, 400]);
ylabel('Number of patients');
grid on;
xlim([0, 1200]);
subplot(2, 3, 2);
plot(squeeze(mean(out.cum_s(1, 2, :, :), 4)), '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(mean(out.cum_s(2, 2, :, :), 4)), '-b', 'linewidth', 1.5);
plot(1:B, mean(p_rec(2, 1:B, :), 3)', '-k', 'linewidth', 1.5);
plot(B+1:T, mean(p_rec(2, B+1:T, :), 3)', '--k', 'linewidth', 1.5);
hold off;
set(gca, 'Box', 'on');
ylim([0, 400]);
ylabel('Number of patients');
grid on;
xlim([0, 1200]);
title('Subgroup 2')
subplot(2, 3, 3);
plot(squeeze(mean(out.cum_s(1, 3, :, :), 4)), '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(mean(out.cum_s(2, 3, :, :), 4)), '-b', 'linewidth', 1.5);
plot(1:B, mean(p_rec(3, 1:B, :), 3)', '-k', 'linewidth', 1.5);
plot(B+1:T, mean(p_rec(3, B+1:T, :), 3)', '--k', 'linewidth', 1.5);
hold off;
set(gca, 'Box', 'on');
ylim([0, 400]);
ylabel('Number of patients');
grid on;
xlim([0, 1200]);
title('Subgroup 3')

subplot(2, 3, 4);
plot(squeeze(mean(out.q_mse(1, 1, 1:5, :), 3))'/500, '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(mean(out.q_mse(2, 1, 1:5, :), 3))'/500, '-b', 'linewidth', 1.5);
hold off;
set(gca, 'Box', 'on');
legend({'C3T-Budget', 'C3T-Budget-E'}, 'fontsize', 10);
ylim([0, 0.1]);
ylabel('MSE');
grid on;
xlim([0, 1200]);
xlabel('Time');
subplot(2, 3, 5);
plot(squeeze(out.q_mse(1, 2, 4, :))'/500, '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(out.q_mse(2, 2, 4, :))'/500, '-b', 'linewidth', 1.5);
hold off;
set(gca, 'Box', 'on');
ylim([0, 0.1]);
ylabel('MSE');
grid on;
xlim([0, 1200]);
xlabel('Time');
subplot(2, 3, 6);
plot(squeeze(out.q_mse(1, 3, 4, :))'/500, '-r', 'linewidth', 1.5);
hold on;
plot(squeeze(out.q_mse(2, 3, 4, :))'/500, '-b', 'linewidth', 1.5);
hold off;
set(gca, 'Box', 'on');
ylim([0, 0.1]);
ylabel('MSE');
grid on;
xlim([0, 1200]);
xlabel('Time');
