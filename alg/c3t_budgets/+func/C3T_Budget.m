function [rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse] = ...
    C3T_Budget(T, B, S, K, pats, arr_rate, tox_thre, eff_thre, p_true, q_true, opt_ind)
% C3T_Budget  Simulated trial with C3T-Budget
%   Input
%     T        : time-horizon
%     B        : budget
%     S        : number of subgroups
%     K        : number of doses
%     pats     : generated patients arrivals
%     arr_rate : patients arrival rate
%     tox_thre : toxicity threshold
%     eff_thre : efficacy threshold
%     p_true   : toxicity probabilities
%     q_true   : efficacy probabilities
%     opt_ind  : MTD doses
%   Output
%     rec      : dose recommendation
%     cum_eff  : cumulative efficacy
%     cum_tox  : cumulative toxicity
%     cum_s    : cumulative recruitment of each subgroup
%     typeI    : type-I error for dose safety
%     typeII   : type-II error for dose safety
%     q_mse    : mean squared error of efficacy

% output
rec = zeros(S, K+1);
cum_s = zeros(S, T);
cum_eff = zeros(1, B);
cum_tox = zeros(1, B);
rec_err = zeros(1, S);
typeI = zeros(1, S);
typeII = zeros(1, S);
q_mse = zeros(S, K, T);


% define variables and parameters
arrive_dist = arr_rate / sum(arr_rate);

eff_temp = 0;
tox_temp = 0;

a0 = 1 / 2; % initialize a
a_max = 1;
c = 0.5;
delta = 1 / B * S * ones(1, S);

eff_w = 0.6;


% calculate dose level for each subgroup
d_func = @(x)1 / 2 .* log((1 + x)./(1 - x));
toxicity = @(x, a)((tanh(x) + 1) / 2).^a;
d = zeros(S, K);
for s = 1:S
    d(s, :) = d_func((p_true(s, :).^(1 / a0) .* 2. - 1));
end

t = 1; %trial
b = 0; %budget

s_arrive = zeros(1, S);
n_choose = zeros(S, K); %number of selection
p_hat = zeros(S, K); %estimated toxicity
q_bar = zeros(S, K); %estimated efficacy
q_hat = zeros(S, K); %estimated efficacy

alpha = zeros(1, S);
D = zeros(S, K);
a_hat = zeros(S, T); %estimated overall a
ak_hat = ones(S, K) * a0; % estimated individual a


% variables for expected improvement
q_a = ones(S, K);
q_b = ones(S, K);
q_ei = zeros(S, K);
q_ei_opt = zeros(S, K, T);
g_ei_p = zeros(S, T);


I = zeros(1, T);
X = zeros(1, T);
Y = zeros(1, T);
H = pats;
Z = zeros(1, T);

while t <= T && b < B
    for s = 1:S
        [~, I_est] = max(n_choose(s, :));
        a_hat(s, t) = ak_hat(s, I_est);
    end
    q_ei_opt(:, :, t) = q_ei;
    q_mse(:, :, t) = abs(q_true-q_bar).^2;

    curr_s = H(t);
    s_arrive(curr_s) = s_arrive(curr_s) + 1;

    if s_arrive(H(t)) < K + 1 % initialize
        I(t) = mod(s_arrive(H(t))-1, K) + 1;
        n_choose(curr_s, I(t)) = 1;
        X(t) = rand() <= q_true(curr_s, I(t));
        eff_temp = eff_temp + X(t);
        Y(t) = rand() <= p_true(I(t));
        tox_temp = tox_temp + Y(t);
        q_bar(curr_s, I(t)) = X(t);
        q_a(curr_s, I(t)) = q_a(curr_s, I(t)) + X(t);
        q_b(curr_s, I(t)) = q_b(curr_s, I(t)) + 1 - X(t);
        for k = 1:K
            q_hat(curr_s, k) = func.UCB_ind(q_bar(curr_s, k), c, n_choose(curr_s, k), sum(n_choose(curr_s, :)));
        end
        q_ei(curr_s, I(t)) = func.EI(q_a(curr_s, I(t)), q_b(curr_s, I(t)));
        p_hat(curr_s, I(t)) = Y(t);
        Z(t) = 1;
        b = b + 1;
        cum_eff(b) = eff_temp;
        cum_tox(b) = tox_temp;
    else
        rho = (B - b) / (T - t + 1);
        table = zeros(S, 4);
        for s = 1:S
            alpha(s) = func.alpha_func(d(s, :), K, delta(s), sum(n_choose(s, :))); % calculate alpha
            D(s, :) = toxicity(d(s, :), a_hat(s, t)) <= tox_thre; % available set

            [Ix, Ii] = max(q_hat(s, :).*D(s, :));
            if q_bar(s, Ii) * eff_w + Ix * (1 - eff_w) >= eff_thre
                table(s, 1) = q_ei(s, Ii);
            else
                table(s, 1) = 0;
            end
            g_ei_p(s, t) = q_ei(s, Ii);
            table(s, 2) = q_hat(s, Ii);
        end
        table(:, 3) = 1:S;
        table(:, 4) = arrive_dist;
        table = sortrows(table, [1, 2], 'descend');

        tilde_s = 1;
        tmp_dist = table(tilde_s, 4);
        while tmp_dist <= rho && tilde_s < S
            tilde_s = tilde_s + 1;
            tmp_dist = tmp_dist + table(tilde_s, 4);
        end
        tilde_s = tilde_s - 1;

        % LP solution
        dose_prob = zeros(S, 1);
        if tilde_s > 0
            dose_prob(table(1:tilde_s, 3)) = 1;
            dose_prob(table(tilde_s+1, 3)) = (rho - sum(table(1:tilde_s, 4))) / table(tilde_s+1, 4);
            if tilde_s + 2 <= S
                dose_prob(table(tilde_s+2:end, 3)) = 0;
            end
        end

        % determine to skip or not
        if dose_prob(curr_s) >= rand() % take dose
            [~, Ii] = max(q_hat(curr_s, :).*D(curr_s, :));
            I(t) = Ii(1);
            X(t) = rand() <= q_true(curr_s, I(t));
            eff_temp = eff_temp + X(t);
            Y(t) = rand() <= p_true(curr_s, I(t));
            tox_temp = tox_temp + Y(t);
            q_bar(curr_s, I(t)) = (q_bar(curr_s, I(t)) * n_choose(curr_s, I(t)) + X(t)) / (n_choose(curr_s, I(t)) + 1);
            p_hat(curr_s, I(t)) = (p_hat(curr_s, I(t)) * n_choose(curr_s, I(t)) + Y(t)) / (n_choose(curr_s, I(t)) + 1);
            n_choose(curr_s, I(t)) = n_choose(curr_s, I(t)) + 1;
            Z(t) = 1;
            b = b + 1;
            q_a(curr_s, I(t)) = q_a(curr_s, I(t)) + X(t);
            q_b(curr_s, I(t)) = q_b(curr_s, I(t)) + 1 - X(t);
            for k = 1:K
                q_hat(curr_s, k) = func.UCB_ind(q_bar(curr_s, k), c, n_choose(curr_s, k), sum(n_choose(curr_s, :)));
            end
            q_ei(curr_s, I(t)) = func.EI(q_a(curr_s, I(t)), q_b(curr_s, I(t)));
            cum_eff(b) = eff_temp;
            cum_tox(b) = tox_temp;
        else % skip patient
            X(t) = -1;
            Y(t) = -1;
            Z(t) = 0;
        end

    end
    if I(t) ~= 0
        ak_hat(curr_s, I(t)) = log(p_hat(curr_s, I(t))) / log((tanh(d(curr_s, I(t))) + 1)/2);
        if ak_hat(curr_s, I(t)) > a_max
            ak_hat(curr_s, I(t)) = a_max;
        end
    end
    t = t + 1;
end


% recommendation and observe results
a_hat_fin = zeros(1, S);
p_out = zeros(S, K);

for s = 1:S
    [~, I_est] = max(n_choose(s, :));
    a_hat_fin(s) = ak_hat(s, I_est);
    p_out(s, :) = toxicity(d(s, :), a_hat_fin(s));
    [Ix, Ii] = max(q_bar(s, :).*(p_out(s, :) <= tox_thre));
    if Ix >= eff_thre
        rec(s, Ii) = 1;
    else
        rec(s, K+1) = 1;
        Ii = K + 1;
    end
    if Ii ~= opt_ind(s)
        rec_err(s) = 1;
    end
    for i = 1:K
        if p_true(s, i) <= tox_thre && p_out(s, i) > tox_thre
            typeI(s) = typeI(s) + 1;
        else
            if p_true(s, i) > tox_thre && p_out(s, i) <= tox_thre
                typeII(s) = typeII(s) + 1;
            end
        end
    end
end

if t <= T
    q_mse(:, :, t:T) = repmat(q_mse(:, :, t-1), [1, 1, T - t + 1]);
end

for tau = 1:t - 1
    if X(tau) > -1
        cum_s(H(tau), tau:end) = cum_s(H(tau), tau:end) + 1;
    end
end

typeI = typeI / K;
typeII = typeII / K;
