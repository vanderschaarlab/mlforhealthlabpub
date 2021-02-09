function [ ucb ] = UCB_ind( p_hat , c , n , ln)
upperbound = 2;

value = p_hat + (c*log(ln)/n)^(1/2);
ucb = min( value, upperbound );

end