function [value] = EI(a, b)
alpha = 0.05;

intv = betainv(1-alpha/2, a, b) - betainv(alpha/2, a, b);
intv_on = betainv(1-alpha/2, a+1, b) - betainv(alpha/2, a+1, b);
intv_off = betainv(1-alpha/2, a, b+1) - betainv(alpha/2, a, b+1);

intv_on = max(0, intv-intv_on);
intv_off = max(0, intv-intv_off);

value = (a * intv_on + b * intv_off) / (a + b);

end