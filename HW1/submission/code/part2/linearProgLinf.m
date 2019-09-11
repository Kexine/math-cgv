function [B, linf_error] = linearProgLinf(data)
%%% data: 2 x n matrix with n data points
%%% B: fitted line's parameters after termination (Y = XB)
%%% linf_error: total L-infinity norm error after termination

n_points = size(data, 2);
Y = data(2,:)';
X = [data(1,:); ones(1, n_points)]';

% prepare linprog arguments
% https://math.stackexchange.com/questions/2589887/how-can-the-infinity-norm-minimization-problem-be-rewritten-as-a-linear-program
f = [0 0 1];
A1 = [X -ones(n_points,1)];
A2 = [-X -ones(n_points,1)];
A = [A1; A2];
b = [Y; -Y];

result = linprog(f, A, b);
B = result(1:2);
linf_error = max(abs(X*B - Y));

end