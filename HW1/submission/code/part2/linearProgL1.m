function [B, l1_error] = linearProgL1(data)
%%% data: 2 x n matrix with n data points
%%% B: fitted line's parameters after termination (Y = XB)
%%% l1_error: total L-1 norm error after termination

n_points = size(data, 2);
Y = data(2,:)';
X = [data(1,:); ones(1, n_points)]';

% prepare linprog arguments
% http://cs.brown.edu/people/pfelzens/engn2520/CS1420_Lecture_4.pdf
f = [0 0 ones(1, n_points)];
A = zeros(2*n_points, n_points+2);
b = zeros(2*n_points, 1);

A(1:2:end, 1:2) = X;
A(2:2:end, 1:2) = -X;
b(1:2:end, :) = Y;
b(2:2:end, :) = -Y;
for i=1:n_points
    A(2*i-1, i+2) = -1;
    A(2*i, i+2) = -1;
end

result = linprog(f, A, b);
B = result(1:2);
l1_error = sum(abs(X*B - Y));

end