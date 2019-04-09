function [Tx, Ty, nInliers] = solveLP(p, pp, e, thetaLowerBound, thetaUpperBound)
%%% p: left image points
%%% pp: right image points
%%% e: outlier threshold
n = size(p, 1);

Tx_lower = thetaLowerBound(1);
Tx_upper = thetaUpperBound(1);
Ty_lower = thetaLowerBound(2);
Ty_upper = thetaUpperBound(2);

% solution = (z1, ... , zn, w1x, ... , wnx, w1y, ... , wny, Tx, Ty)

% for each p and p' => (x, x') and (y, y')
% there are 12 constraint inequalities

A = zeros(12*n, 3*n + 2);
b = zeros(12*n, 1);
for i=1:n
    x = p(i,1); xp = pp(i,1);
    y = p(i,2); yp = pp(i,2);
    
    z_coeffs = [xp-x-e;
                x-xp-e;
                Tx_lower;
                Tx_upper;
                -Tx_lower;
                -Tx_upper;
                yp-y-e;
                y-yp-e;
                Ty_lower;
                Ty_upper;
                -Ty_lower;
                -Ty_upper];
            
    w_coeffs = [-1;
                +1;
                -1;
                -1;
                +1;
                +1];
    
    T_coeffs = [0;
                0;
                0;
                1;
                -1;
                0];
    
    A(12*i-11:12*i, i) = z_coeffs; %z's
    A(12*i-11:12*i-6, n+i) = w_coeffs; %wx's
    A(12*i-5:12*i, 2*n+i) = w_coeffs; %wy's
    A(12*i-11:12*i-6, end-1) = T_coeffs; %Tx's
    A(12*i-5:12*i, end) = T_coeffs; %Ty's
    
    b(12*i-11:12*i, 1) = [0;
                          0;
                          0;
                          Tx_upper;
                          -Tx_lower;
                          0;
                          0;
                          0;
                          0;
                          Ty_upper;
                          -Ty_lower;
                          0];
end

f = [-ones(1,n) zeros(1,2*n+2)];
lb = [zeros(1,n) -inf(1, 2*n) Tx_lower Ty_lower];
ub = [ones(1,n) inf(1, 2*n) Tx_upper Ty_upper];

[solution, cost, exitFlag] = linprog(f, A, b,[],[],lb, ub);

if exitFlag == 1 %no feasible solution is found
    Tx = solution(end-1);
    Ty = solution(end);
    nInliers = -cost;
else
    Tx = inf;
    Ty = inf;
    nInliers = -1;
end

end