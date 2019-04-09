function [B, l1_error, iter] = irlsL1(data, max_iter, tol, visualize)
%%% data: 2 x n matrix with n data points
%%% max_iter: number of maximum iterations before termination
%%% tol: tolerance for the change in B parameters before termination
%%% B: fitted line's parameters after termination (Y = XB)
%%% l1_error: total L1-norm error after termination
%%% iter: number of iterations needed before a termination condition is met

n_points = size(data, 2);

% Ordinary least squares (L-2 norm) solution as the starting point for B
% i.e. all weights are 1
% Model: Y = XB + e
Y = data(2,:)';
X = [data(1,:); ones(1, n_points)]';
B = pinv(X)*Y;

if visualize
    figure(1);
    scatter(data(1,:), data(2,:));
    hold on
end

for iter=1:max_iter
    residuals = abs(X*B - Y); % d(x, y_i)
    residuals(~residuals) = eps;
    w = (1/2) * (1./residuals);
    W = diag(w);
    
    B_prev = B;
    % weighted least squares solution (X'WX)B = X'WY
    B = (X'*W*X)\(X'*W*Y);  % B = inv(X'WX) * (X'WY)
    
    if visualize
        syms x y
        h = fimplicit(y == B(1)*x + B(2), 'LineWidth', 1, 'Color', 'green');
        pause(0.2);
    end
    
    
    if sqrt(sum((B-B_prev).^2)) < tol
        break
    end
    
    if visualize
        delete(h);
    end
end

l1_error = sum(abs(X*B - Y));

end