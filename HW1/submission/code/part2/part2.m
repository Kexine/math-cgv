close all;
clear all;

%% GENERATE TRUE MODEL (y = ax + b) and DATA
[a, b] = generateRandomLineModel();

%% GENERATE DATA
outlier_ratio = 0.1;
threshold =  0.1;
data = generateLineData(a, b, outlier_ratio, threshold);

%% FIT WITH IRLS USING L-1 NORM
% last parameter is to enable/disable visualization
[B_IRLS, l1_error_IRLS, iteration] = irlsL1(data, 1000, 0.000001, false);

%% FIT WITH LINEAR PROGRAMMING USING L-1 NORM
[B_LP_L1, l1_error_LP] = linearProgL1(data);

%% FIT WITH LINEAR PROGRAMMING USING L-INFINITY NORM
[B_LP_Linf, linf_error_LP] = linearProgLinf(data);

%% PLOT RESULTS
figure;
scatter(data(1,:), data(2,:), 'DisplayName', 'DATA');
hold on
syms x y
fimplicit(y == a*x + b, 'LineWidth', 1, 'Color', 'green', 'DisplayName', 'TRUE MODEL');
hold on
fimplicit(y == B_IRLS(1)*x + B_IRLS(2), 'LineWidth', 1, 'Color', 'cyan', 'DisplayName', 'IRLS with L-1 NORM');
hold on
fimplicit(y == B_LP_L1(1)*x + B_LP_L1(2), 'LineWidth', 1, 'Color', 'red',  'DisplayName', 'LP with L-1 NORM');
hold on
fimplicit(y == B_LP_Linf(1)*x + B_LP_Linf(2), 'LineWidth', 1, 'Color', 'black', 'DisplayName', 'LP with L-INF NORM');





