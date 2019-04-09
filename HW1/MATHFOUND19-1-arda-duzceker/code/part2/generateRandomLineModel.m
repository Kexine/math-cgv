function [a, b] = generateRandomLineModel()
%%% a, b: true model(line) parameters

% random line model is generated using polar representation of the line
theta_range = 0:0.03:pi;
theta = randsample(theta_range, 1);
r_range = -5:0.01:5;
r = randsample(r_range, 1);

a = -cot(theta);
b = r/sin(theta);
end