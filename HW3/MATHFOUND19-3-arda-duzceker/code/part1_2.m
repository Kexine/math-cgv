clear all
close all

% choose the dataset and parameters
load("../data/TASK1/curve_data3.mat");
sigmas = [0.001 0.005 0.01 0.05 0.1 0.5];
dx = 0.01;
dy = 0.01;

x_range = min(xi)*1.1:dx:max(xi)*1.1;
y_range = min(yi)*1.1:dy:max(yi)*1.1;
grid_values = zeros(size(y_range, 2), size(x_range, 2));

figure;
for trial=1:size(sigmas, 2)
    sigma = sigmas(trial);
    for i=1:size(y_range, 2)
        y = y_range(i);
        for j=1:size(x_range, 2)
            x = x_range(j);
            x_xi = (x-xi);
            y_yi = (y-yi);
            r2 = x_xi.^2 + y_yi.^2;
            weights = exp(-r2/(sigma^2));
            f = nix.*x_xi + niy.*y_yi;
            c = sum(weights.*f)/sum(weights);
            grid_values(i, j) = c;
        end
    end
    subplot(2, 3, trial);
    imagesc(x_range, y_range, grid_values);
    hold on;
    plot(xi, yi, 'Color', 'red');
    title(sprintf("Sigma: %.3f", sigma));
end

