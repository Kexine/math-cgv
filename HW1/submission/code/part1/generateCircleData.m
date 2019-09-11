function data = generateCircleData(xc, yc, r_model, outlier_ratio, threshold)
%%% xc, yc, r_model: true model(circle) parameters
%%% outlier_ratio: the ratio of points that does not follow model + noise
%%% threshold: max distance to the model for a point to be an inlier
%%% data: 2 x n matrix with n data points

% initialize the parameters
noise_max = threshold;
inlier_ratio = 1 - outlier_ratio;
n_outliers = int64(100*outlier_ratio);
n_inliers = int64(100*inlier_ratio);

% create inlier points using polar coordinates 
theta_range = 0:0.01:2*pi;
theta = randsample(theta_range, n_inliers);
noise = (2*noise_max)*rand(1, n_inliers) - noise_max;
r = noise + r_model;
x_inliers = r .* cos(theta) + xc;
y_inliers = r .* sin(theta) + yc;

% create outlier points randomly (verify later)
x_outliers = zeros(1, n_outliers);
y_outliers = zeros(1, n_outliers);
i = 1;
outlier_max = 10;
outlier_min = -10;
while i <= n_outliers
    current_x = (outlier_max-outlier_min)*rand + outlier_min;
    current_y = (outlier_max-outlier_min)*rand + outlier_min;
    distance_to_circle = abs(sqrt((current_x-xc)^2 + (current_y-yc)^2) - r_model);

    % verify: ensure that the randomly created data point is an outlier
    if distance_to_circle > threshold
        x_outliers(i) = current_x;
        y_outliers(i) = current_y;
        i = i + 1;
    end
end

x = [x_inliers x_outliers];
y = [y_inliers y_outliers];
shuffle = randperm(100);
data = [x(shuffle); y(shuffle)];

end