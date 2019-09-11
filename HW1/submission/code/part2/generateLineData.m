function data = generateLineData(a, b, outlier_ratio, threshold)
%%% a, b: true model(line) parameters
%%% outlier_ratio: the ratio of points that does not follow model + noise
%%% threshold: max distance to the model for a point to be an inlier
%%% data: 2 x n matrix with n data points

% initialize the parameters
noise_max = threshold;
inlier_ratio = 1 - outlier_ratio;
n_outliers = int64(100*outlier_ratio);
n_inliers = int64(100*inlier_ratio);

% create inlier points (y = ax + b + noise)
x_range = -10:0.02:10;
noise = (2*noise_max)*rand(1, n_inliers) - noise_max;
x_inliers = randsample(x_range, n_inliers, true);
y_inliers = a*x_inliers + b + noise;

% create outlier points randomly (verify later)
x_outliers = zeros(1, n_outliers);
y_outliers = zeros(1, n_outliers);
i = 1;
outlier_max = 10;
outlier_min = -10;
while i <= n_outliers
    current_x = (outlier_max-outlier_min)*rand + outlier_min;
    current_y = (outlier_max-outlier_min)*rand + outlier_min;
    
    distance_to_line = abs(current_y -(a*current_x + b));

    % verify: ensure that the randomly created data point is an outlier
    if distance_to_line > threshold
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