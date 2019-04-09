function [xc, yc, r, inliers] = ransacCircle(data, iter, threshold)
%%% data: 2 x n matrix with n data points
%%% iter: number of RANSAC iterations that provides a certain confidence
%%% threshold: the threshold of the distances between points and the fitting line
%%% xc, yc, r: best fitted circle's parameters
%%% inliers: 2 x m matrix with inlier points for best fitted circle

[~,num_pts] = size(data);   % total number of points
best_n_inliers = 0;   % largest number of inliers as a result of the best fit
xc=0;
yc=0;
r=0;

data_x = data(1,:);
data_y = data(2,:);

for i=1:iter
    % randomly select 3 points and fit circle to these points
    point_indices = randperm(num_pts, 3);
    
    point1 = data(:,point_indices(1));
    point2 = data(:,point_indices(2));
    point3 = data(:,point_indices(3));
    
    [xc_temp, yc_temp, r_temp] = fitCircle(point1, point2, point3);
    
    % compute the distances between all points with the fitting circle
    distances = abs(sqrt((data_x-xc_temp).^2 + (data_y-yc_temp).^2) - r_temp);
    
    % find the inliers (distances smaller than the threshold)
    is_inlier = distances <= threshold;
    n_inliers = sum(is_inlier);
    
    % update the number of inliers and fitting model if better model is found
    if n_inliers > best_n_inliers
        inliers = data(:,is_inlier);
        xc = xc_temp;
        yc = yc_temp;
        r = r_temp;
        best_n_inliers = n_inliers;
    end
end

end