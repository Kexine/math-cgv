clear all
close all
rng('shuffle','twister');

%% GENERATE TRUE MODEL
outlier_ratios = [0.05 0.2 0.3 0.7];
[xc, yc, r] = generateRandomCircleModel();

disp("%%%%%% TRUE MODEL %%%%%%");
fprintf("Circle Center: [ %.3f , %.3f ]\n", xc, yc);
fprintf("Circle Radius: %.3f\n\n", r);

%% RANSAC and EXHAUSTIVE SEARCH
% for each outlier ratio perform RANSAC and exhaustive search

outlier_threshold = 0.1;
p = 0.99;
s = 3;

for i = 1:4
    best_inliers_fit = 0;
    inlier_counts = zeros(1, 1000);
    outlier_ratio = outlier_ratios(i);
    data = generateCircleData(xc, yc, r, outlier_ratio, outlier_threshold);
    data_x = data(1,:);
    data_y = data(2,:);
    n_iterations = ceil(log(1-p)/log(1-(1-outlier_ratio)^s));
    
    tic;
    % SECTION 1.2
    for j = 1:1000
        [xc_fit, yc_fit, r_fit, inliers_fit] = ransacCircle(data, n_iterations, 0.1);
        inlier_count = size(inliers_fit, 2);
        inlier_counts(j) = inlier_count;
        
        if inlier_count >= size(best_inliers_fit, 2)
            best_xc_fit = xc_fit;
            best_yc_fit = yc_fit;
            best_r_fit = r_fit;
            best_inliers_fit = inliers_fit;
        end
    end
    ransacElapsed = toc;
    
    % SECTION 1.3
    tic;
    C = nchoosek(1:100, 3);
    best_n_inliers_exhaustive = 0;
    for j = 1:size(C, 1)
        point1 = data(:, C(j,1));
        point2 = data(:, C(j,2));
        point3 = data(:, C(j,3));
        [xc_temp, yc_temp, r_temp] = fitCircle(point1, point2, point3);
        distances = abs(sqrt((data_x-xc_temp).^2 + (data_y-yc_temp).^2) - r_temp);
        % find the inliers with distances smaller than the threshold
        is_inlier = distances < outlier_threshold;
        n_inliers = sum(is_inlier);
        
        % update the number of inliers and fitting model if better model is found
        if n_inliers > best_n_inliers_exhaustive
            xc_exhaustive = xc_temp;
            yc_exhaustive = yc_temp;
            r_exhaustive = r_temp;
            best_n_inliers_exhaustive = n_inliers;
        end
    end
    exhaustiveElapsed = toc;
    
    % plot histogram
    subplot(2,4,i);
    histogram(inlier_counts, 100);
    xlim([0, 100]);
    xlabel('Number of inliers');
    ylabel('Number of experiments');
    
    % plot synthesized data and RANSAC fit
    subplot(2,4,i+4);
    h_data = scatter(data_x, data_y, [], 'red', ...
        'DisplayName', 'RANSAC Outliers');
    hold on
    syms x y
    fimplicit((x-xc)^2 + (y-yc)^2 == r^2, 'LineWidth', 1, ...
        'Color', 'green', 'DisplayName', 'Synth. Model')
    hold on
    fimplicit((x-best_xc_fit)^2 + (y-best_yc_fit)^2 == best_r_fit^2, ...
        'LineWidth', 1, 'Color', 'black', 'DisplayName', 'RANSAC Model')
    hold on
    h_inliers = scatter(best_inliers_fit(1,:), best_inliers_fit(2,:), ...
        [],'blue', 'DisplayName', 'RANSAC Inliers');
    xlim([-10, 10]);
    ylim([-10, 10]);
    legend('Location','southoutside')
    
    fprintf("%%%%%% OUTLIER RATIO: %.2f %%%%%%\n ", outlier_ratio);
    fprintf("-- RANSAC --\n");
    fprintf("Circle Center: [ %.3f , %.3f ]\n", best_xc_fit, best_yc_fit);
    fprintf("Circle Radius: %.3f\n", best_r_fit);
    fprintf("Number of Found Inliers: %d\n", size(best_inliers_fit, 2));
    fprintf("Time Elapsed: %.3f\n", ransacElapsed);
    fprintf("-- EXHAUSTIVE SEARCH --\n");
    fprintf("Circle Center: [ %.3f , %.3f ]\n", xc_exhaustive, yc_exhaustive);
    fprintf("Circle Radius: %.3f\n", r_exhaustive);
    fprintf("Number of Found Inliers: %d\n", best_n_inliers_exhaustive);
    fprintf("Time Elapsed: %.3f\n\n", exhaustiveElapsed);
end




