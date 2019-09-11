% show feature matches between two images
%
% Input:
%   img1        - n x m color image 
%   corner1     - 2 x k matrix, holding keypoint coordinates of first image
%   img2        - n x m color image 
%   corner1     - 2 x k matrix, holding keypoint coordinates of second image
%   fig         - figure id
function showFeatureMatches(img1, img2, inliers1, inliers2, outliers1, outliers2, fig)
    [sx, sy, sz] = size(img1);
    img = [img1, img2];

    inliers2 = inliers2 + repmat([sy, 0]', [1, size(inliers2, 2)]);
    outliers2 = outliers2 + repmat([sy, 0]', [1, size(outliers2, 2)]);

    figure(fig), imshow(img, []);
    hold on, plot(outliers1(1,:), outliers1(2,:), '+r', 'LineWidth', 1.25);
    hold on, plot(outliers2(1,:), outliers2(2,:), '+r', 'LineWidth', 1.25);    
    hold on, plot([outliers1(1,:); outliers2(1,:)], [outliers1(2,:); outliers2(2,:)], 'r', 'LineWidth', 1.25);
    hold on, plot(inliers1(1,:), inliers1(2,:), '+g', 'LineWidth', 1.25);
    hold on, plot(inliers2(1,:), inliers2(2,:), '+g', 'LineWidth', 1.25);    
    hold on, plot([inliers1(1,:); inliers2(1,:)], [inliers1(2,:); inliers2(2,:)], 'g', 'LineWidth', 1.25);
    
end
