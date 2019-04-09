function [xc, yc, r] = generateRandomCircleModel()
%%% xc, yc, r: true model(circle) parameters

% randomly initialize the center and the radius of the circle
% limit x, y, r ranges so that the circle appears "nice"
center_max = 5;
center_min = -5;
xc = (center_max-center_min)*rand + center_min;
yc = (center_max-center_min)*rand + center_min;

radius_max = 9.5 - max(abs(xc), abs(yc));
radius_min = 3;
r = (radius_max-radius_min)*rand + radius_min;

end