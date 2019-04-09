function [xc, yc, r] = fitCircle(point1, point2, point3)
%%% point1, point2, point3: two element array that contains point coor.
%%% xc, yc, r: fitted circle's parameters going through given 3 points

% circle model: a(x2 + y2) + bx + cy + 1 = 0
% homogenous equation: Ax = 0 for 3 points

x1 = point1(1); y1 = point1(2);
x2 = point2(1); y2 = point2(2);
x3 = point3(1); y3 = point3(2);

A = [x1^2+y1^2 x1 y1 1;
     x2^2+y2^2 x2 y2 1;
     x3^2+y3^2 x3 y3 1];

Z = null(A);

coeff = Z ./ Z(end, end);
a = coeff(1);
b = coeff(2);
c = coeff(3);

if a == 0
    xc = 0;
    yc = 0;
    r = 0;
else
    xc = -b/(2*a);
    yc = -c/(2*a);
    r = sqrt((b^2 + c^2 - 4*a)/(4*a^2));
end

end