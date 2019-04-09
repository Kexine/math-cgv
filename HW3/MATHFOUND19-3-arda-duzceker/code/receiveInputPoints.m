function [p, q] = receiveInputPoints(image)
figure;
imshow(image);
title('Retreiving data points...');
hold on

p = [];
q = [];
while true   
    disp('Select a green point (right mouse click to exit)');
    [x, y, button] = ginput(1);
    if button == 3
        break;
    end

    plot(x, y, 'go');
    p = [p; [x y]];

    disp('Select the corresponding red point');
    [x, y, button] = ginput(1);

    plot(x, y, 'ro');
    q = [q; [x y]];
end

hold off
end