function warped_image = applyReverseMapping(image, fv)
image_h = size(image, 1);
image_w = size(image, 2);

fv_x = fv(:, :, 1);
fv_y = fv(:, :, 2);

[xq, yq] = meshgrid(1:image_w, 1:image_h);

vx = xq(~isnan(fv_x));
vy = yq(~isnan(fv_y));

fv_x = fv_x(~isnan(fv_x));
fv_y = fv_y(~isnan(fv_y));

vq_x = griddata(fv_x(:), fv_y(:), vx, xq, yq);
vq_y = griddata(fv_x(:), fv_y(:), vy, xq, yq);

warped_image = zeros(size(image));
for x=1:image_w
    for y=1:image_h
        inverse_x_transform = round(vq_x(y, x));
        inverse_y_transform = round(vq_y(y, x));
        if ~isnan(inverse_x_transform) && ~isnan(inverse_y_transform)
            if (0 < inverse_x_transform) && (inverse_x_transform < image_w)
                if (0 < inverse_y_transform) && (inverse_y_transform < image_h)
                    warped_image(y, x, :) = image(inverse_y_transform, inverse_x_transform, :);
                end
            end
        end
    end
end

warped_image = uint8(warped_image);
end