function fav = affineTransform(image, p, q, a)

n_points = size(p, 1);
image_h = size(image, 1);
image_w = size(image, 2);

fav = zeros(image_h, image_w, 2);
for vx = 1:image_w
    for vy = 1:image_h
        v = [vx vy];
        
        weights = 1./(abs(sqrt(sum((v - p).^2, 2)))).^(2*a);
        p_star = sum(weights .* p, 1) / sum(weights);
        q_star = sum(weights .* q, 1) / sum(weights);
        
        p_hat = p - p_star;
        q_hat = q - q_star;
        
        pwp = zeros(2, 2);
        for i = 1:n_points
            p_hat_i = p_hat(i, :);
            w_i = weights(i);
            pwp = pwp + p_hat_i' * w_i * p_hat_i;
        end
        
        wpq = zeros(2, 2);
        for j = 1:n_points
            p_hat_j = p_hat(j, :);
            q_hat_j = q_hat(j, :);
            w_j = weights(j);
            wpq = wpq + w_j * p_hat_j' * q_hat_j;
        end
        
        M = pwp \ wpq;
        
        fav(vy, vx, :) = (v - p_star) * M + q_star;
    end
end