function fsv = similarityTransform(image, p, q, a)

n_points = size(p, 1);
image_h = size(image, 1);
image_w = size(image, 2);

fsv = zeros(image_h, image_w, 2);
for vx = 1:image_w
    for vy = 1:image_h
        v = [vx vy];
        
        weights = 1./(abs(sqrt(sum((v - p).^2, 2)))).^(2*a);
        p_star = sum(weights .* p, 1) / sum(weights);
        q_star = sum(weights .* q, 1) / sum(weights);
        
        p_hat = p - p_star;
        q_hat = q - q_star;
        
        M = zeros(2, 2);
        mu = 0;
        for i = 1:n_points
            p_hat_i = p_hat(i, :);
            q_hat_i = q_hat(i, :);
            w_i = weights(i);
            mu = mu + w_i * (p_hat_i * p_hat_i');
            
            p_hat_i_op = [-p_hat_i(2) p_hat_i(1)];
            q_hat_i_op = [-q_hat_i(2) q_hat_i(1)];
            
            M = M + w_i * [p_hat_i; -p_hat_i_op] * [q_hat_i' -q_hat_i_op'];
        end
        
        M = M / mu;
        
        fsv(vy, vx, :) = (v - p_star) * M + q_star;
    end
end