function frv = rigidTransform(image, p, q, a)

n_points = size(p, 1);
image_h = size(image, 1);
image_w = size(image, 2);

frv = zeros(image_h, image_w, 2);
for vx = 1:image_w
    for vy = 1:image_h
        v = [vx vy];
        
        weights = 1./(abs(sqrt(sum((v - p).^2, 2)))).^(2*a);
        p_star = sum(weights .* p, 1) / sum(weights);
        q_star = sum(weights .* q, 1) / sum(weights);
        
        p_hat = p - p_star;
        q_hat = q - q_star;
        
        M = zeros(2, 2);
        mu1 = 0;
        mu2 = 0;
        for i = 1:n_points
            p_hat_i = p_hat(i, :);
            q_hat_i = q_hat(i, :);
            w_i = weights(i);
            p_hat_i_op = [-p_hat_i(2) p_hat_i(1)];
            q_hat_i_op = [-q_hat_i(2) q_hat_i(1)];
            
            mu1 = mu1 + w_i * (q_hat_i * p_hat_i');
            mu2 = mu2 + w_i * (q_hat_i * p_hat_i_op');
            
            M = M + w_i * [p_hat_i; -p_hat_i_op] * [q_hat_i' -q_hat_i_op'];
        end
        
        M = M / sqrt(mu1^2 + mu2^2);
        
        frv(vy, vx, :) = (v - p_star) * M + q_star;
    end
end