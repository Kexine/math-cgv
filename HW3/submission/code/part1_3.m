clear all
close all

addpath(genpath('../data'));

[V, F, UV, C, N] = readOFF('bunny.off');
sigma = 0.01;
threshold = 1;

xi = V(:, 1);
yi = V(:, 2);
zi = V(:, 3);

nix = N(:, 1);
niy = N(:, 2);
niz = N(:, 3);

n_vertices = size(V, 1);
V_new = V;
for iter=1:1
    FX = zeros(size(V));
    GRAD_FX = zeros(size(V));
    for index=1:n_vertices
        x = V_new(index, 1);
        y = V_new(index, 2);
        z = V_new(index, 3);
        x_xi = (x-xi);
        y_yi = (y-yi);
        z_zi = (z-zi);
        r2 = x_xi.^2 + y_yi.^2 + z_zi.^2;
        weights = exp(-r2/(sigma^2));
        fi = nix.*x_xi + niy.*y_yi + niz.*z_zi;
        fx = sum(weights.*fi)/sum(weights);
        grad_weights_x = (-2/sigma^2) * weights .* x_xi;
        grad_weights_y = (-2/sigma^2) * weights .* y_yi;
        grad_weights_z = (-2/sigma^2) * weights .* z_zi;
        grad_fx_x = (sum(weights.*nix) + sum(grad_weights_x.*(fi - fx)))/sum(weights);
        grad_fx_y = (sum(weights.*niy) + sum(grad_weights_y.*(fi - fx)))/sum(weights);
        grad_fx_z = (sum(weights.*niz) + sum(grad_weights_z.*(fi - fx)))/sum(weights);
        FX(index, :) = fx;
        GRAD_FX(index, :) = [grad_fx_x grad_fx_y grad_fx_z];
    end
    
    step = FX.*GRAD_FX;
    change = norm(step);
    
    if change < threshold
        break;
    end
    
    V_new = V_new - step;
end

N = per_vertex_normals(V_new, F);
writeOFF('trial.off', V_new, F, [], [], N);