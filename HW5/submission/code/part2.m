clear
close all

Iorig = imread('../lotr.jpg');
Iorig = rgb2gray(Iorig);
[h,w] = size(Iorig);

% Noise
In = imnoise(Iorig,'gaussian',0,0.05);

% Show
figure(1)
clf
subplot(1,2,1)
imshow(Iorig);
title('Original image')
subplot(1,2,2)
imshow(In);
title('Noisy image')
box off

%% TASK 1
sigma = 0.5;
n_iter = 32;
task1(Iorig, In, sigma, n_iter);

%% TASK 2
alpha = 0;
time_step = 0.05;
n_iter = 96;
task2(Iorig, In, alpha, time_step, n_iter);

%% TASK 3
lambda = 2;
task3(Iorig, In, lambda);


%% FUNCTIONS
function task1(original_image, noisy_image, sigma, n_iter)
original_image = double(original_image);
n_rows = 2;
n_cols = 2;
n_skip = n_iter/(n_rows * n_cols);
kernel = getGaussianKernel(sigma);
errors = zeros(1,n_iter);
figure;
sgtitle('Gaussian Filtering Resulting Images');
for i=1:n_iter
    filtered_image = applyKernel(noisy_image, kernel);
    error = mean(((original_image - filtered_image).^2), 'all');
    errors(i) = error;
    if rem(i, n_skip) == 0
        subplot(n_rows, n_cols, i/n_skip);
        imshow(uint8(filtered_image));
        title(sprintf('Iteration: %d', i));
    end
    noisy_image = filtered_image;
end

figure;
plot(errors);
title('Gaussian Filtering - Evolution of the Errors')
xlabel('Iterations');
ylabel('Mean-Squared-Error');

[value, index] = min(errors);
disp("MSE of best iteration with Gaussian filtering: [iteration error]")
disp([index value])
end

function task2(original_image, noisy_image, alpha, time_step, n_iter)
original_image = double(original_image);
noisy_image = double(noisy_image);
n_rows = 2;
n_cols = 2;
n_skip = n_iter/(n_rows * n_cols);
kernel = getLaplacianKernel(alpha);
errors = zeros(1, n_iter);
figure;
sgtitle('Heat Diffusion Resulting Images');
for i=1:n_iter
    filtered_image = applyKernel(noisy_image, kernel);
    filtered_image = noisy_image + time_step*filtered_image;
    error = mean(((original_image - filtered_image).^2), 'all');
    errors(i) = error;
    if rem(i, n_skip) == 0
        subplot(n_rows, n_cols, i/n_skip);
        imshow(uint8(filtered_image));
        title(sprintf('Iteration: %d', i));
    end
    noisy_image = filtered_image;
end

figure;
plot(errors);
title('Heat Diffusion - Evolution of the Errors')
xlabel('Iterations');
ylabel('Mean-Squared-Error');

[value, index] = min(errors);
disp("MSE of best iteration with heat diffusion: [iteration error]")
disp([index value])
end

function task3(original_image, noisy_image, lambda)
original_image = double(original_image);
noisy_image = double(noisy_image);
[h, w] = size(original_image);
n = h * w;

a = 1 + 4*lambda;
b = -lambda;
values = repmat([b b a b b], n, 1);
A = spdiags(values, [-h -1 0 1 h], n, n);
noisy_image_vec = noisy_image(:);
smoothed_image_vec = A \ noisy_image_vec;
smoothed_image = reshape(smoothed_image_vec, h, w);

error = mean(((original_image - smoothed_image).^2), 'all');
disp("MSE after variational denoising:")
disp(error)

figure;
subplot(1, 2, 1)
imshow(uint8(noisy_image))
title('Noisy Image')
subplot(1, 2, 2)
imshow(uint8(smoothed_image))
title('Image After Variational Denoising')
end

function kernel = getGaussianKernel(sigma)
kernel_radius = ceil(3 * sigma);
kernel_size = 2*kernel_radius + 1;
kernel = fspecial('gaussian', kernel_size, sigma);
end

function kernel = getLaplacianKernel(alpha)
kernel = fspecial('laplacian', alpha);
end

function filtered_image = applyKernel(image, kernel)
pad_width = (size(kernel, 1) - 1)/2;
padded_image = padarray(image, [pad_width pad_width], 'replicate', 'both');
filtered_image = conv2(padded_image, kernel, 'valid');
end



