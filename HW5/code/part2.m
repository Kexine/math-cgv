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

sigma = 0.5;
n_iter = 32;
task1(Iorig, In, sigma, n_iter);


%% TASK 1 FUNCTIONS
function task1(original_image, noisy_image, sigma, n_iter)
original_image = double(original_image);
n_rows = 2;
n_cols = n_iter/n_rows/4;
kernel = getGaussianKernel(sigma);
errors = zeros(1,n_iter);
figure;
sgtitle('Gaussian Filtering Resulting Images');
for i=1:n_iter
    filtered_image = applyGaussianKernel(noisy_image, kernel);
    error = mean(((original_image - filtered_image).^2), 'all');
    errors(i) = error;
    if rem(i, 4) == 0
        subplot(n_rows, n_cols, i/4);
        imshow(uint8(filtered_image));
        title(sprintf('Iteration: %d', i));
    end
    noisy_image = filtered_image;
end

figure;
plot(errors);
title('Evolution of the errors over the number of filtering')
xlabel('Iterations');
ylabel('Mean-Squared-Error');
end

function kernel = getGaussianKernel(sigma)
kernel_radius = ceil(3 * sigma);
kernel_size = 2*kernel_radius + 1;
kernel = fspecial('gaussian', kernel_size, sigma);
end

function filtered_image = applyGaussianKernel(image, kernel)
pad_width = (size(kernel, 1) - 1)/2;
padded_image = padarray(image, [pad_width pad_width], 'replicate', 'both');
filtered_image = conv2(padded_image, kernel, 'valid');
end


%% TASK 2 FUNCTIONS

