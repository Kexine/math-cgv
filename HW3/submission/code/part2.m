clear all
close all

addpath(genpath('../data/TASK2'));
image = imread('ginger.png');

[p, q] = receiveInputPoints(image);

% set a
a = 0.5;

image_h = size(image, 1);
image_w = size(image, 2);

fav = affineTransform(image, p, q, a);
fsv = similarityTransform(image, p, q, a);
frv = rigidTransform(image, p, q, a);

warped_affine = applyReverseMapping(image, fav);
warped_similarity = applyReverseMapping(image, fsv);
warped_rigid = applyReverseMapping(image, frv);

figure;
subplot(2, 2, 1);
imshow(image);
hold on; plot(p(:,1), p(:,2), 'go');
hold on; plot(q(:,1), q(:,2), 'ro');
subplot(2, 2, 2);
imshow(warped_affine);
title('Affine Warped Image');
subplot(2, 2, 3);
imshow(warped_similarity);
title('Similarity Warped Image');
subplot(2, 2, 4);
imshow(warped_rigid);
title('Rigid Warped Image');
