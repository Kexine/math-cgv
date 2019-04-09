clear all
close all

load('data/ListInputPoints.mat');
imageSize = size(imread('data/InputLeftImage.png'));

leftImagePoints = ListInputPoints(:, 1:2);
rightImagePoints = ListInputPoints(:, 3:4);

inlierThreshold = 3;

thetaLowerBounds = [];
thetaUpperBounds = [];
objectiveLowerBounds = [];
objectiveUpperBounds = [];
thetaOptimizers = [];

problemList = struct('thetaLowerBounds', thetaLowerBounds, ...
                     'thetaUpperBounds', thetaUpperBounds, ...
                     'objectiveLowerBounds', objectiveLowerBounds, ...
                     'objectiveUpperBounds', objectiveUpperBounds, ...
                     'thetaOptimizers', thetaOptimizers);

thetaLowerBound = -[imageSize(2) imageSize(1)];
thetaUpperBound = [imageSize(2) imageSize(1)];

problemList = addNewProblem(leftImagePoints, rightImagePoints, ...
    inlierThreshold, problemList, thetaLowerBound, thetaUpperBound);

ubs = [];
lbs = [];
highest_lb = 0;
bestProblemIndex = -1;
while true
    bestProblemIndex = findBestCandidate(problemList);
    
    lb = problemList.objectiveLowerBounds(bestProblemIndex);
    ub = problemList.objectiveUpperBounds(bestProblemIndex);
    
    if lb > highest_lb
        highest_lb = lb;
    end
    ubs = [ubs max(problemList.objectiveUpperBounds)];
    lbs = [lbs highest_lb];
    
    if ub - lb < 1
        break
    end
    
    problemList = branchAndSolve(leftImagePoints, rightImagePoints, ....
        inlierThreshold, problemList, bestProblemIndex);
    
    problemList = removeBadCandidates(problemList);
end

leftImage = imread('data/InputLeftImage.png');
rightImage = imread('data/InputRightImage.png');

Tx = problemList.thetaOptimizers(bestProblemIndex, 1);
Ty = problemList.thetaOptimizers(bestProblemIndex, 2);

isInlierX = abs(leftImagePoints(:,1) + Tx - rightImagePoints(:,1)) <= inlierThreshold;
isInlierY = abs(leftImagePoints(:,2) + Ty - rightImagePoints(:,2)) <= inlierThreshold;

isInlier = isInlierX & isInlierY;
inlierIndices = find(isInlier);
outlierIndices = find(~isInlier);

leftInliers = leftImagePoints(inlierIndices,:);
rightInliers = rightImagePoints(inlierIndices, :);
leftOutliers = leftImagePoints(outlierIndices,:);
rightOutliers = rightImagePoints(outlierIndices, :);

showFeatureMatches(leftImage, rightImage, leftInliers', rightInliers', leftOutliers', rightOutliers', 1)

figure(2);
plot(ubs, '-or', 'DisplayName', 'Upper Bound'); hold on;
plot(lbs, '-ob', 'DisplayName', 'Lower Bound');
grid on
xticks(1:1:size(ubs, 2));
ylabel('Number of inliers');
xlabel('Iterations');
title('Convergence of Branch and Bound')


fprintf('Indices of the inlier matches:\n');
for i=1:size(inlierIndices)
    fprintf('%d ', inlierIndices(i))
end
fprintf('\n\n');
fprintf('Indices of the outlier matches:\n');
for i=1:size(outlierIndices)
    fprintf('%d ', outlierIndices(i))
end
fprintf('\n\n');

fprintf('Globally optimal solution (Tx, Ty) = (%.1f, %.1f)\n', Tx, Ty);


