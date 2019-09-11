function bestIndex = findBestCandidate(problemList)

nProblems = size(problemList.objectiveLowerBounds, 1);
bestIndex = 1;
bestLower = problemList.objectiveLowerBounds(1);
bestUpper = problemList.objectiveUpperBounds(1);

for i=2:nProblems
    currentLower = problemList.objectiveLowerBounds(i);
    currentUpper = problemList.objectiveUpperBounds(i);
    
    comparison = compareBounds(currentLower, currentUpper, bestLower, bestUpper);
    if comparison
        bestIndex = i;
        bestLower = currentLower;
        bestUpper = currentUpper;
    end
end

end