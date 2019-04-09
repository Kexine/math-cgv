function problemList = removeBadCandidates(problemList)

lowerBounds = problemList.objectiveLowerBounds;
upperBounds = problemList.objectiveUpperBounds;

maxLowerBound = max(lowerBounds);
badCandidates = find(upperBounds < maxLowerBound);

badCandidates = sort(badCandidates, 'descend');
for i=1:length(badCandidates)
    indexToRemove = badCandidates(i);
    problemList = removeProblem(problemList, indexToRemove);
end


end