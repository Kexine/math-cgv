function problemList = removeProblem(problemList, indexToRemove)

problemList.objectiveLowerBounds(indexToRemove) = [];
problemList.objectiveUpperBounds(indexToRemove) = [];
problemList.thetaLowerBounds(indexToRemove, :) = [];
problemList.thetaUpperBounds(indexToRemove, :) = [];
problemList.thetaOptimizers(indexToRemove, :) = [];

end