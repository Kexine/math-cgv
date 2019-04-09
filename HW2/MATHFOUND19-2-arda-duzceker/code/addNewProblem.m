function problemList = addNewProblem(p, pp, e, problemList, thetaLowerBound, thetaUpperBound)

[Tx, Ty, objectiveUpperBound] = solveLP(p, pp, e, thetaLowerBound, thetaUpperBound);
objectiveLowerBound = testModel(p, pp, e, Tx, Ty);
thetaOptimizer = [Tx Ty];

if objectiveUpperBound >= 0
    problemList.thetaLowerBounds = [problemList.thetaLowerBounds; thetaLowerBound];
    problemList.thetaUpperBounds = [problemList.thetaUpperBounds; thetaUpperBound];
    problemList.objectiveLowerBounds = [problemList.objectiveLowerBounds; objectiveLowerBound];
    problemList.objectiveUpperBounds = [problemList.objectiveUpperBounds; objectiveUpperBound];
    problemList.thetaOptimizers = [problemList.thetaOptimizers; thetaOptimizer];
end

end