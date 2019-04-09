function problemList = branchAndSolve(p, pp, e, problemList, index)

thetaLowerBound = problemList.thetaLowerBounds(index, :);
thetaUpperBound = problemList.thetaUpperBounds(index, :);

problemList = removeProblem(problemList, index);

Tx_lower = thetaLowerBound(1);
Tx_upper = thetaUpperBound(1);
Ty_lower = thetaLowerBound(2);
Ty_upper = thetaUpperBound(2);

Tx_range = Tx_upper - Tx_lower;
Ty_range = Ty_upper - Ty_lower;

if Tx_range > Ty_range
    middle = floor(Tx_lower + Tx_range/2);
    
    % first branch
    if middle >= Tx_lower
        thetaLowerBound = [Tx_lower Ty_lower];
        thetaUpperBound = [middle   Ty_upper];
        problemList = addNewProblem(p, pp, e, problemList, thetaLowerBound, thetaUpperBound);
    end
    
    % second branch
    if middle+1 <= Tx_upper
        thetaLowerBound = [middle+1 Ty_lower];
        thetaUpperBound = [Tx_upper Ty_upper];
        problemList = addNewProblem(p, pp, e, problemList, thetaLowerBound, thetaUpperBound);
    end                       
else
    middle = floor(Ty_lower + Ty_range/2);
    
    % first branch
    if middle >= Ty_lower
        thetaLowerBound = [Tx_lower Ty_lower];
        thetaUpperBound = [Tx_upper   middle];
        problemList = addNewProblem(p, pp, e, problemList, thetaLowerBound, thetaUpperBound);
    end
    
    % second branch
    if middle+1 <= Ty_upper
        thetaLowerBound = [Tx_lower middle+1];
        thetaUpperBound = [Tx_upper Ty_upper];
        problemList = addNewProblem(p, pp, e, problemList, thetaLowerBound, thetaUpperBound);
    end
end


end