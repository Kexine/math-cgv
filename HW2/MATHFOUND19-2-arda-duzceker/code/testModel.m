function nInliers = testModel(p, pp, e, Tx, Ty)
px = p(:, 1);
py = p(:, 2);
ppx = pp(:, 1);
ppy = pp(:, 2);

isInlierX = abs(px + Tx - ppx) <= e;
isInlierY = abs(py + Ty - ppy) <= e;

isInlier = isInlierX & isInlierY;

nInliers = sum(isInlier);

end