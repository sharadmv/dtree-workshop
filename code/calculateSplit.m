function fTree = calculateSplit(fTree, xTree, yTree, n, beta)

n

numThreshPerFeat = 5;
minNodeSize = 5;
if n<3
  depth = 1;
else
  depth = 0;
end

bestThres = 0;
x = xTree{n};
y = yTree{n};

origH = entropy(y);

minH = inf;

for i=1:size(x,2)
  threshes = sort(x(:,i));
  threshSpace = floor(size(threshes,1)/numThreshPerFeat);
  if threshSpace == 0
    threshSpace = 1;
  end
  threshes = threshes(1:threshSpace:end);
  threshes = unique(threshes);
  threshes = [0];
  for j=1:size(threshes,1)
    thresh = threshes(j);
    [h, xGr, yGr, xLe, yLe] = entropyUsingSplit(x, y, i, thresh, depth, numThreshPerFeat);
    if minH > h
      minH = h;
      bestXGr = xGr;
      bestYGr = yGr;
      bestXLe = xLe;
      bestYLe = yLe;
      bestF = i;
      bestThresh = thresh;
    end
  end
end
minH
size(x,2)
[z, a, b, c, d] = entropyUsingSplit(x, y, bestF, bestThresh, 0, numThreshPerFeat);
z


if origH > minH && origH - minH > beta && size(x,1) > minNodeSize 
  fTree(n, :) = [bestF bestThresh 0 1 -1 -1];

  grN = size(fTree, 1) + 1;
  fTree(n,5) = grN;
  xTree{grN} = bestXGr;
  yTree{grN} = bestYGr;
  [fTree] = calculateSplit(fTree, xTree, yTree, grN, beta);

  leN = size(fTree, 1) + 1;
  xTree{leN} = bestXLe;
  yTree{leN} = bestYLe;
  fTree(n,6) = leN;
  [fTree] = calculateSplit(fTree, xTree, yTree, leN, beta);
else
  num1 = sum(y);
  num0 = size(y, 1) - num1;
  fTree(n, :) = [0 0 num1>num0 2 0 0];
end
