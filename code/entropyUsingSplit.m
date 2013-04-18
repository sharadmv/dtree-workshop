function [h, xGr, yGr, xLe, yLe] = entropyUsingSplit(x, y, f, threshold, depth, numThreshPerFeat)

% Find which index go to which array
j = find(x(:,f) > threshold);
k = find(x(:,f) <= threshold);

% Split into x and y where it is less than greater than threshold
xGr = x(j,:);
yGr = y(j,:);
xLe = x(k,:);
yLe = y(k,:);

if depth > 0
  hGr = inf;
  for i=1:size(xGr,2)
    threshes = sort(xGr(:,i));
    threshSpace = floor(size(threshes,1)/numThreshPerFeat);
    if threshSpace == 0
      threshSpace = 1;
    end
    threshes = threshes(1:threshSpace:end);
    threshes = unique(threshes);
    for j=1:size(threshes,1)
      thresh = threshes(j);
      [h, xGr1, yGr1, xLe1, yLe1] = entropyUsingSplit(xGr, yGr, i, thresh, depth - 1, numThreshPerFeat);
      if hGr > h
        hGr = h;
      end
    end
  end
  hLe = inf;
  for i=1:size(xLe,2)
    threshes = sort(xLe(:,i));
    threshSpace = floor(size(threshes,1)/numThreshPerFeat);
    if threshSpace == 0
      threshSpace = 1;
    end
    threshes = threshes(1:threshSpace:end);
    threshes = unique(threshes);
    for j=1:size(threshes,1)
      thresh = threshes(j);
      [h, xGr2, yGr2, xLe2, yLe2] = entropyUsingSplit(xLe, yLe, i, thresh, depth - 1, numThreshPerFeat);
      if hLe > h
        hLe = h;
      end
    end
  end
else


  % Calculate the entropy
  hGr = entropy(yGr);
  hLe = entropy(yLe);
end

% Get the sizes
sizeGr = size(xGr, 1);
sizeLe = size(xLe, 1);
sizeTo = size(x, 1);

% Calculate average entropy
if sizeTo == 0
  disp('hi')
end
h = sizeGr/sizeTo * hGr + sizeLe/sizeTo * hLe;
