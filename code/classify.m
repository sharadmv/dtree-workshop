function [y] = classify(fTree, x)

n = 1;

while (fTree(n, 4) ~= 2)
  j = fTree(n, 1);
  threshold = fTree(n, 2);
  if x(1, j) > threshold
    n = fTree(n, 5);
  else
    n = fTree(n, 6);
  end
end

y = fTree(n, 3);

