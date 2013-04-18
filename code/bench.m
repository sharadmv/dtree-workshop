function [err_rate] = bench(fTree, x, y)

yGen = [];

for i=1:size(x,1)
  yGen(i,1) = classify(fTree, x(i,:));
end

correct = sum(y == yGen);
wrong = size(x, 1) - correct;

err_rate = wrong/size(x,1);

