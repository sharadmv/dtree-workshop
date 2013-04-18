function [h] = entropy(y)

numTotal = size(y,1);

%Assume y is either 1 or 0 so you can sum to find number of 1's
num1 = sum(y);
num0 = numTotal - num1;

p0 = num0/numTotal;
p1 = num1/numTotal;

if p0 == 0 || p1 == 0
  h = 0;
else
  h = - (p0 * log2(p0) + p1 * log2(p1));
end

