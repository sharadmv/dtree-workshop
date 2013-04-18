function err_rate = main(beta, numThreshPerFeat, deep)
load('spamData.mat');

%{
fTree = [0 0 0 0 0 0];
xTree{1} = Xtrain;
yTree{1} = ytrain;
[fTree] =  calculateSplit(fTree, xTree, yTree, 1, .039);
bench(fTree, Xtest, ytest);
%}

fTree = [0 0 0 0 0 0];
xTree{1} = Xtrain;
yTree{1} = ytrain;

% calculateSplit(fTree, xTree, yTree, n, beta, numThreshPerFeat, deep)
% beta - a parameter to tweak which only splits at end of there is a certain entropy gain
% numThreshPerFeat - the maximum number of thresholds to look at per feature
% deep - is a bit whether or not to use deepening look ahead
%[fTree] =  calculateSplit(fTree, xTree, yTree, 1, .039, 400, 0);
[fTree] =  calculateSplit(fTree, xTree, yTree, 1, beta, numThreshPerFeat, deep);
err_rate = bench(fTree, Xtest, ytest);
