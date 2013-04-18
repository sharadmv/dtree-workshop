load('spamData.mat');

fTree = [0 0 0 0 0 0];
xTree{1} = Xtrain;
yTree{1} = ytrain;

[fTree] =  calculateSplit(fTree, xTree, yTree, 1, .039);
bench(fTree, Xtest, ytest);




%fTrees = randomTrees(Xtrain, ytrain, 100, .04);
%benchRandomTrees(fTrees, Xtest, ytest);

