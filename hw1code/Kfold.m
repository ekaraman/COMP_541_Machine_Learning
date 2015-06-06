function  [trainfolds, testfolds] = Kfold(N, K)
% Kfold Compute indices for K-fold cross validaiton
% N = num data
% K = num folds, if K=-1, use leave-one-out CV
% [trainfolds{i}, testfolds{i}] = indices of i'th fold
%
% Example:
% [trainfolds, testfolds] = Kfold(100, 3)
% testfolds{1} = 1:33, trainfolds{1} = 34:100
% testfolds{2} = 34:66, trainfolds{2} = [1:33 67:100]
% testfolds{3} = 67:100, trainfolds{3} = [1:66]
% (last fold gets all the left over so has different length)
%
% Example
% [trainfolds, testfolds] = Kfold(100, -1)
% trainfolds{1} = 1:99, testfolds{1} = 100, etc


if K==-1
  for i=1:N
    trainfolds{i} = setdiff(1:N, i);
    testfolds{i} = i;
  end
  return
end

ndx = 1;
for i=1:K
  low(i) = ndx;
  Nbin(i) = fix(N/K);
  if i==K
    high(i) = N;
  else
    high(i) = low(i)+Nbin(i)-1;
  end
  testfolds{i} = low(i):high(i);
  trainfolds{i} = setdiff(1:N, testfolds{i});
  ndx = ndx+Nbin(i);
end

