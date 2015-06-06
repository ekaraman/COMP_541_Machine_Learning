%1,2
Xtrain=dlmread('C:\Users\KARAMAN\Google Drive\COURSES\COMP 541 Machine Learning\HW1\hw1code\knnClassify3CTrain.txt',' ','A1..B:');
ytrain=dlmread('C:\Users\KARAMAN\Google Drive\COURSES\COMP 541 Machine Learning\HW1\hw1code\knnClassify3CTrain.txt',' ','C1..C:');
Xtest=dlmread('C:\Users\KARAMAN\Google Drive\COURSES\COMP 541 Machine Learning\HW1\hw1code\knnClassify3CTest.txt',' ','A1..B:');
ytest=dlmread('C:\Users\KARAMAN\Google Drive\COURSES\COMP 541 Machine Learning\HW1\hw1code\knnClassify3CTest.txt',' ','C1..C:');
gscatter (Xtrain(:,1),Xtrain(:,2),ytrain(:,1),'rbg','+*x')
%3
hold on
gscatter (Xtest(:,1),Xtest(:,2),ytest(:,1),'rbg','+*x');
[ypred] = knnClassify(Xtrain, ytrain, Xtest, 10);
error = 0;
for i = 1:size(ypred,1)
    if ypred (i) ~= ytest (i)
        s=140;
        scatter (Xtest(i,1),Xtest(i,2),s,'k')
        error = error+1;
    end
end
%4
XtestGrid = makeGrid2d(Xtrain);
ypredGrid = knnClassify (Xtrain, ytrain, XtestGrid, 1);
% plotLabeledData (XtestGrid, ypredGrid);
%5
error_rate = 1;
hold off
clf;
figure (2)
hold on
error_rate = zeros(20,1);
dfreedom = zeros(20,1);

for i = 1:20
    error = 0;
    [ypred] = knnClassify(Xtrain, ytrain, Xtest, i);
    for j = 1:size(ypred,1)
        if ypred (j) ~= ytest (j)
            error = error+1;
        end
    end
    error_rate(i) = error/size(ypred,1);
    dfreedom(i) = size(ypred,1)/i;
end
semilogx (dfreedom,error_rate);

for i = 1:20
    error = 0;
    [ypred] = knnClassify(Xtrain, ytrain, Xtrain, i);
    for j = 1:size(ypred,1)
        if ypred (j) ~= ytrain (j)
            error = error+1;
        end
    end
    error_rate(i) = error/size(ypred,1);
    dfreedom(i) = size(ypred,1)/i;
end

semilogx (dfreedom,error_rate);

hold off

