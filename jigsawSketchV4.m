%Read sketh xml data and get idm features.
clear all;
% Set image and log directories
date = datestr(now,30);
logFile = ['log_', date, '.txt'];
logDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\Logs';

%Open log file
fileID = fopen([logDir,'\',logFile],'w');
fprintf(fileID,'%s\n',date);

jigsawPath = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Sketch\JigsawResult';
dataPath = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Sketch\TestData';
fileXml = [dataPath,'\*.xml'];
filedir = rdir(fileXml);
sketchXml = read_sketch(filedir(5,1).name);
[row, strokeSize] = size(sketchXml);
feat =zeros(strokeSize,720);

%Plot coordinates
figNo = 1;
%axis off;
set(gca,'visible','off')
hFig = figure(figNo);
figNo = figNo + 1;
set(hFig,'Position',[0, 0, 700, 1200]);
axis([0 1200 -700 0]);
figure(hFig);
hold on;
for i = 1 : size(sketchXml,2)
    plot(sketchXml(1,i).coords(:,1), -sketchXml(1,i).coords(:,2));
end
hold off;
%close(hFig);

for i = 1 : strokeSize
  %feat(i,:) = scontext(sketchXml(1,i), 3, 12, 50);
  %feat(i,:) = image_zernike(sketchXml(1,i), 12);
  feat(i,:)=idm(sketchXml(1,i),50,10,4);
  %feat(i,:)= feat(i,:) * 1000;
end

nrow = size (feat,1);
ncol = size (feat,2);
featSize = nrow;

j1D = 16;
jSize = [j1D, j1D, ncol];
%Initialize jigsaw mean
jMean = zeros(jSize) - 1;

featMean = zeros (1,ncol);
featStd = zeros (1,ncol);
featVar = zeros (1,ncol);

for i = 1 : ncol
    featMean (1,i) = mean2(feat(:,i));
    featStd (1,i) = std2 (feat(:,i));
    featVar (1,i)= featStd (1,i) ^ 2;
end

%Initialize Jigsaw Mean
for i =  1 : j1D
    for j =  1 : j1D
        for k = 1 : ncol
            done = false;
             while (~done)
                 jMean(i,j,k) = random('norm', featMean(1,k), featStd(1,k));
                 if (jMean(i,j,k) >= 0)
                     done = true;
                 end
             end
        end
    end
end

% %Set fully connected grid
grid = int32((ones(nrow,nrow)) * 1);

%TEST%
%grid = int32((ones(nrow,nrow)) * 0);
%TEST%

grid(1:nrow + 1:nrow * nrow) = 0;
%Set 4 connected grid of image pixels. 4 connected grid defined in
%Isize1DxIsize1D matrix (eg. if image size is 128x128, 4 conected grid 
%defined in 16384x16384 matrix) because alpha expansion grap cut code
%define neigberhood of pixels as this way.
                     %# Get the matrix size
% size=2;
% diagVec1 = repmat([ones(size-1,1); 0],size,1);    %# Make the first diagonal vector
%                                             %# (for horizontal connections)
% diagVec1 = diagVec1(1:end-1);               %# Remove the last value
% diagVec2 = ones(size*(size-1),1);                 %# Make the second diagonal vector
%                                             %#   (for vertical connections)
% adj = diag(diagVec1,1)+...                  %# Add the diagonals to a zero matrix
%       diag(diagVec2,size);
% adj = adj+adj.';                            %# Add the matrix to a transposed
%                                             %# copy of itself to make it
%                                             %# symmetric
%Get upper triangular part of adj matrix 

grid = triu(grid);
%Set weights of neighbourhood edges
%w = 1000;
%grid = grid .* w;

dist=zeros(strokeSize,strokeSize);

%Euclidian distance from between middle points of strokes
centers = zeros(strokeSize,2);
 for i = 1 : strokeSize
     middlePoint1 = ceil(sketchXml(1,i).npts / 2);
     x1 = sketchXml(1,i).coords(middlePoint1,1);
     y1 = sketchXml(1,i).coords(middlePoint1,2);
     centers(i,1) = x1;
     centers(i,2) = y1;
     for j = 1 : strokeSize
         if ((i ~= j) && (j > i))
             middlePoint2 = ceil(sketchXml(1,j).npts / 2);
             x2 = sketchXml(1,j).coords(middlePoint2,1);
             y2 = sketchXml(1,j).coords(middlePoint2,2);
             dist(i,j) = sqrt((x2 - x1)^2 + (y2 - y1)^2);
         end
     end
 end

%find nodes in searchBox for each nodes and set displacement for each node
%in search box.
displacement = zeros (strokeSize, strokeSize,2);
for i = 1 : strokeSize
    for j = 1 : strokeSize
        %if ((i ~= j) && (j > i))
        if ((i ~= j))
            xq = centers (j,1);
            yq = centers (j,2);   
            displacement(i,j,1) = centers (j,1) - centers(i,1);
            displacement(i,j,2) = centers (j,2) - centers(i,2);
        end
    end
end

%dist = int32(floor(dist .* 1000));
%Find distance between strokes.
%Set weights of neighbourhood edges
%w = 100;

% G = graph(double(grid));
% plot(G);

%Set variance matrix between each node
checkDispVar = zeros (strokeSize,strokeSize);
for i = 1 : strokeSize
    for j =  1 : strokeSize
        if (i ~= j)
            disp1 = displacement(i,j,:);
            for k =  i+1 : strokeSize
                for m = 1 : strokeSize
                    if ~((i == k) && (j == m)) && (k ~= m)
                        disp2 = displacement(k,m,:);
                        tmpVar = var([disp1; disp2]);
                        meanVar = sum(tmpVar,3)/2;
                        if (meanVar < 300)
                            featDist = zeros(1,2);
                            featDist(1) = sqrt(sum(((feat(i,:) - feat(k,:)).^2),2));
                            featDist(2) = sqrt(sum(((feat(j,:) - feat(m,:)).^2),2));
                            meanFeatDist = mean(featDist,2);
                            varFeatDist = var(featDist);
                            if (meanFeatDist <= 6) && (varFeatDist < 0.4)
                                %checkDispVar(i,j) = 1;
                                %checkDispVar(k,m) = 1;
                                if (j > i) && (m > k)
                                    grid(i,j) = 1000;
                                    grid(k,m) = 1000;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


%Alpha expansion code uses sparse matrix
Sgrid = sparse(double(grid));

%Set Smooth Cost matrix (label size X label size)
%Potts Model
smoothCost = ones(j1D * j1D,j1D * j1D);
smoothCost(1:(j1D * j1D) + 1:(j1D * j1D) * (j1D * j1D)) = 0;

%Truncated linear model
% smoothCost = zeros(j1D * j1D,j1D * j1D);
% fixCost = 3;
% for i = 1 : j1D*j1D
%     flag = 1;
%     for j = i + 1 : j1D*j1D
%         if (flag <= fixCost)
%             smoothCost(i,j) = flag;
%             flag = flag + 1;
%         else
%             smoothCost(i,j) = fixCost;
%         end
%     end
% end
% smoothCost = smoothCost + smoothCost';

%Set Label offset matrix
labelSize = j1D * j1D;
offset = zeros(labelSize,2);

%set offset values for feat(1,1)
%Namely, finding offset values of assigning pixel(1,1) each jigsaw pixel.
% index = 1; 
% for i = 1 : j1D
%     for j = 1 : j1D
%         offset (index,1) = j;
%         offset (index,2) = i;
%         index = index + 1;
%     end
% end
index = 1; 
for i = 1 : j1D
    for j = 1 : j1D
        offset (index,1) = 1 - i;
        offset (index,2) = 1 - j;
        index = index + 1;
    end
end

%Initialize energy
E_old = 1;
E_new = 0;
em = 1;
flag = 1;

%Beginning of EM algorithm
while (flag)
    fprintf(fileID,'%s\n',['###############   EM iteration nu:   ',num2str(em),'    #################']);
    %Set data cost matrix
    fprintf(fileID,'%s\n','setting data cost matrix');
    dataCost = zeros (labelSize, featSize);
    for i = 1 : labelSize
        i
        for j = 1 : featSize
            %Convert offset value to jigsaw index
            IX = centers(j,1);
            IY = centers(j,2);
            jX = mod ((IX - offset (i,1)),j1D);
            if (jX == 0) 
                jX = j1D;
            end
            jY = mod ((IY - offset (i,2)),j1D);
            if (jY == 0) 
                jY = j1D;
            end
            for k = 1 : ncol   
                if (jX < 0 || jX > j1D || jY < 0 || jY > j1D)
                    fprintf(fileID,'%s\n','index error');
                end
                dataCost(i,j) = (feat(j,k) - jMean(jX,jY,k))^2 + dataCost(i,j);
            end
            dataCost(i,j) = dataCost(i,j);
            dataCost(i,j) = int32(floor(dataCost(i,j)));
            dataCost(i,j) = int32(floor(dataCost(i,j) .* 10000));
        end
    end
    
    
    %Create graph cut handle
    fprintf(fileID,'%s\n','Create grap cut handle');
    h = GCO_Create(featSize,labelSize);
    
    %Set data cost matrix for alpha expansion graph cut
    fprintf(fileID,'%s\n','setting data cost matrix');
    GCO_SetDataCost(h,dataCost);
    
    %Setting Smoothcost
    %GCO_SetSmoothCost(h, smoothCost);
    
    %Setting Neighborhood relation
    fprintf(fileID,'%s\n','setting neighborhood relation matrix');
    GCO_SetNeighbors(h,Sgrid);
    
    %Start expanssion so as to assign labels
    fprintf(fileID,'%s\n','Expansion step begin');
    GCO_Expansion(h);
    %GCO_Swap(h);
    
    %Assign optimized label values for each pixel to label matrix (16384x1)
    fprintf(fileID,'%s\n','Set labels');
    label = GCO_GetLabeling(h);
    %Get optimized energy
    fprintf(fileID,'%s\n','Get optimized energy');
    [E_new D S] = GCO_ComputeEnergy(h);
    fprintf(fileID,'%s\n','Energy computed.');
    fprintf(fileID,'%s\n',['Total Energy = ', num2str(E_new)]);
    fprintf(fileID,'%s\n',['Data Cost Energy = ', num2str(D)]);
    fprintf(fileID,'%s\n',['Smooth Cost Energy = ', num2str(S)]);
    
    %Check While loop terminate case
    if (em == 1)
        E_old = E_new;
        E_new = 0;
    else
        if (E_new < E_old)
            E_old = E_new;
        else
            flag = 0;
            fprintf(fileID,'%s\n','#################################');
            fprintf(fileID,'%s\n','##########    CONVERGED      ##########');
            fprintf(fileID,'%s\n',['#######  EM = ', num2str(em), '   ########']);
        end
    end
    
    %Update Jigsaw mean
    fprintf(fileID,'%s\n','Updating jMean and jVar');
    jigsawLabel = zeros (j1D,j1D);
    jigsawAssignedFeat = zeros (j1D,j1D,ncol);
    for i = 1 : featSize
        %Convert 1D pixel to image 2D index
        IX = centers(i,1);
        IY = centers(i,2);
        zX = mod((IX - offset(label(i),1)),j1D);
        if (zX == 0) 
            zX = j1D;
        end
        zY = mod((IY - offset(label(i),2)),j1D);
        if (zY == 0) 
            zY = j1D;
        end
        jigsawLabel(zX,zY) = jigsawLabel(zX,zY) + 1;
         for j = 1 : ncol
            jigsawAssignedFeat(zX,zY,j) = jigsawAssignedFeat(zX,zY,j) + feat(i,j);
         end
    end
    
    %Update Jigsaw mean
    for i = 1 : j1D
        for j = 1 : j1D
            for k = 1 : ncol
                if (jigsawLabel(i,j) > 0)
                    jMean(i,j,k) = ((jigsawAssignedFeat(i,j,k)) / (jigsawLabel(i,j)));
                %else
                    %jMean(i,j,k) = 0;
                end
            end
        end
    end
    
    fprintf(fileID,'%s\n','jMean is updated')
    
    fprintf(fileID,'%s\n',['###############   EM iteration nu:   ',num2str(em),'  finished #################']);
    
    %Find nodes assigned to same labels
    sameLabels = find (jigsawLabel>1);
    for i = 1 : size(sameLabels)
        nodesAssignedtoSameLabels = find (label == sameLabels(i));
    end
    
    em = em + 1;
    label
end

fprintf(fileID,'%s\n','Job completed, pls check reconstructed image.');

%Draw labels

cd(jigsawPath);
delete('*.jpg');
uniqLabels = unique(label);
subPlotNo = (j1D*j1D)+2;
for i = 1: size(uniqLabels,1)
    tmpLabels = find (label == uniqLabels(i));
    hFig = figure(figNo);
    figNo = figNo + 1;
    set(hFig,'Position',[0, 0, 1200, 700]);
    set(gca,'visible','off')
    axis([0 1200 -700 0]);
	figure(hFig);
    hold on;
    for j = 1 : size(tmpLabels)
        plot(sketchXml(1,tmpLabels(j)).coords(:,1), -sketchXml(1,tmpLabels(j)).coords(:,2));
    end
    plotName = ['Label ',int2str(uniqLabels(i)),'.jpg'];
    plotTitle = ['Label ',int2str(uniqLabels(i))];
    title(plotTitle);
    saveas(gcf,plotName);
    hold off;
    close(hFig);
%     img = imread(plotName);
%     hSFig = figure(subPlotNo);
%     figure(hSFig);
%     hold on;
%     subplot(double(j1D),double(j1D),double(uniqLabels(i))), subimage(img);
%     hold off;
end

fclose(fileID);

%save('jigsawsketch.mat');