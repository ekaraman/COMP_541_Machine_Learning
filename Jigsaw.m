clear all;

% read an image
I = double(imread('C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\128_Dog.png'))/255;

% the size of the jigsaw 
jSize = [32 32 3];

%Initialize jigsaw mean, jigsaw variance and offset map L matrices
jMean = zeros(jSize);
jVar = zeros(jSize);
L = zeros (size(I(:,:,1))); % size of 128 x 128

%Acoording to the paper:
%For our experiments,
%we fix the hyperparameters mean_zero to .5, beta to 1, b to three times the 
%inverse data variance and a to the square of b.

%constants
beta = 1;
mean_0 = 0.5;

%Calculate mean and variance of image elements in order to find a and b
%constants

Isize = size(I);
sumX = sum(sum(I, 1), 2);
sumXX = sum(sum(I.^2, 1), 2);
pixelMean = sumX ./ prod(Isize(1:end-1));
pixelVar = sumXX ./ prod(Isize(1:end-1)) - pixelMean.^2;    
pixelStd = sqrt(sumXX ./ prod(Isize(1:end-1)) - pixelMean.^2);

%Initialize Gamma distribution parameters
b = 3 * (1 / pixelVar);
a = b.^2;

%JIGSAW INITIALIZATION STEP
%According to the paper:

%First, the jigsaw is initialised by setting the precisions "Lambda" to
%the expected value under the prior b=a and the "means" to Gaussian noise
%with the same mean and variance as the data.

%Initialize Jigsaw Variance
for i = 1 : Isize(end)
    jVar(:,:,i) = b(i) / a(i);
end

%Initialize Jigsaw Mean
for i = 1 : Isize(end)
    jMean (:,:,i) = random('norm', pixelMean(1,i), pixelVar(1,i), jSize(1) , jSize(1));
end


%Find size of Data_Cost matrix
iASize = Isize(1)^2; %column size of data cost matrix = 16384
iALSize = jSize (1)^2; % row size of data cost matrix = 1024

% DATA COST Nedir?
% Data Cost: Resimdeki her bir pixele jigsawdaki pixellerin atanmasý
% neticesinde elde edilen maliyet. Yani, resim 128*128 = 16384 pixel,
% jigsaw 32x32 1024 pixel; data cost matrixi 1024x16384 boyutunda. Diðer
% bir deyiþle her pixele jigsawdaki 1024 pixeli tek tek atayýp maliyeti
% hesaplýyoruz. Paperda data costtan bahsetmiyor. Ben data cost olarak
% (image_pixel_value - jigsaw_pixel_value)^2 deðerini kullandým.

%Set neighbours for label matrix
%Graph cut algoritmasýnda label tanýmlamak zorundasýnýz. Paperda
%neighborhood ilþkisi bir yerde þu þekilde ifade edilmiþ:

%We want the images to consist of coherent pieces of the jigsaw, and 
%so we define a Markov random field on the offset map to encourage
%neighboring pixels to have the same offsets. 

%Markov Rrandom field 4 connected gridde tanýmlanmýþ ve "interaction
%potential" olarak Pott's model kullanýlmýs. Potts modelde eger label
%degeri farklý ise sabit bir deðer ile cezlandýrýyorsunuz. Paperda sabit
%olarka alpha = 5 (her bir RGB kanal için) alýnmýþ. Ben bu degeri asagýda
%smoot cost matrixini hesaplarken kullandým.

% Aþaðýda 4  conencted grid binary matrixte tanýmlanmýþtýr. matrix
% 16384x16384 boyutundadýr. Graph cut algorithmasý kodu bu þekilde
% kullandýðý için neighborhood bu þekilde tanýmlanmýþtýr.

[r,c] = size(I(:,:,1));                   %# Get the matrix size
diagVec1 = repmat([ones(c-1,1); 0],r,1);  %# Make the first diagonal vector
                                          %#   (for horizontal connections)
diagVec1 = diagVec1(1:end-1);             %# Remove the last value
diagVec2 = ones(c*(r-1),1);               %# Make the second diagonal vector
                                          %#   (for vertical connections)
adj = diag(diagVec1,1)+...                %# Add the diagonals to a zero matrix
      diag(diagVec2,c);
adj = adj+adj.';                          %'# Add the matrix to a transposed
                                          %#   copy of itself to make it
                                          %#   symmetric
%adj
triuAdj = triu(adj);
%Kod sparse matrix kullanýyor
Sadj = sparse(triuAdj);
%End of setting of neighborhood 

%Graph cut kodu için Image pixel matrixi 1x16384 boyutunda matrix haline
%getirilmiþtir.
Iarray = reshape (I,[1,iASize,3]);


%Beginning of EM algorithm
for em = 1 : 1 %Beginning of EM iteration

disp ('EM iteration nu=');
em
%Update jarray for calculating data cost
jarray = reshape (jMean,[1,iALSize,3]);
dataCost = zeros (iALSize, iASize);
disp ('setting data cost matrix');

%Data cost yukarýda anlatldýðý gibi hesaplanmaktadýr.
%datacost(i,j)=(image_pixel_value - jigsaw_pixel_value)^2
%Her bir pixel için RGB kanallarda elde edilen data cost deðerleri
%toplanmýþtýr.
for i = 1 : iALSize
    for j = 1 : iASize
        for k = 1 : 3
            dataCost(i,j) = (Iarray(1,j,k) - jarray(1,i,k))^2 + dataCost(i,j);
        end
%Grap cut algoritmasýnda data cost olarak sadece integer deðerler kabul
%ediyor. Ben de bu nedenle elde ettiðim data cost deðerlerini 1000 ile
%çarparak en yakýn integer deðere yuvarladým.

        dataCost(i,j) = dataCost(i,j) * 1000;
        
    end
end

%graphcut integer32 kullanýyor
dataCost = int32(dataCost);

%Graph cut algorithmasý bir handle yaratýyor.
h = GCO_Create(iASize,iALSize);

%Grap cut algoritmasý hesapladýðýmýz datacost matrixini set ediyor.
disp ('setting data cost matrix');
GCO_SetDataCost(h,dataCost);

%Paperda jigsawun coherent pieces'den oluþmasý için 4 connected gird 
%üzerinde markov random field tanýmlandýðýndan bahsediyor ve "interaction
%potential" olarak Pott's model kullanýlmýs. Potts modelde eger label
%degeri farklý ise sabit bir deðer ile cezlandýrýyorsunuz. Paperda sabit
%olarak alpha = 5 (her bir RGB kanal için) alýnmýþ. Bu nedenle aþaðýda 
%smooth cost olarak komþu olmyan deðerler için 5 olacak þekilde
%hesaplanmýþtýr.
%Bu yaklaþýmýn doðrum olmayabileceðini yeni fark ettim:
%paperda ayný label deðeri olmayacaklar için cezalandýrýyor ben burda komþu
%olmayanlarý cezalandýrýyorum. Ayrýca ilk basta label belli deðil bu
%durumda ne alýnacak. Ayrýca ben burda smooth costu 5 alýyor ama data
%costlar 1000 civarýnda buda etkiliyor olabilir mi?

disp ('setting smooth cost matrix');
%Calculate Smooth Cost
smoothCost = int32(zeros (iALSize, iALSize));
for i = 1 : iALSize
    for j = 1 : iALSize
        if (i == j) 
            smoothCost(i,j) = 0;
        else
            smoothCost(i,j)  = 5;
        end
    end
end

%Smooth cost graph cut algoritmasý için set ediliyor.
disp ('setting smooth cost');
GCO_SetSmoothCost(h,smoothCost);

%Neighborhood graphcut için set ediliyor.
disp ('setting neighborhood matrix');
GCO_SetNeighbors(h,Sadj);

%Expansion step ile en uygun label deðerleri ataýyor.
disp('Expansion step begin');
%GCO_SetVerbosity(h,2);
GCO_Expansion(h);

%Hesaplanan label deðerleri label vektorüne atanýyor (16384x1)
disp('Show labels');
label = GCO_GetLabeling(h);

%Update Jigsaw Mean
disp('Update Jigsaw Mean');

%Bir jigsaw pixeli resimde birden çok pixele atanabilir.

%Find the set of image pixels that are mapped to the jigsaw pixel z
sI = [Isize(1), Isize(1)];% [128 128]
sJ = [jSize(1), jSize(1)];% [32 32]


for i = 1: iALSize 
    %Herbir jigsaw pixelinin hangi pixellere atandýðý bulunur 
    xIndex = find (label == i);
    %tek boyutlu label indexi 2 boyuta çevrilir
    [Xj,Yj] = ind2sub(sJ,i);
   
    if (isempty(xIndex) == 0)
        %Bir jigsaw pixeline kaç tane pixel atandýðý bulunur
        xDim = size (xIndex,1);
        for j = 1 : 3
            %X(z) is the set of image pixels that are mapped to the jigsaw 
            %pixel z across all images. Burada xZ jigsaw mean ve
            %varyansýnýn update edilmeinde kullanýlan bahse konu jigsaw
            %pixeline atanan image pixel deðerleriinin toplamý olarak
            %kullanýlmýþtýr. Paperda mean ve varyasýn update edilmesi
            %eþitliðine bakýldýðýnda daha net bir þekilde anlaþýlabilir.
            xZ = 0;
            xZ2 = 0;
            for k = 1 : xDim
                %Convert 1D label index to 2D image index 
                %Atanan pixel deðerlerinin indexi array index olduðu için
                %burada 2 boyutlu matrix indexine çevrilir.
                [Xi,Yi] = ind2sub(sI,xIndex(k));
                %Bahse konu jigsawa atanan image pixellerinin deðerlerinin
                %toplamý ve pixel deðerlerinin karelerinin toplamý jigsaw
                %mean ve varyans deðerlerinin update edilmesi için hesaplanýr. 
                xZ = xZ + I (Xi,Yi,j);
                xZ2 = xZ + (I (Xi,Yi,j)^2);
            end
            %jigsaw mean ve varyans deðerleri update edilir.
            jMean(Xj,Yj,j) = (((beta * mean_0) + xZ) / (beta + xDim));
            jVar(Xj,Yj,j) = ((b(:,:,j) + (beta * mean_0^2) - ((beta + xDim) * jMean(Xj,Yj,j)^2) + xZ2 ) / (a(:,:,j) + xDim));
        end
    end
end

end %end of EM iteration
imwrite(jMean,'jigsaw.png');
%figure, imagesc(jMean), title('Jigsaw');




