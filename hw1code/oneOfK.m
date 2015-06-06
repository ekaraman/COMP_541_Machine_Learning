function yy = oneOfK(y, K)
% y(n) in {1,...,K}
% yy(n,:) is a K dimensional bit vector, where yy(n, y(n)) = 1
% Example
%>> oneOfK([1 2 1 3], 4)
%     1     0     0     0
%     0     1     0     0
%     1     0     0     0
%     0     0     1     0
     
N = length(y);
yy = zeros(N, K);
ndx = sub2ind(size(yy), 1:N, y(:)');
yy(ndx) = 1;

