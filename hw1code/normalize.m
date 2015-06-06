function [M, z] = normalize(A, dim)
% NORMALIZE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(A) normalise the whole array, where c is the normalizing constant
% [M, c] = normalise(A, dim)
% If dim is specified, we normalise the specified dimension only.
% dim=1 means each column sums to one
% dim=2 means each row sums to one

if nargin < 2
  z = sum(A(:));
  % Set any zeros to one before dividing
  % This is valid, since s=0 iff all A(i)=0, so we will get 0/1=0
  s = z + (z==0);
  M = A / s;
elseif dim==1 % normalize each column
  z = sum(A);
  s = z + (z==0);
  M = A ./ repmat(s, size(A,1), 1);
elseif dim==2 % normalize each row
  z = sum(A,2);
  s = z + (z==0);
  M = A ./ repmat(s, 1, size(A,2));
else
  % Keith Battocchi - v. slow because of repmat
  z=sum(A,dim);
  s = z + (z==0);
  L=size(A,dim);
  d=length(size(A));
  v=ones(d,1);
  v(dim)=L;
  c=repmat(s,v');
  M=A./c;
end


