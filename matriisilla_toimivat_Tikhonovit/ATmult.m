function [ res ] = ATmult( A,m )
% ATmult counts the transpose of matrix A and multiplies it with vector m
res = A.'*m(:);
end

