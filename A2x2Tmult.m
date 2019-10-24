function [ res ] = A2x2Tmult( A,c11,c12,c21,c22,m1,m2 )
% Make combination matrix A^T and multiply with vector m
AT = [c11*A.' c12*A.'; c21*A.' c22*A.'];
m = [m1(:);m2(:)];
res = AT*m;
end