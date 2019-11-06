function [ res ] = A2x2Tmult( a,c11,c12,c21,c22,m1,m2 )
% Make combination matrix A^T and multiply with vector m
AT = [c11*a.' c21*a.'; c12*a.' c22*a.'];
m = [m1; m2];
res = AT*m;
end
