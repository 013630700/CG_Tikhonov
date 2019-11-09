function [ res ] = A2x2Tmult( a,c11,c12,c21,c22,m1,m2 )
% AT*m
res = [c11*a.'*m1 + c21*a.'*m2; c12*a.'*m1 + c22*a.'*m2];
end
