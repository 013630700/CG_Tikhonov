function [ res ] = A2x2mult( a,c11,c12,c21,c22,g1,g2)
% A*g
Ag1 = c11*a*g1+c12*a*g2;
Ag2 = c21*a*g1+c22*a*g2;
% ATAg
ATAg = [a'*Ag1*c11+a'*Ag2*c21; a'*Ag1*c12+ a'*Ag2*c22];
res = ATAg;
end
