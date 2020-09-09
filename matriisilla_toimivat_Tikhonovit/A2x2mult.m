function [ res ] = A2x2mult(a,c11,c12,c21,c22,g)
%function [ res ] = A2x2mult(a,c11,c12,c21,c22,g)
% This function calculates multiplication A*g for system with two images
% and two materials.
%
% Version 1.0, September 13, 2019
% (c) Salla Latva-�ij� and Samuli Siltanen 
%
% Routine for the two-energies and two materials scheme related to S Siltanen's 
% method for solution of the conjugate gradient Tikhonov algorithm. Computes 
% multiplication A*g of the combination system matrix and the combined target images g1 and g2. 
% 
% Arguments
% a         system matrix for one material measurements
% c11       attenuation coefficient for material one (iodine) imaged with energy E1=30kV (low)
% c12       attenuation coefficient for material two (al) imaged with energy E1=30kV (low)
% c21       attenuation coefficient for material one (iodine) imaged with energy E2=50kV (high)
% c22       attenuation coefficient for material two (al) imaged with energy E1=50kV (high)
% g         reconstruction images of target 1 (aluminum) and target 2
%           (iodine), stacked in vertical vector 
% Returns
% res       vertical vector with length (a*g1*2)
%
% Last revision Salla Latva-Äijö & Samuli Siltanen Nov 2019

% Perform the needed matrix multiplications
g1 = g(1:(end/2));
g2 = g((end/2+1):end);
ag1 = a*g1(:);
ag2 = a*g2(:);

% Calculate the parts needed for block matrix multiplication
res1 = c11*ag1;
res2 = c12*ag2;
res3 = c21*ag1;
res4 = c22*ag2;

% Combine results to the final result
res = [res1 + res2; res3 + res4]; %m
end

