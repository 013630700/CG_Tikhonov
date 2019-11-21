function [ res ] = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang,N)
% function [ res ] = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang,N)
% This function calculates multiplication AT*m for system with 
% two images and two energies with using iradon.
%
% Version 1.0, September 13, 2019
% (c) Salla Latva-Äijö and Samuli Siltanen 
% 
% Routine for the two-energies and two materials scheme related to S Siltanen's 
% method for solution of the conjugate gradient Tikhonov algorithm. Computes 
% multiplication A^T*m of the combination system matrix transpose (without 
% the matrix)and the combined measurements m, which refer to measurements of 
% material one (iodine) and material two (aluminum).
% 
% Arguments
% c11       attenuation coefficient for material one (iodine) imaged with energy E1=30kV (low)
% c12       attenuation coefficient for material two (al) imaged with energy E1=30kV (low)
% c21       attenuation coefficient for material one (iodine) imaged with energy E2=50kV (high)
% c22       attenuation coefficient for material two (al) imaged with energy E1=50kV (high)
% m         measurement (sinogram) of material one
% ang       measurement angles used in X-raying the target
% N         ?xN is the size of measurement m1 and m2
% Returns
% res       gives unfiltered backprojection of the images and gives unfiltered
%           images one top of each another
% 
% Last revision Salla Latva-Äijö Sep 2019
%m  = m(:);
m1 = m(1:(end/2));
m1 = reshape(m1, [61 N]);
m2 = m((end/2+1):end);
m2 = reshape(m2, [61 N]);

corxn = 7.65; % Incomprehensible correction factor

% Perform the needed matrix multiplications. Now a.' multiplication has been
% switched to iradon
am1 = iradon(m1,ang,'none');
am1 = am1(2:end-1,2:end-1);
am1 = corxn*am1;

am2 = iradon(m2,ang,'none');
am2 = am2(2:end-1,2:end-1);
am2 = corxn*am2;

% Compute the parts of the result individually
res1 = c11*am1;
res2 = c21*am2;
res3 = c12*am1;
res4 = c22*am2;

% Collect the results together
res = [res1 + res2; res3 + res4];
end