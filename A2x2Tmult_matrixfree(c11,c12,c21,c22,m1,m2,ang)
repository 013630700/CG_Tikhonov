function [ res ] = A2x2Tmult_matrixfree(c11,c12,c21,c22,m1,m2,ang)
% function [ res ] = A2x2Tmult_matrixfree(c11,c12,c21,c22,m1,m2,ang)
% This function calculates multiplication AT*m for system with 
% two images and two energies with using iradon.
%
% Version 1.0, September 13, 2019
% (c) Salla Latva-Äijö and Samuli Siltanen 
% 
% Routine for the two-energies and two materials scheme related to S Siltanen's 
% method for solution of the conjugate gradient Tikhonov algorithm. Computes 
% multiplication A^T*m of the combination system matrix transpose (without 
% the matrix)and the combined measurements (=sinograms) m1 
% and m2, which refer to material one (iodine) and material two (aluminum).
% 
% Arguments
% c11       attenuation coefficient for material one (iodine) imaged with energy E1=30kV (low)
% c12       attenuation coefficient for material two (al) imaged with energy E1=30kV (low)
% c21       attenuation coefficient for material one (iodine) imaged with energy E2=50kV (high)
% c22       attenuation coefficient for material two (al) imaged with energy E1=50kV (high)
% m1        measurement (sinogram) of material one
% m2        measurement (sinogram) of material one
% ang       measurement angles used in X-raying the target
% Returns
% res       gives unfiltered backprojection of the images and gives unfiltered
%           images one top of each another
% 
% Last revision Salla Latva-Äijö Sep 2019


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



% Old version (saved for comparison)
%function [ res ] = A2x2Tmult( a,c11,c12,c21,c22,m1,m2 )
%% AT*m
%res = [c11*a.'*m1 + c21*a.'*m2; c12*a.'*m1 + c22*a.'*m2];
%end
