% Build a matrix for parallel beam tomography measurement.
%
% In this routine, it is possible to choose to things:
% - Parameter "M" giving the resolution of the computational model.
%   The tomographic image has size MxM.
% - Parameter "Nang" specifying the number of tomographic projection
%   directions. The directions span the whole 360 degree circle, and
%   therefore it is a good idea to choose Nang to be an odd number. 
%   (If Nang is even, there will be redundancy as some of the measurements 
%   appear twice.)
%
% The computed tomography matrix is saved to a ".mat" file for use in
% further routines in this folder. The name of the file contains the
% parameter values: for example, the file "RadonMatrix_32_17.mat"
% corresponds to a 32x32 pixel image and 17 projection directions.
% 
% Samuli Siltanen October 2017

%% Choices for the user
% Choose the size of the unknown. The image has size MxM.
M = 40;
% Choose the angles for tomographic projections
Nang = 65; % odd number is preferred

%% Definitions and initializations
% Some definitions
n = M*M;
angles = [0:(Nang-1)]*360/Nang;

% Let's find out the eventual size of matrix A. Namely, the size is
% k x n, where n is as above and k is the total number of X-rays. 
% The number of X-rays depends in a complicated way on Matlab's inner 
% workings in the routine radon.m.
tmp = radon(zeros(M,M),angles);
k   = length(tmp(:));

% Initialize matrix A as sparse matrix
A = sparse(k,n);

%% Construct matrix A column by column. 
% This is a computationally wasteful method, but we'll just do it here. 
% For large values of M or Nang this will take a very long time to compute. 
for iii = 1:n
    % Construct iii'th unit vector
    unitvec = zeros(M,M);
    unitvec(iii) = 1;
    
    % Apply radon.m to a digital phantom having value "1" in exactly one 
    % pixel and vaue "0" in all other pixels
    tmp = radon(unitvec,angles);
    
    % Insert a new column to the tomography matrix
    A(:,iii) = sparse(tmp(:));
    
    % Monitor the run
    if mod(iii,round(n/10))==0
        disp([iii n])
    end
end
%% Save the result to disc 
% The filename contains the parameter values for M and Nang
savecommand = ['save -v7.3 RadonMatrix_', num2str(M), '_', num2str(Nang), ' A M n Nang angles'];
eval(savecommand)