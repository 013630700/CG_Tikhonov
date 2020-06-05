% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
%Modified by salla 29.5.2020
clear all;
% Choose resolution
N = 40;
% Regularization parameter
alpha = 10;              
% Measure computation time later; start clocking here
tic
% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives
% target = M2;
% P  = length(radon(target,0));
% M  = length(ang);
% A = sparse(M*P,N^2);
% 
% % Construct measurement matrix column by column. The trick is to construct
% % targets with elements all 0 except for one element that equals 1.
% for mmm = 1:M
%     for iii = 1:N^2
%         tmpvec                  = zeros(N^2,1);
%         tmpvec(iii)             = 1;
%         A((mmm-1)*P+(1:P),iii) = radon(reshape(tmpvec,N,N),ang(mmm));
%         if mod(iii,100)==0
%             disp([mmm, M, iii, N^2])
%         end
%     end
% end
% 
% % Test the result
% Rtemp = radon(target,ang);
% Rtemp = Rtemp(:);
% Mtemp = A*target(:);
% disp(['If this number is small, then the matrix A is OK: ', num2str(max(max(abs(Mtemp-Rtemp))))]);
% 
% % Save the result to file (with filename containing the resolution N)
% eval(['save RadonMatrixE2', num2str(N), ' A ang target N P Nang']);

% Load radonMatrix
eval(['load RadonMatrixE2', num2str(N), ' A ang N P Nang']);
%% Construct target
f1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [N N]);
f2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [N N]);
f1=f1(:,:,1);
f2=f2(:,:,1);
target1=f1;
target2=f2;
f1 = f1(:);
f2 = f2(:);
% Define constants
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044; %Al 30kV
c22 = 0.994;%Al 50kV

%% Construct the different measurements
% Energy E1
m11 = c11*A*f1;
m21 = c21*A*f1;
%Energy E2
m12 = c12*A*f2;
m22 = c22*A*f2;
%%
% In the simulated measurement, we combine the measurements as follows
m1 = m11 + m12;
m2 = m21 + m22;

%m = [m1; m2];
%%
% We combine the matrix A
%A = [c11*A c12*A; c21*A c22*A];
% You have now the non-noisy measurements 

% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter.o
b     = A.'*m1(:);

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
x   = b;          % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters
% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!
Hx     = (A.')*(A*x) + alpha*x; 
r      = b-Hx;
rho(1) = r.'*r;
% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    w          = (A.')*(A*p) + alpha*p;
    a          = rho(kkk)/(p.'*w);
    x          = x + a*p;
    r          = r - a*w;
    rho(kkk+1) = r.'*r;
    disp([kkk K])
end
recn1 = reshape(x,N,N);
%%
% You have now the non-noisy measurements 

% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter.o
b2     = A.'*m2(:);

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K2   = 80;         % maximum number of iterations
x2   = b2;          % initial iterate is the backprojected data
rho2 = zeros(K2,1); % initialize parameters
% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!
alpha2= 10;
Hx2     = (A.')*(A*x2) + alpha2*x2; 
r2      = b2-Hx2;
rho2(1) = r2.'*r2;
% Start iteration
for kkk2 = 1:K2
    if kkk2==1
        p2 = r2;
    else
        beta2 = rho2(kkk2)/rho2(kkk2-1);
        p2    = r2 + beta2*p2;
    end
    w2          = (A.')*(A*p2) + alpha2*p2;
    a2          = rho2(kkk2)/(p2.'*w2);
    x2          = x2 + a2*p2;
    r2          = r2 - a2*w2;
    rho2(kkk2+1) = r2.'*r2;
    disp([kkk2 K2])
end
recn2 = reshape(x2,N,N);
% Determine computation time
comptime2 = toc;

% Compute relative errors
%target1 = target1./max(max(target1));
%recn1 = recn1./max(max(recn1));
err_sup1 = max(max(abs(target1-recn1)))/max(max(abs(target1)));
err_squ1 = norm(target1(:)-recn1(:))/norm(target1(:));

%target2 = target2./max(max(target2));
%recn2 = recn2./max(max(recn2));
err_sup2 = max(max(abs(target2-recn2)))/max(max(abs(target2)));
err_squ2 = norm(target2(:)-recn2(:))/norm(target2(:));

% Save result to file
%eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% View the results
%XRMG_Tikhonov_plot(N)
figure(1);
clf
imagesc(recn1);
colormap gray
axis equal
axis off
title(['Tikhonov: error ', num2str(round(err_squ1*100)), '%'])

%% Take a look at the results
figure(3);
% Original target1
subplot(2,2,1);
imagesc(reshape(target1,N,N));
colormap gray;
axis square;
axis off;
title({'target1 +matrix, PVC'});
% Reconstruction of target1
subplot(2,2,2)
imagesc(recn1);
colormap gray;
axis square;
axis off;
title(['recn1 Relative error=', num2str(err_squ1), ', \alpha=', num2str(alpha)]);
% Original target2
subplot(2,2,3)
imagesc(reshape(target2,N,N));
colormap gray;
axis square;
axis off;
title({'target2 +matrix, Iodine'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['recn2 Relative error=' num2str(err_squ2), ', iter=' num2str(K2)]);