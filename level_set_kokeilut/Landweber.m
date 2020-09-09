% Reconstruct a slice from real X-ray data using Landweber iteration.
%
% Tatiana Bubba and Samuli Siltanen September 2016

% Load the measurement matrix and the sinogram from
%load LotusData128
clear all;

% Construct phantom. You can modify the resolution parameter N.
M=40;
N=40;
M1 = imresize(double(imread('HY_Al.bmp')),[N N]);
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;
m = radon(M1,ang);

 
% Load precomputed results of the routine (commented above)
% with the particular values of j and Nang above
loadcommand = ['load RadonMatrix_', num2str(M), '_', num2str(Nang), ' angles A'];
eval(loadcommand)

% Compute a Tikhonov regularized reconstruction using
% conjugate gradient algorithm pcg.m
N     = sqrt(size(A,2));
alpha = 10; % regularization parameter
fun   = @(x) A.'*(A*x)+alpha*x;
b     = A.'*m(:);
x     = pcg(fun,b); %1e-06, min(N,40)
%X = pcg(AFUN,B) accepts a function handle AFUN instead of the matrix A.
% AFUN(X) accepts a vector input X and returns the matrix-vector product A*X.

% Compute a Tikhonov regularized reconstruction from only
% 60 projections
% [mm,nn] = size(m);
% ind     = [];
% for iii=1:nn/6
%     ind = [ind,(1:mm)+(6*iii-6)*mm];
% end
% m2    = m(:,1:6:end);
% A     = A(ind,:);
% alpha = 10; % regularization parameter
% fun   = @(x) A.'*(A*x)+alpha*x;
% b     = A.'*m2(:);
% x2    = pcg(fun,b);

% Take a look at the sinograms and the reconstructions
figure(1)
clf
subplot(2,2,1)
imagesc(m)
colormap gray
axis square
axis off
title('Sinogram, 360 projections')
subplot(2,2,3)
imagesc(m)
colormap gray
axis square
axis off
title('Sinogram, 60 projections')
subplot(2,2,2)
imagesc(reshape(x,N,N))
colormap gray
axis square
axis off
title({'Tikhonov reconstruction,'; '360 projections'})
subplot(2,2,4)
imagesc(reshape(x2,N,N))
colormap gray
axis square
axis off
title({'Tikhonov reconstruction,'; '60 projections'})

figure(2)
clf
plotim = reshape(x2,N,N);
plotim = plotim-min(plotim(:));
plotim = plotim/max(plotim(:));
plotim = plotim.^(.7);
imagesc(reshape(x2,N,N))
colormap gray
axis square
axis off