% Calculates two material decomposition using matrixfree functions.

% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
clear all;

% Regularization parameter
alpha  = 10000;              

% Choose relative noise level in simulated noisy data
noiselevel = 0.01;

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11    = 42.2057; %Iodine 30kV
c21    = 60.7376; %Iodine 50kV
c12    = 3.044;   %Al 30kV
c22    = 0.994;   %Al 50kV

% Construct phantom. You can modify the resolution parameter N.
N      = 40;
g1     = imresize(double(imread('HY_Al.bmp')),[N N]);
g2     = imresize(double(imread('HY_square_inv.jpg')),[N N]);
g      = [g1;g2];

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; 
angle0  = -90;
ang     = angle0 + [0:(Nang-1)]/Nang*180;

% Simulate noisy measurements; here including inverse crime
m       = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N);
% Construct noisy data
m       = m + noiselevel*max(abs(m(:)))*randn(size(m));

% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T m.
% The positive constant alpha is the regularization parameter.

b = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang,N);

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
x   = b;          % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!

Hg  = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang,N);
Hg = Hg+alpha*g;
r    = b-Hg;
rho(1) = r(:).'*r(:);

% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    w          = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,p,ang,N),ang,N);
    w          = w+alpha*g;
    aS         = rho(kkk)/(p(:).'*w(:));
    g          = g + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    disp([kkk K])
%     figure(2)
%     imshow(g,[]);
%     pause;
% %   recn2 = g(1601:3200);
% %   recn2 = reshape(recn2,N,N);
% %   imshow(recn2,[]);
% %   pause(0.2);
end
recn1 = g(1:(end/2),1:end);
recn2 = g((end/2)+1:end,1:end);

% Determine computation time
comptime = toc;

% Compute relative errors
% Target 1
err_sup1 = max(max(abs(g1-recn1)))/max(max(abs(g1)));
err_squ1 = norm(g1(:)-recn1(:))/norm(g1(:));
% Target 2
err_sup2 = max(max(abs(g2-recn2)))/max(max(abs(g2)));
err_squ2 = norm(g2(:)-recn2(:))/norm(g2(:));

%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(1);
% Original target1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of target1
subplot(2,2,2)
imagesc(recn1);
colormap gray;
axis square;
axis off;
title('M1 BB reco1 ');
% Original target2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['M2 BB reco2, iter=' num2str(K)]);