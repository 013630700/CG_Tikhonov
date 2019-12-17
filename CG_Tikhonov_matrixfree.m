% Calculates two material decomposition using matrixfree functions.

% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
% Modified by Salla 5.12.2019
clear all;

%% Choises for the user
% Regularization parameter
alpha1  = 100;              
alpha2  = 10000;
N       = 40;

% Choose relative noise level in simulated noisy data
noiselevel = 0.00001;

% Measure computation time later; start clocking here
tic

% Define attenuation coefficients: Iodine and Al
material1='aluminum';
material2='Iodine';
c11    = 42.2057; %Iodine 30kV
c21    = 60.7376; %Iodine 50kV
c12    = 3.044;   %Al 30kV
c22    = 0.994;   %Al 50kV

% Define attenuation coefficients: Iodine and PVC
% material1='PVC';
% material2='Iodine';
% c11    = 42.2057; %Iodine 30kV
% c21    = 60.7376; %Iodine 50kV
% c12    = 2.096346;%PVC 30kV
% c22    = 0.640995;%PVC 50kV
%% Construct phantom. You can modify the resolution parameter N.
g1     = imresize(double(imread('HY_Al.bmp')),[N N]);
g2     = imresize(double(imread('HY_square_inv.jpg')),[N N]);

% % Normalize original data
% g1=normalize(g1);
% g2=normalize(g2);

% Combine
g      = [g1(:);g2(:)];

% Choose measurement angles (given in degrees, not radians). 
Nang    = 40; 
angle0  = -90;
ang     = angle0 + [0:(Nang-1)]/Nang*180;

% Simulate noisy measurements
m       = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N); 

% Add noise
m       = m + noiselevel*max(abs(m(:)))*randn(size(m));
%m=max(m,0);
%
% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter
b = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang);
% figure(102);
% imshow(b,[]);
%%
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 300;         % maximum number of iterations
g   = b;          % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!


%alpha matriisin tilalle: Reg_mat
Reg_mat = [alpha1*eye(N^2),zeros(N^2);zeros(N^2),alpha2*eye(N^2)];
Hg  = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang) + Reg_mat*g(:);

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
    w          = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,p,ang,N),ang);
    w          = w + Reg_mat*p;
    aS         = rho(kkk)/(p.'*w);
    g          = g + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    %disp([kkk K])
%     figure(2)
%     imshow(g,[]);
%     pause;
% %   recn2 = g(1601:3200);
% %   recn2 = reshape(recn2,N,N);
% %   imshow(recn2,[]);
% %   pause(0.2);
end
recn1 = reshape(g(1:(end/2),1:end),N,N);
recn2 = reshape(g((end/2)+1:end,1:end),N,N);

% % Normalize image
% recn1 = normalize(recn1);
% recn2 = normalize(recn2);

% Determine computation time
comptime = toc;
%% Compute the error
% Square error of reconstruction 1:
%err_sup1 = max(max(abs(g1-recn1)))/max(max(abs(g1)));
err_squ1 = norm(g1(:)-recn1(:))/norm(g1(:));
% Target 2
%err_sup2 = max(max(abs(g2-recn2)))/max(max(abs(g2)));
err_squ2 = norm(g2(:)-recn2(:))/norm(g2(:));

%% Take a look at the results
figure(1);
% Original phantom1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({material1,' Phantom1, matrixfree'});
% Reconstruction of phantom1
subplot(2,2,2)
imagesc(recn1);
colormap gray;
axis square;
axis off;
title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1)]);
% Original target2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({material2,'Phantom2, matrixfree'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['Relative error=' num2str(err_squ2), ', \alpha_2=' num2str(alpha2)]);
%%
figure(6);
% Original phantom1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap jet;
axis square;
axis off;
title({material1,', Phantom1, BB, matrixfree'});
% Reconstruction of phantom1
subplot(2,2,2)
recn1=reshape(g(1:(length(g)/2)),N,N);
imagesc(recn1);
colormap jet;
axis square;
axis off;
title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1), ', \beta=', num2str(beta)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap jet;
axis square;
axis off;
title({material2,', Phantom2, BB, matrix free'});
% Reconstruction of phantom2
subplot(2,2,4)
recn2=reshape(g(length(g)/2+1:end),N,N);
imagesc(recn2);
colormap jet;
axis square;
axis off;
title(['Relative error=' num2str(err_squ2), ', \alpha_2=' num2str(alpha2)]);
toc