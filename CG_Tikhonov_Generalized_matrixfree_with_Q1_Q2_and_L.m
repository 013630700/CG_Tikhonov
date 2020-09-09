% Calculates two material decomposition using matrixfree functions. The
% idea here is that we use spot operator to reduce memory issues. We have
% added new regularization term and combined it to Generalized Tikhonov
% regularization, for improving the end result.

% Example computations related to X-ray tomography. Here we apply Generalized Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
% Modified by Salla 5.12.2019

clear all;
% Measure computation time
tic
%% Choises for the user
% Choose the size of the unknown. The image has size NxN.
N       = 512;
% Regularization parameter
alpha  = 5000;%10;%80;             
beta  = 200;%89;%150; %10000
% Choose relative noise level in simulated noisy data
noiselevel = 0.01;
iter = 42;
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang     = angle0 + [0:(Nang-1)]/Nang*180;

%% Define attenuation coefficients: Iodine and Al
% material1='aluminum';
% material2='Iodine';
% Attenuation coefficients fromNIST database. (Divided by rho)
c11    = 1.491; %PVC 30 30kV
c12    = 8.561; %Iodine 30kV
c21    = 0.456; %PVC 50kV
c22    = 12.32;   % Iodine 50kV

%% Construct phantom. You can modify the resolution parameter N.

M1 = imresize(double(imread('material1.png')), [N N]);
M2 = imresize(double(imread('material2.png')), [N N]);
% Select only one color channel
M1=M1(:,:,1);
M2=M2(:,:,1);
% Vektorize
g1      = M1(:);
g2      = M2(:);
% Combine
g      = [g1(:);g2(:)];

%% Start reconstruction

% Simulate noisy measurements
m       = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N); 

% Add noise/poisson noise
m  = m + noiselevel*max(abs(m(:)))*randn(size(m));
% m = imnoise(m,'poisson');

b = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang);
%%
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
g   = zeros(2*N*N,1);          % initial iterate is the backprojected data
rho = zeros(iter,1); % initialize parameters

%% Create L-matrix for Generalized Tikhonov!!!
% Tehd‰‰n L matriisi Kronecker tulon avulla...
L = spdiags([-ones(N-1,1), ones(N-1,1); 0, 1], 0:1, N, N);
L1 = kron(speye(N),L);
L2 = kron(L, speye(N));
L1T = L1';
L2T = L2';
Hg  = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang);
L      = L1T*L1 + L2T*L2;
%Built eye matrix
opMatrix = opEye(N^2);
% Built the NEW REGULARIZATION TERM + L
Q2 = [alpha*L,beta*opEye(N^2);beta*opEye(N^2),alpha*L];
% Add the term to Hg:
Hg     = Hg+Q2*g;
r    = b-Hg;
rho(1) = r(:).'*r(:);

% Start iteration
%figure(1);
for kkk = 1:iter
    if kkk==1
        p = r;
    else
        bee = rho(kkk)/rho(kkk-1);
        p    = r + bee*p;
    end
    w          = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,p,ang,N),ang);
    w          = w + Q2*p;
    aS         = rho(kkk)/(p.'*w);
    g          = g + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
% figure(2)
%   recn1 = reshape(g(1:(end/2),1:end),N,N);
%   imshow(recn1,[]);
  
  % Calculate the error
  CG1 = reshape(g(1:(end/2),1:end),N,N);
  CG2 = reshape(g((end/2)+1:end,1:end),N,N);
  err_CG1 = norm(M1(:)-CG1(:))/norm(M1(:));
  err_CG2 = norm(M2(:)-CG2(:))/norm(M2(:));
  err_total = (err_CG1+err_CG2)/2;
  % Calculate the similarity index
  SSIM1 = ssim(M1,CG1);
  SSIM2 = ssim(M2,CG2);
  SSIM_total = (SSIM1+SSIM2)/2;
  format short e
  % Monitor the run
  disp(['Iteration ', num2str(kkk,'%4d'),', total error value ',num2str(round(err_total,3),'%.3f'),', total SSIM value ',num2str(SSIM_total,'%.3f')])
  
  %Check if the error is as small enough
    if err_total < 1*10^-6
        disp('virhe alle kohinan!')
        break;
    end
end
CG1 = reshape(g(1:(end/2),1:end),N,N);
CG2 = reshape(g((end/2)+1:end,1:end),N,N);

% Determine computation time
comptime = toc;

% Take a look at the results
figure(4);
% Original phantom1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({'Phantom1, matrixfree'});
% Reconstruction of phantom1
subplot(2,2,2)
imagesc(CG1);
colormap gray;
axis square;
axis off;
title(['Approximate error ', num2str(round(err_CG1*100,1)), '%, \alpha=', num2str(alpha), ', \beta=', num2str(beta)]);
% Original target2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({'Phantom2, matrixfree'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(CG2);
colormap gray;
axis square;
axis off;
title(['Approximate error ' num2str(round(err_CG2*100,1)), '%, iter=' num2str(iter)]);

%%%%% Otherkind of plot %%%%%%
figure(59)
%imagesc([M1,CG1,M2,CG2]);
imshow([M1,CG1;M2,CG2]);
title(['Approximate error: ' num2str(round(err_CG2*100,1)),'%,    SSIM: ' num2str(round(SSIM1,3)) ',         Approximate error: ' num2str(round(err_CG2*100,1)),'%,    SSIM: ' num2str(round(SSIM2,3)) ', iter: ' num2str(iter)]);
colormap gray;
axis off;

%% For color figures
% figure(6);
% % Original phantom1
% subplot(2,2,1);
% imagesc(reshape(g1,N,N));
% colormap jet;
% axis square;
% axis off;
% title({material1,', Phantom1, BB, matrixfree'});
% % Reconstruction of phantom1
% subplot(2,2,2)
% recn1=reshape(g(1:(length(g)/2)),N,N);
% imagesc(recn1);
% colormap jet;
% axis square;
% axis off;
% title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1), ', \beta=', num2str(beta)]);
% % Original M2
% subplot(2,2,3)
% imagesc(reshape(g2,N,N));
% colormap jet;
% axis square;
% axis off;
% title({material2,', Phantom2, BB, matrix free'});
% % Reconstruction of phantom2
% subplot(2,2,4)
% recn2=reshape(g(length(g)/2+1:end),N,N);
% imagesc(recn2);
% colormap jet;
% axis square;
% axis off;
% title(['Relative error=' num2str(err_squ2), ', \alpha_2=' num2str(alpha2)]);

% Save the result to disc 
%save('from_CG_Tik_matrixfree', 'CG1', 'CG2', 'err_CG1', 'err_CG2');