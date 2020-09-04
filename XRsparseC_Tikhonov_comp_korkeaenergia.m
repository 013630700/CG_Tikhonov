% Example computations related to X-ray tomography.
% Here we apply generalized Tikhonov regularization and solve
% the normal equations using the conjugate gradient method.
%
% We solve the minimization problem
%         min (f^T H f - 2 b^T f),
% where
%         H = A^T A + alpha*I
% and
%         b = A^T m,
% with m the measurement and alpha>0 the regularization parameter.
%
%
% Jennifer Mueller and Samuli Siltanen, October 2012
% Modified by Salla 2020
clear all;
N       = 128;

% Choose relative noise level in simulated noisy data
%noiselevel = 0.0001;
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;

% Define attenuation coefficients: Iodine and PVC
% material1='PVC';
% material2='Iodine';
% c11    = 42.2057; %Iodine 30kV
c22    = 60.7376; %Iodine 50kV
% c12    = 2.096346;%PVC 30kV
c21    = 0.640995;%PVC 50kV

%c11 = 1.7237;%PVC 30kV
%c12 = 37.57646; %Iodine 30kV
% c21 = 0.3686532;%PVC 50kV
% c22 = 32.404; %Iodine 50kV

%material1='Iodine';
%material2='bone';
%c12    = 37.57646; %Iodine 30kV
%c22    = 32.404; %Iodine 50kV
%c11    = 2.0544;%Bone 30kV
%c21    = 0.448512;%Bone 50kV

% Construct phantom. You can modify the resolution parameter N.
M1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [N N]);
M2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [N N]);
%M1 = imresize(double(imread('selkaranka_phantom.jpg')), [N N]);
%M2 = imresize(double(imread('selkaranka_phantom_nurin.jpg')), [N N]);
M1=M1(:,:,1);
M2=M2(:,:,1);
figure(6);
target=M2;
%target=max(target,0);
imshow(target,[]);
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
%eval(['load RadonMatrixE2', num2str(N), ' A ang target N P Nang']);

% This function calculates multiplication A*g for system with two images
% and two materials, without constructing the matrix A.

% Perform the needed matrix multiplications. Now a matrix multiplication
% has been switched to radon
%mL = c11*radon(M1,ang)+c12*radon(M2,ang);
mH = c21*radon(M1,ang)+c22*radon(M2,ang);
% valitaan tähän energian 2 mittaus mittausdataksemme:
mH = reshape(mH,[length(radon(target,0)),Nang]);

%noiselevel=0.001;
% Add noise
%mncn = mnc + noiselevel*max(abs(mnc(:)))*randn(size(mnc));

% Maximum number of iterations. You can modify this number and observe the
% consequences.
K = 20;         

% Regularization parameter
alpha = 10;

% Construct right hand side
corxn = 7.65; % Incomprehensible correction factor
% Perform the needed matrix multiplications. Now A.' multiplication has been
% switched to iradon
b = iradon(mH,ang,'none');
b = b(2:end-1,2:end-1);
b = corxn*b;
figure(7);
imshow(b,[])
% Compute the parts of the result individually
% res1 = c11*am1(:);
% res2 = c21*am2(:);
% res3 = c12*am1(:);
% res4 = c22*am2(:);

% Collect the results together
%res = [res1 + res2; res3 + res4];

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
recnH    = zeros(size(b));          % initial iterate is the backprojected data
rho     = zeros(K,1); % initialize parameters

% Compute residual using matrix-free implementation.
Hf     = radon(recnH,ang);
Hf     = iradon(Hf,ang,'none');
Hf     = Hf(2:end-1,2:end-1);
Hf     = corxn*Hf;
Hf     = Hf + alpha*recnH;
r      = b-Hf;
rho(1) = r(:).'*r(:);

%figure(8);
% Start iteration
for kkk = 1:(K-1)
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    w          = radon(p,ang);
    w          = iradon(w,ang,'none');
    w          = w(2:end-1,2:end-1);
    w          = corxn*w;
    w          = w + alpha*p;
    a          = rho(kkk)/(p(:).'*w(:));
    recnH       = recnH + a*p;
    r          = r - a*w;
    rho(kkk+1) = r(:).'*r(:);
    if mod(kkk,10)==0
        disp([kkk K])
    end
    %imshow(recnH,[]);
    %pause(0.2);
end
recnH=max(recnH,0);
%%
% Save result to file
%save XRsparseTikhonov recn alpha target

% Show images of the results
%Compute relative error
target = target./max(max(target));
recnH = recnH./max(max(recnH));
err_squ = norm(target(:)-recnH(:))/norm(target(:));

% Plot reconstruction image
figure(9);
clf
imagesc(recnH);
colormap gray
axis equal
axis off
title(['Tikhonov: error ', num2str(round(err_squ*100)), '%'])

% Save to disk
originalImage = recnH;
outputBaseFileName = 'HighEnergyTikRecon.PNG';
imwrite(originalImage, outputBaseFileName);
% Recall from disk:
recalledImage = imread(outputBaseFileName);
figure(8)
imshow(recalledImage);
fontSize=15;
title('Recalled Image', 'FontSize', fontSize);

% XRsparseC_Tikhonov_plot
imagesc(recnH);
colormap gray;
axis square;
axis off;
title({'High energy CG, Iodine in PVC, matrixfree, error ' num2str(round(err_squ*100)),'%'});