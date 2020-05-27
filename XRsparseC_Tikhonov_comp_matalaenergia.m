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
N       = 40;

% Choose relative noise level in simulated noisy data
%noiselevel = 0.0001;
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;

% Define attenuation coefficients: Iodine and PVC
material1='PVC';
material2='Iodine';
c11    = 42.2057; %Iodine 30kV
%c21    = 60.7376; %Iodine 50kV
c12    = 2.096346;%PVC 30kV
%c22    = 0.640995;%PVC 50kV

% Construct phantom. You can modify the resolution parameter N.
M1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [N N]);
M2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [N N]);
M1=M1(:,:,1);
M2=M2(:,:,1);
%figure(6);
%imshow(M1,[]);
%figure(7);
%imshow(M2,[]);
% Combine
%g       = [g1;g2];

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives
% target = M1;
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
% eval(['save RadonMatrix', num2str(N), ' A ang target N P Nang']);
% 
% % Load radonMatrix
eval(['load RadonMatrix', num2str(N), ' A ang target N P Nang']);

% This function calculates multiplication A*g for system with two images
% and two materials, without constructing the matrix A.

% Perform the needed matrix multiplications. Now a matrix multiplication
% has been switched to radon
ag1 = radon(M1,ang);
ag2 = radon(M2,ang);
% Calculate the parts needed for block matrix multiplication
res1 = c11*ag1;% c11*A*M1
res2 = c12*ag2;% c12*A*M2
%res3 = c21*ag1(:);% c21*A*M1
%res4 = c22*ag2(:);% c22*A*M2

% Combine results
%Energiaa 1 vastaava mittaus:
m1 = res1+res2;
%Energiaa 2 vastaava mittaus:
%m2 = res3 + res4;
% valitaan t�h�n energian 1 mittaus mittausdataksemme:
mnc = reshape(m1,[61,65]);
figure;
imshow(mnc,[]);
%%
%T�t� ei tarvita, koska t�ss� rekonstruoimme vain energiaa 1
%m2 = res3+res4;
% % Energy E1
% m11 = c11*A*M1;
% m21 = c21*A*M1;
% %Energy E2
% m12 = c12*A*M2;
% m22 = c22*A*M2;
% %%
% % In the simulated measurement, we combine the measurements as follows
% m1 = m11 + m12;
% m2 = m21 + m22;
%%
% We combine the matrix A
% Emme luo my�sk��n j�ttil�ismatriisia, koska t�m� on matriisivapaa
% menetelm�!? Mutta tarvitsemmeko t�llaisen yhdistetyn matriisin?
%A = [c11*A c12*A];
% T�t� funktioita emme voi k�ytt��, koska se on suunniteltu kahden kuvan
% tapauskselle
%mnc = A2x2mult(a,c11,c12,c21,c22,g);on jo laskettu
% Se voidaan korvata Radonilla ja saadaan sinogrammi
%Meid�n t�ytyy laskea [c11*A c12*A]*g
%noiselevel=0.001;
% Add noise
%mncn = mnc + noiselevel*max(abs(mnc(:)))*randn(size(mnc));

% Maximum number of iterations. You can modify this number and observe the
% consequences.
K = 200;         

% Regularization parameter
alpha = 0.00001;

% Load noisy measurements from disc. The measurements have been simulated
% (avoiding inverse crime) in routine XRsparse3_NoCrimeData_comp.m
%load XRsparse_NoCrime N mnc mncn measang target

% Construct right hand side
corxn = 7.65; % Incomprehensible correction factor
% Perform the needed matrix multiplications. Now a.' multiplication has been
% switched to iradon
am1 = iradon(mnc,ang,'none');
am1 = am1(2:end-1,2:end-1);
am1 = corxn*am1;
b=am1;
%am2 = iradon(m2,ang,'none');
%am2 = am2(2:end-1,2:end-1);
%am2 = corxn*am2;

% Compute the parts of the result individually
% res1 = c11*am1(:);
% res2 = c21*am2(:);
% res3 = c12*am1(:);
% res4 = c22*am2(:);

% Collect the results together
%res = [res1 + res2; res3 + res4];

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
recn    = b;          % initial iterate is the backprojected data
rho     = zeros(K,1); % initialize parameters

% Compute residual using matrix-free implementation.
Hf     = radon(recn,ang);
Hf     = iradon(Hf,ang,'none');
Hf     = Hf(2:end-1,2:end-1);
Hf     = corxn*Hf;
Hf     = Hf + alpha*recn;
r      = b-Hf;
rho(1) = r(:).'*r(:);

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
    recn       = recn + a*p;
    r          = r - a*w;
    rho(kkk+1) = r(:).'*r(:);
    if mod(kkk,10)==0
        disp([kkk K])
    end
end
%%
% Save result to file
%save XRsparseTikhonov recn alpha target

% Show images of the results
% Compute relative error
err_squ = norm(target(:)-recn(:))/norm(target(:));

% Plot reconstruction image
figure(1)
clf
imagesc(recn);
colormap gray
axis equal
axis off
title(['Tikhonov: error ', num2str(round(err_squ*100)), '%'])
%XRsparseC_Tikhonov_plot