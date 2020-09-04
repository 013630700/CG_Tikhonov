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
N       = 512;

% Choose relative noise level in simulated noisy data
%noiselevel = 0.0001;
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;
noiselevel = 0.01;

% Define attenuation coefficients: Iodine and PVC
%Fisrst index is energy and secons index refers to material
% material1='PVC';
% material2='Iodine';
% Low energy measurements:
c11    = 2.096346;%PVC 30kV
c12    = 42.2057; %Iodine 30kV
% High energy measurements
% c21    = 60.7376; %Iodine 50kV
% c22    = 0.640995;%PVC 50kV

% Construct phantom. You can modify the resolution parameter N.
M1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [N N]);
M2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [N N]);
%M1 = imresize(double(imread('selkaranka_phantom.jpg')), [N N]);
%M2 = imresize(double(imread('selkaranka_phantom_nurin.jpg')), [N N]);
M1=M1(:,:,1);
M2=M2(:,:,1);
%figure(6);
%imshow(M1,[]);
%figure(7);
%imshow(M2,[]);
target=M1; %%%HEI MIKSI TÄSSÄ ON TARGET M1, aiheuttaako ongelmaa?
%target=max(target,0);
%figure(1);
%imshow(target,[]);
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
% eval(['load RadonMatrix', num2str(N), ' A ang target N P Nang']);

% This function calculates multiplication A*g for system with two images
% and two materials, without constructing the matrix A.

% Perform the needed matrix multiplications. Now a matrix multiplication
% has been switched to radon
m11 = c11*radon(M1,ang);
m12 = c12*radon(M2,ang);
mL = c11*radon(M1,ang)+c12*radon(M2,ang);

% Sinogram:
mnc = reshape(mL,[length(radon(target,0)),Nang]);
%figure(2);
%imshow(mnc,[]);
%%
% Add noise
%mncn = mnc + noiselevel*max(abs(mnc(:)))*randn(size(mnc));
mnc = mnc + noiselevel*max(abs(mnc(:)))*randn(size(mnc));
% Maximum number of iterations. You can modify this number and observe the
% consequences.
K = 20;         

% Regularization parameter
alpha = 10;

% Construct right hand side
corxn = 7.65; % Correction factor
% Perform the needed matrix multiplications. Now A.' multiplication has been
% switched to iradon
b = iradon(mnc,ang,'none');
b = b(2:end-1,2:end-1);
b = corxn*b;
%figure(3);
%imshow(b,[]);
%%
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
recnL    = zeros(size(b));          % initial iterate is the backprojected data
rho     = zeros(K,1); % initialize parameters

% Compute residual using matrix-free implementation.
Hf     = radon(recnL,ang);
Hf     = iradon(Hf,ang,'none');
Hf     = Hf(2:end-1,2:end-1);
Hf     = corxn*Hf;
Hf     = Hf + alpha*recnL;
r      = b-Hf;
rho(1) = r(:).'*r(:);
%figure(4);
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
    recnL       = recnL + a*p;
    r          = r - a*w;
    rho(kkk+1) = r(:).'*r(:);
    if mod(kkk,10)==0
        disp([kkk K])
    end
    %imshow(recnL,[]);
    %pause(0.2);
end
recnL=max(recnL,0);
%%
% Save result to file
%save XRsparseTikhonov recn alpha target

% Show images of the results
% Compute relative error
target = target./max(max(target));
recnL = recnL./max(max(recnL));
err_squ = norm(target(:)-recnL(:))/norm(target(:));

% Plot reconstruction image
figure(5)
clf
imagesc(recnL);
colormap gray
axis equal
axis off
title(['Tikhonov recnL: error ', num2str(round(err_squ*100)), '%'])
% Save to disk
originalImage = recnL;
outputBaseFileName = 'lowEnergyTikRecon.PNG';
imwrite(originalImage, outputBaseFileName);
% Recall from disk:
recalledImage = imread(outputBaseFileName);
figure(8)
imshow(recalledImage);
fontSize=15;
title('Recalled Image', 'FontSize', fontSize);
%XRsparseC_Tikhonov_plot
figure(6)
imagesc(recnL);
%colorbar;
colormap jet;
axis square;
axis off;
F=getframe;
imwrite(F.cdata,'Image.png','png');
%title({'Low energy CG,Iodine in PVC, matrixfree, error ' num2str(round(err_squ*100)) '%'});
% binaryImage = grayImage == 255;
% borderMask = binaryImage-imclearborder(binaryImage);
% grayImage(borderMask) = 0;

% Segmentation 1
% % level = graythresh(F.cdata);
% % BW = imbinarize(F.cdata,level);
% % imshowpair(F.cdata,BW,'montage');

% Segmentation 2
% [L,N] = superpixels(F.cdata,33);
% figure(99)
% BW = boundarymask(L);
% imshow(imoverlay(F.cdata,BW,'cyan'),'InitialMagnification',67)

% Segmentation 3
% Segment Image into Three Levels Using Two Thresholds
% I = imread('Image.png');
% imshow(I)
% I=I(:,:,1);
% imshow(I)
% J = grayconnected(I,130,110);
% imshow(J)
% figure(88)
% imshow(J)
% axis off
% title('Original Image')
% thresh = multithresh(I,2);
% seg_I = imquantize(I,thresh);
% RGB = label2rgb(seg_I); 	 
% figure;
% imshow(RGB)
% axis off
% title('RGB Segmented Image')
%% Segmentation with image segmenter
% [BW,maskedImage] = segmentImage(F.cdata);
% figure(9);
% imshow(BW);
% figure(10);
% segIm=imresize(double(BW), [N N]);
% segIm=segIm(:,:,1);
% imagesc(segIm);
% axis square;
% axis off;


% Compute relative error
%err_squ_seg = norm(target(:)-segIm(:))/norm(target(:));
%title(['Segmentation result: error ', num2str(round(err_squ_seg*100)), '%'])