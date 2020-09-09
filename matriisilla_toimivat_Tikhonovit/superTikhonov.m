% Here we simulate X-ray tomographic measurement without avoiding inverse
% crime

clear all;

%% Define the measurement setup
% Choose the size of the unknown. The image has size MxM.
M = 40;

% Choose the angles 
Nang = 65;

% Load precomputed results of the routine tomo01_RadonMatrix_comp.m
% with the particular values of j and Nang above
loadcommand = ['load RadonMatrix_', num2str(M), '_', num2str(Nang), ' A M angles'];
eval(loadcommand)

%% Construct target
f1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [M M]);
f2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [M M]);
f1=f1(:,:,1);
f2=f2(:,:,1);
f1 = f1(:);
f2 = f2(:);

%% Define constants
% Original imaginary ones
% c_11 = 20;
% c_21 = 10;
% c_12 = 3;
% c_22 = .2;

% c_11 = 2.096; %PVC 30kV
% c_21 = 0.641; %PVC 50kV
% c_12 = 3.044; %Al 30kV
% c_22 = 0.994;%Al 50kV

c_11 = 42.2057; %Iodine 30kV
c_21 = 60.7376; %Iodine 50kV
c_12 = 3.044; %Al 30kV
c_22 = 0.994;%Al 50kV
%% Construct the different measurements
% Energy E1
m_11 = c_11*A*f1;
m_21 = c_21*A*f1;
%Energy E2
m_12 = c_12*A*f2;
m_22 = c_22*A*f2;
%%
% In the simulated measurement, we combine the measurements as follows
m_1 = m_11 + m_12;
m_2 = m_21 + m_22;

m = [m_1; m_2];
%%
% We combine the matrix A
A = [c_11*A c_12*A; c_21*A c_22*A];

%% We compute a Tikhonov solution
alpha = 100; % regularization parameter
fun = @(reco) A.'*(A*reco)+alpha*reco;
b = A.'*m(:);
reco= pcg(fun,b);

%% Compute FBP reconstructions for finding out the error
% FBP reconstruction of phantom 1
f1 = reshape(f1,[M,M]);
theta = 1:2.7692:179;
%theta = 1:1:180;
R1 = radon(f1,theta);
recoFBP1 = iradon(R1,theta);


% FBP reconstruction of phantom 2
f2 = reshape(f2,[M,M]);
R2 = radon(f2,theta);
recoFBP2 = iradon(R2,theta);
%% Find out the error of the results

% % Error in Tikhonov reconstruction 1
% recoFBP1 = recoFBP1(2:end-1,2:end-1);
% err_squ1 = norm(recoFBP1(:)-reco1(:))/norm(recoFBP1(:));
% disp(['Square norm relative error for first reconstruction: ', num2str(err_squ1)]);
% 
% % Error in Tikhonov reconstruction 2
% recoFBP2 = recoFBP2(2:end-1,2:end-1);
% err_squ2 = norm(recoFBP2(:)-reco2(:))/norm(recoFBP2(:));
% disp(['Square norm relative error for second reconstruction: ', num2str(err_squ2)]);
% 
reco1 = reco(size(reco,1)/2);
reco2 = reco(size(reco,1)/2+1:size(reco,1));

% Square Error in Tikhonov reconstruction 1
err_squ1 = norm(f1(:)-reco1(:))/norm(f1(:));
disp(['Square norm relative error for first reconstruction: ', num2str(err_squ1)]);

% Square Error in Tikhonov reconstruction 2
err_squ2 = norm(f2(:)-reco2(:))/norm(f2(:));
disp(['Square norm relative error for second reconstruction: ', num2str(err_squ2)]);

% Sup error in reco1
err_sup1 = max(max(abs(f1(:)-reco1(:))))/max(max(abs(reco1)));
disp(['Sup relative error for first reconstruction: ', num2str(err_sup1)]);

% Sup error in reco2
err_sup2 = max(max(abs(f2(:)-reco2(:))))/max(max(abs(reco2)));
disp(['Sup relative error for second reconstruction: ', num2str(err_sup2)]);
%% Take a look to the reconstructions
% Look the 1. and second reconstruction
M=40;
figure(10);

subplot(2,2,1);
imagesc(reshape(f1,[M,M]));
colormap;
axis square;
axis off;
title({'M1, original'})

subplot(2,2,3);
reco1 = reco(1:size(reco,1)/2);
imagesc(reshape(reco1,M,M));
colormap;
axis square;
axis off;
title({'Tikhonov reconstruction,'; '65 projections'});

subplot(2,2,2)
imagesc(reshape(f2,[M,M]));
colormap;
axis square;
axis off;
title({'M2, original'});

subplot(2,2,4);
reco2 = reco(size(reco,1)/2+1:size(reco,1));
imagesc(reshape(reco2,M,M));
colormap;
axis square;
axis off;
title({'Tikhonov reconstruction,'; '65 projections'});