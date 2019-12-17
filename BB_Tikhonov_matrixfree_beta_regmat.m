%%%% MULTI ENERGY MATERIAL DECOMPOSITION CODE BARZILAI BORWEIN WITH NEW REGULARIZATION TERM %%%%
% Here I replace the matrix multiplications by functions, which still use matrices
% Reconstruct two material phantom, imaged with two different energies, using Barzilai-Borwain
% conjugate gradient method. Second regularization term added!
clear all;
% Measure computation time: start clocking here
tic
%% Choices for the user
% Choose the size of the unknown. The image has size MxM.
M          = 40;
% Adjust regularization parameters
alpha1     = 100;             
alpha2     = 100;
beta       = 0.2;
% Adjust noise level and number of iterations
noiselevel = 0.0001;
iter       = 200;
% Choose the angles for tomographic projections
Nang       = 65; % odd number is preferred
ang        = [0:(Nang-1)]*360/Nang;
%% Define attenuation coefficients c_ij of the two materials
% 
% Al and PVC
% c_11 = 2.096; %PVC 30kV
% c_21 = 0.641; %PVC 50kV
% c_12 = 3.044; %Al 30kV
% c_22 = 0.994; %Al 50kV

% Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV
% Construct target
% Here we use simulated phantoms for both materials.
% Define overlapping materials: HY phantoms!
M2 = imresize(double(imread('HY_Al.bmp')), [M M]);
M1 = imresize(double(imread('HY_square_inv.jpg')), [M M]);
% Take away negative pixels
M1 = max(M1,0);
M2 = max(M2,0);
% Vektorize
g1 = M1(:);
g2 = M2(:);
% Combine
g  =[g1;g2];
%% Start reconstruction
% Simulate measurements
m  = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,M);
% Add noise
m  = m + noiselevel*max(abs(m(:)))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter
b = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang);

% Initialize g (the image vector containing both reconstuctions one after another)
%old version: g = zeros(2*M*M,1);
g = b; % initial iterate is the backprojected data
%% Start the iteration part
% First iteration (special one)
N = size(g,1);
% Initialize gradient
graddu = zeros(N,1);

% Count the second regterm
i1 = 1:N; i2 = [N/2+1:N,1:N/2];
for j = 1:N
    graddu(j) = g(i1(j))*g(i2(j))^2;
end
disp('Creating gradient function...');
graddu = graddu(:);
% Set different regularization parameters alpha1 and alpha2 for the images
RegMat = [alpha1*eye(M^2),zeros(M^2);zeros(M^2),alpha2*eye(M^2)];
gradF1 = 2*A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,M),ang)-2*b+RegMat*g+beta*graddu;
disp('Gradient function F1 ready!');
%% Decide the first step size
lambda = 0.0001;
oldu   = g;
g      = max(0,g-lambda*gradF1);
%% Iterate (this is the real iteration scheme)
disp('Iterating... ' );
%figure(1);
for iii = 1:iter
    % Gradient for g again:
    i1 = 1:N; i2 = [N/2+1:N,1:N/2];
    for j=1:N
        graddu(j) = g(i1(j))*g(i2(j))^2;
    end
    graddu = graddu(:);
    g = g(:);
    oldgradF1 = gradF1;
    % Count new gradient
    %old version:gradF1 = 2*(A'*A)*g-2*A'*m+alpha1*2*g;%+alpha2*graddu;
    gradF1 = 2*A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,M),ang)-2*b+RegMat*g+beta*graddu;
    % Update the step size
    lambda = (g-oldu)'*(g-oldu)/((g-oldu)'*(gradF1-oldgradF1));
    oldu = g;
    g = max(0,g-lambda*gradF1);
    % Show how the iterations proceed
    if mod(iii,10)==0
        disp([num2str(iii),'/' num2str(iter)]);
    end
%     reco1=reshape(g(1:(N/2)),M,M);
%     imshow(reco1,[]);
%     reco2=reshape(g(N/2+1:N),M,M);
%     imshow(reco2,[]);
end
disp('Iteration ready!');

% Here we need to separate the different images by naming them as follows:
reco1=reshape(g(1:(N/2)),M,M);
reco2=reshape(g(N/2+1:N),M,M);

% Here we calculated the error separately for each phantom
% Square Error in Tikhonov reconstruction 1
err_squ1 = norm(M1(:)-reco1(:))/norm(M1(:));
disp(['Square norm relative error for first reconstruction: ', num2str(err_squ1)]);
% Square Error in Tikhonov reconstruction 2
err_squ2 = norm(M2(:)-reco2(:))/norm(M2(:));
disp(['Square norm relative error for second reconstruction: ', num2str(err_squ2)]);

% Sup norm errors:
% Sup error in reco1
err_sup1 = max(max(abs(M1(:)-reco1(:))))/max(max(abs(reco1)));
disp(['Sup relative error for first reconstruction: ', num2str(err_sup1)]);
% Sup error in reco2
err_sup2 = max(max(abs(M2(:)-reco2(:))))/max(max(abs(reco2)));
disp(['Sup relative error for second reconstruction: ', num2str(err_sup2)]);
%% Take a look at the results
figure(3);
% Original M1
subplot(2,2,1);
imagesc(reshape(M1,M,M));
colormap gray;
axis square;
axis off;
title({'M1, BB, matrix free'});
% Reconstruction of M1
subplot(2,2,2)
reco1=reshape(g(1:(N/2)),M,M);
imagesc(reco1);
colormap gray;
axis square;
axis off;
title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1), ', \alpha_2=', num2str(alpha2)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(M2,M,M));
colormap gray;
axis square;
axis off;
title({'M2, BB, matrix free'});
% Reconstruction of M2
subplot(2,2,4)
reco2 = reshape(g(N/2+1:N),M,M);
imagesc(reco2);
colormap gray;
axis square;
axis off;
title(['Relative error=' num2str(err_squ2), ', \beta=' num2str(beta), ', iter=' num2str(iter)]);
toc