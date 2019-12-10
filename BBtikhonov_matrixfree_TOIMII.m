%%%% MATRIX FREE MULTI ENERGY MATERIAL DECOMPOSITION CODE BARZILAI BORWEIN WITH NEW REGULARIZATION TERM %%%%
% Reconstruct two material phantom, imaged with two different energies, using Barzilai-Borwain
% conjugate gradient method. Second regularization term added!
clear all;
%% Choices for the user
N       = 40;% Choose the size of the unknown. The image has size NxN.
% Adjust regularization parameters
alpha1  = 10; %10             
alpha2  = 10;%100
beta    = 0.5; % better results with small values!

% Choose relative noise level in simulated noisy data
noiselevel = 0.00001;

% Measure computation time later; start clocking here
tic
%% Define attenuation coefficients c_ij of the two materials
% Blood and bone
% c11 = 0.4083; %blood 30kV
% c21 = 0.2947; %blood 50kV
% c12 = 1.331;  %bone 30kV
% c22 = 0.4242; %bone 50kV

% Al and PVC
% c11 = 2.096; %PVC 30kV
% c21 = 0.641; %PVC 50kV
% c12 = 3.044; %Al 30kV
% c22 = 0.994; %Al 50kV

% Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

%% Construct target ****************************ADD c*************************************
% Here we use simulated phantoms for both materials.
g1 = imresize(double(imread('HY_Al.bmp')), [N N]);
g2 = imresize(double(imread('HY_square_inv.jpg')), [N N]);

% % Normalize original data
% g1=normalize(g1);
% g2=normalize(g2);

% Combine
g=[g1(:);g2(:)];

% Choose measurement angles (given in degrees, not radians). 
Nang    = 40; 
angle0  = -90;
ang     = angle0 + [0:(Nang-1)]/Nang*180;

% Simulate noisy measurements; here including inverse crime
m = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N);

% Add noise
m       = m + noiselevel*max(abs(m(:)))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter

b = A2x2Tmult_matrixfree(c11,c12,c21,c22,m,ang);

% Initialize g (the image vector containing both reconstuctions one after
% another)
K=250;% Choose the number off iterations

g = b; % initial iterate is the backprojected data

%alpha matriisin tilalle: Reg_mat
RegMat = [alpha1*eye(N^2),zeros(N^2);zeros(N^2),alpha2*eye(N^2)];
%% Start the iteration part
% Initializations for first iteration (special one)
M = size(g,1);
% Initialize second regularization term (gradient?)
regul=zeros(M,1);
% Count the gradient
i1 = 1:M; i2 = [M/2+1:M,1:M/2];
for j=1:M
    regul(j) = g(i1(j))*g(i2(j))^2;
end
% For historical reasons, we have here the handmade gradient for 2*2 system:
% graddu = [u(1)*u(5)^2; u(2)*u(6)^2; u(3)*u(7)^2; u(4)*u(8)^2; u(5)*u(1)^2;u(6)*u(2)^2;u(7)*u(3)^2; u(8)*u(4)^2];
disp('Creating second regularization function...');
regul = regul(:);

% Gradient has now also beta term:
% old_version: gradF1 = 2*(A'*A)*g-2*A'*m+alpha1*2*g;%+beta*graddu;
gradF1 = A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang)-b+RegMat*g+beta*regul;
disp('Gradient function F1 ready!');

%% Decide the first step size
lambda = 0.0001;
oldu = g;
g = max(0,g-lambda*gradF1);
%% Iterate (this is the real iteration scheme)
disp('Iterating... ' );
for iii = 1:K
    % Gradient for g again:
    i1 = 1:M; i2 = [M/2+1:M,1:M/2];
    for j=1:M
        regul(j) = g(i1(j))*g(i2(j))^2;
    end
    regul = regul(:);
    oldgradF1 = gradF1;
    % Calculate updated gradient
    % old version: gradF1 = 2*(A'*A)*g-2*A'*m+alpha1*2*g;%+alpha2*graddu;
    gradF1 = 2*A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang)-b+RegMat*g+beta*regul;
    % for historical reasons, we have this line here remembering how this
    % was before the second regularization term
    %gradF1 = 2*(A'*A)*u-2*A'*m+alpha*graddu;
    
    % Update the step size  
    lambda = (g-oldu)'*(g-oldu)/((g-oldu)'*(gradF1-oldgradF1));
    oldu = g;
    g = max(0,g-lambda*gradF1);
    % Show how the iterations proceed
    if mod(iii,10)==0
        disp([num2str(iii),'/' num2str(K)]);
    end
end
disp('Iteration ready!');

recn1=reshape(g(1:(length(g)/2)),N,N);
recn2=reshape(g(length(g)/2+1:end),N,N);
% % Normalize image
% recn1 = normalize(recn1);
% recn2 = normalize(recn2);
% Determine computation time
comptime = toc;
%% Compute the error: Check the accuracy of the data. The error should not be 
% Square error of reconstruction 1
err_squ1 = norm(g1(:)-recn1(:))/norm(g1(:));
% Square error of reconstruction 2
err_squ2 = norm(g2(:)-recn2(:))/norm(g2(:));

% Sup norm errors:
% Sup error in reco1
% err_sup1 = max(max(abs(g1(:)-recn1(:))))/max(max(abs(recn1)));
% Sup error in reco2
% err_sup2 = max(max(abs(g2(:)-recn2(:))))/max(max(abs(recn2)));
%% Take a look at the results
figure(3);
% Original phantom1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({'Phantom1, BB, matrixfree'});
% Reconstruction of phantom1
subplot(2,2,2)
recn1=reshape(g(1:(length(g)/2)),N,N);
imagesc(recn1);
colormap gray;
axis square;
axis off;
title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1), ', \beta=', num2str(beta)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({'Phantom2, BB, matrix free'});
% Reconstruction of phantom2
subplot(2,2,4)
recn2=reshape(g(length(g)/2+1:end),N,N);
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['Relative error=' num2str(err_squ2), ', \alpha_2=' num2str(alpha2)]);
toc