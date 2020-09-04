%%%% MULTI ENERGY MATERIAL DECOMPOSITION CODE BARZILAI BORWEIN WITH NEW REGULARIZATION TERM %%%%
% Here I replace the matrix multiplications by matrix free functions
% Reconstruct two material phantom, imaged with two different energies, using Barzilai-Borwain
% optimization method. Second regularization term added!
clear all;
% Measure computation time: start clocking here
tic
%% Choices for the user
% Choose the size of the unknown. The image has size MxM.
M          = 40;
% Adjust regularization parameters
alpha1     = 10;             
alpha2     = 10;
beta       = 85;
% Adjust noise level and number of iterations
noiselevel = 0.01;
iter       = 162;
% Choose the angles for tomographic projections
Nang       = 65; % odd number is preferred
ang        = [0:(Nang-1)]*360/Nang;
%% Define attenuation coefficients c_ij of the two materials
% % Iodine and PVC
% c11    = 42.2057; %Iodine 30kV
% c21    = 60.7376; %Iodine 50kV
% c12    = 2.096346;%PVC 30kV
% c22    = 0.640995;%PVC 50kV

% %Korjatut kertoimet
%  material1='Iodine';
%  material2='PVC';
% c11    = 1.7237;%PVC 30kV
% c12    = 37.57646; %Iodine 30kV
% c21    = 0.3686532;%PVC 50kV 
% c22    = 32.404; %Iodine 50kV

c11    = 1.491; %PVC 30 30kV
c12    = 8.561; %Iodine 30kV
c21    = 0.456; %PVC 50kV
c22    = 12.32;   % Iodine 50kV
% Iodine and Al
% c11 = 42.2057; %Iodine 30kV
% c21 = 60.7376; %Iodine 50kV
% c12 = 3.044;   %Al 30kV
% c22 = 0.994;   %Al 50kV

% Construct target
% Here we use simulated phantoms for both materials.
% Define overlapping materials: HY phantoms!
%M1 = imresize(double(imread('HY_Al.bmp')), [M M]);
%M2 = imresize(double(imread('HY_square_inv.jpg')), [M M]);
M1 = imresize(double(imread('material1.png')), [M M]);
M2 = imresize(double(imread('material2.png')), [M M]);
M1=M1(:,:,1);
M2=M2(:,:,1);
% % Try to normalize the image between 0 and 255
% min1=min(min(M1));
% max1=max(max(M1));
% M1 = double(255 .* ((double(M1)-double(min1))) ./ double(max1-min1));
% 
% min1=min(min(M2));
% max1=max(max(M2));
% M2 = double(255 .* ((double(M2)-double(min1))) ./ double(max1-min1));

% Vektorize
g1 = M1(:);
g2 = M2(:);
% Combine
g  =[g1;g2];
%% Start reconstruction
% Simulate measurements SINOGRAM
m  = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,M);
m_matrixfree=m;
%g sisältää nyt molemmat kuvat dropattuna vektoreiksi. Mutta reaali
%mittauksessa meillä ei ole g:tä. Meillä on vain m eli sinogrammi.

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
%b_matrixfree=b;

% % Last revision Salla Latva-Äijö Sep 2019
% m1 = m(1:(end/2));
% m1 = reshape(m1, [length(m)/(2*length(ang)) length(ang)]);
% m2 = m((end/2+1):end);
% m2 = reshape(m2, [length(m)/(2*length(ang)) length(ang)]);

corxn = 7.65; % Incomprehensible correction factor

% % Perform the needed matrix multiplications. Now a.' multiplication has been
% % switched to iradon
% am1 = iradon(m1,ang,'none');
% am1 = am1(2:end-1,2:end-1);
% am1 = corxn*am1;
% 
% am2 = iradon(m2,ang,'none');
% am2 = am2(2:end-1,2:end-1);
% am2 = corxn*am2;

% % % Compute the parts of the result individually
% % res1 = c11*am1(:);
% % res2 = c21*am2(:);
% % res3 = c12*am1(:);
% % res4 = c22*am2(:);

% % % Collect the results together
% % res = [res1 + res2; res3 + res4];
% % b=res;
%%
% Initialize g (the image vector containing both reconstuctions one after another)
%old version: 
g = zeros(2*M*M,1);
%g = b; % initial iterate is the backprojected data
%% Start the iteration part
% First iteration (special one)
N = size(g,1);
% Initialize gradient
graddu = zeros(N,1);

% Count the second regterm
i1 = 1:N; i2 = [N/2+1:N,1:N/2];
for j = 1:N
    graddu(j) = g(i2(j));
    %graddu(j) = g(i1(j))*g(i2(j))^2;vanha!
end
disp('Creating gradient function...');
graddu = graddu(:);
% Set different regularization parameters alpha1 and alpha2 for the images
%Tehdään eye matriisi
opMatrix = opEye(M^2);
zeroMatrix = opZeros(M^2);
RegMat = [alpha1*opMatrix,zeroMatrix;zeroMatrix,alpha2*opMatrix];
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
        %graddu(j) = g(i1(j))*g(i2(j))^2;
        graddu(j) = g(i2(j));
    end
    graddu = graddu(:);
    g = g(:);
    oldgradF1 = gradF1;
    % Count new gradient
    %old version:gradF1 = 2*(A'*A)*g-2*A'*m+alpha1*2*g;%+alpha2*graddu;
    gradF1 = 2*A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,M),ang)-2*b+RegMat*sqrt(g)+beta*graddu;
    % Update the step size
    lambda = (g-oldu)'*(g-oldu)/((g-oldu)'*(gradF1-oldgradF1));
    oldu = g;
    g = max(0,g-lambda*gradF1);
    % Show how the iterations proceed
    %if mod(iii,10)==0
    %    disp([num2str(iii),'/' num2str(iter)]);
    %end
%     reco1=reshape(g(1:(N/2)),M,M);
%     imshow(reco1,[]);
    %BB2=reshape(g(N/2+1:N),M,M);
    %imshow(BB2,[]);
      %Check if the error is as small enough
  CG1 = reshape(g(1:(end/2),1:end),M,M);
  CG2 = reshape(g((end/2)+1:end,1:end),M,M);
  err_CG1 = norm(M1(:)-CG1(:))/norm(M1(:));
  err_CG2 = norm(M2(:)-CG2(:))/norm(M2(:));
  err_total = (err_CG1+err_CG2)/2;
      format short e
    % Monitor the run
    disp(['Iteration ', num2str(iii,'%4d'),', total error value ',num2str(err_total)])
end
disp('Iteration ready!');

% Here we need to separate the different images by naming them as follows:
BB1=reshape(g(1:(N/2)),M,M);
BB2=reshape(g(N/2+1:N),M,M);

% Here we calculate the error separately for each phantom
err_BB1 = norm(M1(:)-BB1(:))/norm(M1(:));
disp(['Square norm relative error for first reconstruction: ', num2str(err_BB1)]);
% Square Error in Tikhonov reconstruction 2
err_BB2 = norm(M2(:)-BB2(:))/norm(M2(:));
disp(['Square norm relative error for second reconstruction: ', num2str(err_BB2)]);

% Sup norm errors:
err_sup1 = max(max(abs(M1(:)-BB1(:))))/max(max(abs(BB1)));
disp(['Sup relative error for first reconstruction: ', num2str(err_sup1)]);
% Sup error in reco2
err_sup2 = max(max(abs(M2(:)-BB2(:))))/max(max(abs(BB2)));
disp(['Sup relative error for second reconstruction: ', num2str(err_sup2)]);
%% Take a look at the results
figure(2);
% Original M1
subplot(2,2,1);
imagesc(reshape(M1,M,M));
colormap gray;
axis square;
axis off;
title({'M1, BB, matrix free'});
% Reconstruction of M1
subplot(2,2,2)
BB1=reshape(g(1:(N/2)),M,M);
imagesc(BB1);
colormap gray;
axis square;
axis off;
title(['Approximate error ', num2str(round(err_BB1*100,1)), '%, \alpha_1=', num2str(alpha1), ', \alpha_2=', num2str(alpha2)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(M2,M,M));
colormap gray;
axis square;
axis off;
title({'M2, BB, matrix free'});
% Reconstruction of M2
subplot(2,2,4)
BB2 = reshape(g(N/2+1:N),M,M);
imagesc(BB2);
colormap gray;
axis square;
axis off;
title(['Approximate error ', num2str(round(err_BB2*100,1)), '%, \beta=' num2str(beta), ', iter=' num2str(iter)]);
toc

% Save the result to disc 
save('from_BB_Tik_matrixfree', 'BB1', 'BB2', 'err_BB1', 'err_BB2');