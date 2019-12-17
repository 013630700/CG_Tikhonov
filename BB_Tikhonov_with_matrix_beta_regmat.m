%%%% MULTI ENERGY MATERIAL DECOMPOSITION CODE BARZILAI BORWEIN WITH NEW REGULARIZATION TERM %%%%
%
% Reconstruct two material phantom, imaged with two different energies, using Barzilai-Borwain
% conjugate gradient method. Second regularization term added!
clear all;
% Measure computation time later; start clocking here
tic
%% Choices for the user
% Choose the size of the unknown. The image has size MxM.
M       = 40;
% Adjust regularization parameters
alpha1  = 100; %10
alpha2  = 100;
beta    = 0.2;
% Choose relative noise level in simulated noisy data
noiselevel = 0.0001;
iter    = 200;
% Choose the angles for tomographic projections
Nang    = 65; % odd number is preferred
ang        = [0:(Nang-1)]*360/Nang;
%% Define attenuation coefficients c_ij of the two materials
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
M2 = imresize(double(imread('HY_Al.bmp')), [M M]);
M1 = imresize(double(imread('HY_square_inv.jpg')), [M M]);
% Take away negative pixels
M1 = max(M1,0);
M2 = max(M2,0);
% Vektorize
g1 = M1(:);
g2 = M2(:);
% Combine
g=[g1;g2];
% Definitions and initializations
% Some definitions
n = M*M;
angles = [0:(Nang-1)]*360/Nang;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Let's find out the eventual size of matrix A. Namely, the size is
% % k x n, where n is as above and k is the total number of X-rays. 
% % The number of X-rays depends in a complicated way on Matlab's inner 
% % workings in the routine radon.m.
% tmp = radon(zeros(M,M),angles);
% k   = length(tmp(:));
% 
% % Initialize matrix A as sparse matrix
% A = sparse(k,n);
% %% Construct matrix A column by column. 
% % This is a computationally wasteful method, but we'll just do it here. 
% % For large values of M or Nang this will take a very long time to compute. 
% for iii = 1:n
%     % Construct iii'th unit vector
%     unitvec = zeros(M,M);
%     unitvec(iii) = 1;
%     
%     % Apply radon.m to a digital phantom having value "1" in exactly one 
%     % pixel and vaue "0" in all other pixels
%     tmp = radon(unitvec,angles);
%     
%     % Insert a new column to the tomography matrix
%     A(:,iii) = sparse(tmp(:));
%     
%     % Monitor the run
%     if mod(iii,round(n/10))==0
%         disp([iii n])
%     end
%     
% end
% % Save the result to disc 
% % The filename contains the parameter values for M and Nang
% savecommand = ['save -v7.3 RadonMatrix_', num2str(M), '_', num2str(Nang), ' A M n Nang angles'];
% eval(savecommand)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load precomputed results of the routine (commented above)
% with the particular values of j and Nang above
loadcommand = ['load RadonMatrix_', num2str(M), '_', num2str(Nang), ' M angles A'];
eval(loadcommand)
a = A;
%% Start reconstruction
% Simulate noisy measurements; here including inverse crime
m = A2x2mult(a,c11,c12,c21,c22,g);
% Add noise
m  = m + noiselevel*max(abs(m(:)))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter
b = A2x2Tmult(a,c11,c12,c21,c22,m);

% Initialize g (the image vector containing both reconstuctions one after another)
g = b; % initial iterate is the backprojected data
%% Start the iteration part
% First iteration (special one)
N = size(g,1);

% Initialize gradient
graddu=zeros(N,1);

% Count the gradient
i1 = 1:N; i2 = [N/2+1:N,1:N/2];
for j=1:N
    graddu(j) = g(i1(j))*g(i2(j))^2;
end
% For historical reasons, we have here the handmade gradient for 2*2 system:
% graddu = [u(1)*u(5)^2; u(2)*u(6)^2; u(3)*u(7)^2; u(4)*u(8)^2; u(5)*u(1)^2;u(6)*u(2)^2;u(7)*u(3)^2; u(8)*u(4)^2];
disp('Creating gradient function...');
graddu = graddu(:);
% Different regularization parameters for both images
RegMat = [alpha1*eye(M^2),zeros(M^2);zeros(M^2),alpha2*eye(M^2)];
gradF1 = 2*A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g))-2*b+RegMat*2*g+beta*graddu;
disp('Gradient function F1 ready!');
%% Decide the first step size
lambda = 0.0001;
oldu = g;
g = max(0,g-lambda*gradF1);
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
    gradF1 = 2*A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g))-2*b+RegMat*2*g+beta*graddu;
    % for historical reasons, we have this line here remembering how this
    % was before the second regularization term
    %gradF1 = 2*(A'*A)*u-2*A'*m+alpha*graddu;
    
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

% Here we calculated the square error separately for each phantom
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
%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(2);
% Originbal M1
subplot(2,2,1);
imagesc(reshape(M1,M,M));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of M1
subplot(2,2,2)
reco1=reshape(g(1:(N/2)),M,M);
imagesc(reco1);
colormap gray;
axis square;
axis off;
%title(['M1 BB reco1, \alpha_1 = ' num2str(alpha1)]);
title(['Relative error=', num2str(err_squ1), ', \alpha_1=', num2str(alpha1)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(M2,M,M));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of M2
subplot(2,2,4)
reco2=reshape(g(N/2+1:N),M,M);
imagesc(reco2);
colormap gray;
axis square;
axis off;
%title(['M2 BB reco2, iter=' num2str(K)]);
title(['Relative error=' num2str(err_squ2), ', \beta=' num2str(beta)]);
toc