% Example computations related to X-ray tomography. This code computes 
% reconstructions for two materials and two energies system with Tikhonov 
% regularization and solves the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
%
% Revisited by Salla Latva-Äijö 17.11.2019

clear all;
% Regularization parameter
alpha = 0.001;

% Maximum number of iterations
MAXITER = 100;               

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

%% Load measurement matrix
%eval(['load RadonMatrix', num2str(N), ' A measang target N P Nang']);
% ei, vaan tässä lasketaan se!
% Construct phantom. You can modify the resolution parameter N.
%N      = 4;
%target1 =[1 1 1 1; 1 1 1 1; 1 1 0 0;1 1 0 0];
%target2= [0 0 0 0; 0 0 0 0; 0 0 1 1; 0 0 1 1];
% Construct phantom. You can modify the resolution parameter N.
N      = 40;
target1 = imresize(double(imread('HY_Al.bmp')),[N N]);
target2 = imresize(double(imread('HY_square_inv.jpg')),[N N]);

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; % Miksi N on sama kuin resoluutio? Sen ei tarvi olla.
angle0  = -90;
measang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% P  = length(radon(target1,0));
% M  = length(measang);
% A = sparse(M*P,N^2);
% 
% % Construct measurement matrix column by column. Matrix is same for both 
% % materials, so one is enought. The trick is to construct
% % targets with elements all 0 except for one element that equals 1.
% for mmm = 1:M
%     for iii = 1:N^2
%         tmpvec                  = zeros(N^2,1);
%         tmpvec(iii)             = 1;
%         A((mmm-1)*P+(1:P),iii) = radon(reshape(tmpvec,N,N),measang(mmm));
%         if mod(iii,100)==0
%             disp([mmm, M, iii, N^2])
%         end
%     end
% end
% 
% % Test the result
% Rtemp = radon(target1,measang);
% Rtemp = Rtemp(:);
% Mtemp = A*target1(:);
% disp(['If this number is small, then the matrix A is OK: ', num2str(max(max(abs(Mtemp-Rtemp))))]);
% 
% % Save the result to file (with filename containing the resolution N)
% eval(['save RadonMatrix', num2str(N), ' A measang target1 N P Nang']);

% Load radonMatrix
eval(['load RadonMatrix', num2str(N), ' A measang target N P Nang']);
a=A;

% % % %% Load noisy measurements EIPÄS, nut tehdään RIKOS
% % % %eval(['load XRMC_NoCrime', num2str(N), ' N mnc mncn']);
% % % [sinogram1,s] = radon(target1,measang);
% % % [sinogram2,s] = radon(target2,measang);

%% Kokeillaan korvata ylläoleva lasku matriisikertolaskuilla, koska 
% tämä on matriiseja käyttävä koodi, ja kaiken pitäisi toimia
m11 = c11*Amult(a,target1);% Energy 1 (low, 30kV)
m21 = c21*Amult(a,target2);% Energy 2 (high, 50kV)
m12 = c12*Amult(a,target1);% Energy 1 (low, 30kV)
m22 = c22*Amult(a,target2);% Energy 2 (high, 50kV)

% Sinograms of the simulated measurements: In reality the materials are in
% the same object like this:
m1 = m11 + m12;
m2 = m21 + m22;
m = [m1; m2];

% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% % The positive constant alpha is the regularization parameter.o
% b1     = A.'*sinogram1(:);
% b2     = A.'*sinogram2(:);

b1     = ATmult(a,m1);
b2     =ATmult(a,m2);
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
x1   = b1;          % initial iterate is the backprojected data
x2   = b2;
rho1 = zeros(K,1); % initialize parameters
rho2 = zeros(K,1);
% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!
Hx1     = ATmult(a,Amult(A,x1)) + alpha*x1;
Hx2     = ATmult(a,Amult(A,x2)) + alpha*x2;
r1      = b1-Hx1;
r2      = b2-Hx2;
rho1(1) = r1.'*r1;
rho2(1) = r2.'*r2;
% Start iteration
for kkk = 1:K
    if kkk==1
        p1 = r1;
        p2 = r2;
    else
        beta1 = rho1(kkk)/rho1(kkk-1);
        beta2 = rho2(kkk)/rho2(kkk-1);
        p1    = r1 + beta1*p1;
        p2    = r2 + beta2*p2;
    end
    w1          = ATmult(a,Amult(A,p1)) + alpha*p1;
    w2          = ATmult(a,Amult(A,p2)) + alpha*p2;
    a1          = rho1(kkk)/(p1.'*w1);
    a2          = rho2(kkk)/(p2.'*w2);
    x1          = x1 + a1*p1;
    x2          = x2 + a2*p2;
    r1          = r1 - a1*w1;
    r2          = r2 - a2*w2;
    rho1(kkk+1) = r1.'*r1;
    rho2(kkk+1) = r2.'*r2;
    disp([kkk K])
end
recn1 = reshape(x1,N,N);
recn2 = reshape(x2,N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
%err_sup = max(max(abs(target1-recn1)))/max(max(abs(target1)));
%err_squ = norm(target1(:)-recn1(:))/norm(target1(:));

% Save result to file
%eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(3);
% Original target1
subplot(2,2,1);
imagesc(reshape(target1,N,N));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of target1
subplot(2,2,2)
reco1=reshape(recn1,N,N);
imagesc(reco1);
colormap gray;
axis square;
axis off;
title('M1 BB reco1 ');
% Original target2
subplot(2,2,3)
imagesc(reshape(target2,N,N));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of target2
subplot(2,2,4)
reco2=reshape(x2,N,N);
imagesc(reco2);
colormap gray;
axis square;
axis off;
title(['M2 BB reco2, iter=' num2str(K)]);