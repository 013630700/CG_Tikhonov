% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
clear all;

% Regularization parameter
alpha = 1;              

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

% Construct phantom. You can modify the resolution parameter N.
N      = 40;
target1 = imresize(double(imread('HY_Al.bmp')),[N N]);
target2 = imresize(double(imread('HY_square_inv.jpg')),[N N]);
% Vektorize
g1 = target1(:);
g2 = target2(:);
% Combine
g=[g1;g2];

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; 
angle0  = -90;
measang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% target = target1;
% P  = length(radon(target,0));
% M  = length(measang);
% A = sparse(M*P,N^2);
% 
% % Construct measurement matrix column by column. The trick is to construct
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
% Rtemp = radon(target,measang);
% Rtemp = Rtemp(:);
% Mtemp = A*target(:);
% disp(['If this number is small, then the matrix A is OK: ', num2str(max(max(abs(Mtemp-Rtemp))))]);
% 
% % Save the result to file (with filename containing the resolution N)
% eval(['save RadonMatrix', num2str(N), ' A measang target N P Nang']);

% Load radonMatrix
eval(['load RadonMatrix', num2str(N), ' A measang target N P Nang']);
a = A;
%% Load noisy measurements EIPÄS, tehdään RIKOS
%eval(['load XRMC_NoCrime', num2str(N), ' N mnc mncn']);
%[m,s] = radon(g1,measang);
% % If you want, you can make the sinogram by radon
% [m1,s] = radon(target1,measang);
% [m2,s] = radon(target2,measang);

% Check the sinograms: 
% figure(1)
% imshow(m1,[])
% figure(2)
% imshow(m2,[]);


m11 = c11*a*g1;% Energy 1 (low, 30kV)
m21 = c21*a*g1;% Energy 2 (high, 50kV)
m12 = c12*a*g2;% Energy 1 (low, 30kV)
m22 = c22*a*g2;% Energy 2 (high, 50kV)
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
% The positive constant alpha is the regularization parameter.o
%b     = ATmult(A,m);  %old version: b     = A.'*m(:);
b = A2x2Tmult(a,c11,c12,c21,c22,m1,m2);
size(b)
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
x   = b;          % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!

% very old version: Hx     = (A.')*(A*x) + alpha*x; 
% second try:       Hx     = (A.')*Amult(A,x) + alpha*x;
Hx     = A2x2mult(a,c11,c12,c21,c22,g1,g2);
size(Hx)
AT     = [c11*a.' c21*a.'; c12*a.' c22*a.'];
Hx     = AT*Hx;
Hx     = Hx + alpha*x;
r      = b-Hx;
r1     = r(1:1600);
r2     = r(1601:3200);
rho(1) = r(:).'*r(:);

%int p1 and p2
p1=zeros(size(g1));
p2=zeros(size(g2));

% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;
        p1 = r1;
        p2 = r2;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
        p1    = r1 + beta*p1;
        p2    = r2 + beta*p2;
    end
    % Tässä lasketaan matriisi kertolasku A*p. p on siis tilapäinen muuttuja
    % x:lle eli se vastaa kuvaa.
    Ap         = A2x2mult(a,c11,c12,c21,c22,p1,p2);
    % Tässä laskemme vielä transpoosin
    w          = AT*Ap;
    %Lisätään alpha*x regularisointi termi
    w          = w + alpha*x;
    w1         = w(1:1600);
    w2         = w(1601:3200);
    %En tiedä mikä on aS. Se on joku luku mikä tässä vaihtuu. Yksi
    %luku.Ehkä se on se residuaali?
    %p= [p1(:);p2(:)];
    aS          = rho(kkk)/(p.'*w);
    aS1         = rho(kkk)/(p1.'*w1);
    aS2         =rho(kkk)/(p2.'*w2);
    %Tässä kuvaan lisätään residuaali kertaa tilapäismuuttuva kuva
    x          = x + aS*p;
    %Mikä on r? r on myös kuva.
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    disp([kkk K])
end

recn = x;
recn1 = x(1:1600);
recn1 = reshape(recn1,N,N);
recn2 = x(1601:3200);
recn2 = reshape(recn2,N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
%err_sup = max(max(abs(g1-recn)))/max(max(abs(g1)));
%err_squ = norm(g1(:)-recn(:))/norm(g1(:));

% Save result to file
% eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% View the results
%XRMG_Tikhonov_plot(N)
%figure(100);
%imshow(recn,[]);
%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(3);
% Original target1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of target1
subplot(2,2,2)
imagesc(recn1);
colormap gray;
axis square;
axis off;
title('M1 CG recn1 ');
% Original target2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['M2 CG recn2, iter=' num2str(K)]);