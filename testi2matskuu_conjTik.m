% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% The following routines must be precomputed:
% XRMA_matrix_comp.m and XRMC_NoCrimeData_comp.m.
%
% Jennifer Mueller and Samuli Siltanen, October 2012
clear all;

% Regularization parameter
alpha = 10;

% Maximum number of iterations
MAXITER = 100;               

% Measure computation time later; start clocking here
tic

% % Define coefficients: Iodine and Al
% c11 = 42.2057; %Iodine 30kV
% c21 = 60.7376; %Iodine 50kV
% c12 = 3.044;   %Al 30kV
% c22 = 0.994;   %Al 50kV

%Blood and bone
c11 = 0.4083; %blood 30kV
c21 = 0.2947; %blood 50kV
c12 = 1.331;  %bone 30kV
c22 = 0.4242; %bone 50kV
%% Load measurement matrix
%eval(['load RadonMatrix', num2str(N), ' A measang target N P Nang']);
% ei, vaan tässänyt lasketaan se!
% Construct phantom. You can modify the resolution parameter N.
N      = 40;
I1 = imread('HY_Al.bmp');
target1 = double(I1);
target1 = imresize(target1, [N N]);

I2 = imread('HY_square_inv.jpg');
target2 = double(I2);
target2 = imresize(target2, [N N]);

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; 
angle0  = -90;
measang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
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

%% Load noisy measurements EIPÄS, tehdään RIKOS
% eval(['load XRMC_NoCrime', num2str(N), ' N mnc mncn']);

% % If you want, you can make the sinogram by radon
% [m1,s] = radon(target1,measang);
% [m2,s] = radon(target2,measang);

% Check the sinograms: 
% figure(1)
% imshow(m1,[])
% figure(2)
% imshow(m2,[]);

% Or you can simulate the measurements by multiplying with coefficients and
% matrices, then you need to
% Vectorize
target1 = target1(:);
target2 = target2(:);


m11 = c11*A*target1;% Energy 1 (low, 30kV)
m21 = c21*A*target1;% Energy 2 (high, 50kV)
m12 = c12*A*target2;% Energy 1 (low, 30kV)
m22 = c22*A*target2;% Energy 2 (high, 50kV)

% Sinograms of the simulated measurements: In reality the materials are in
% the same object like this:
m1 = m11 + m12;
m2 = m21 + m22;
% % % Vectorize
% % m1 = m1(:);
% % m2 = m2(:);
%%
% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter.o
% old version:    b     = A.'*m(:);
% second version: b     = ATmult(A,m);
b     = A2x2Tmult(A,c11,c12,c21,c22,m1,m2); % the new version, which combines the measurements into same vector b
% If you want you can check what b does to measurements:
% The look identical after A2x2Tmult.the materials have mixed.
% figure(13)
% b1 = b(1:1600);
%     b1 = reshape(b1,N,N);
%     imshow(b1,[]);
%     figure(14);
% b2 = b(1601:3200);
%     b2 = reshape(b2,N,N);
%     imshow(b2,[]);
%%
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
g1  = zeros(N);   % initialization
g2  = zeros(N);   % initialization%old version x   = b;         
% initial iterate is the backprojected data
g   = b;

rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!
% old: Hx     = (A.')*(A*x) + alpha*x; 
% second: Hx     = (A.')*Amult(A,x) + alpha*x;

Hx      = A2x2mult(A,c11,c12,c21,c22,g1,g2) + alpha*g(:);
%Hx     = A2x2Tmult(A,A2x2mult(A,c11,c12,c21,c22,g1,g2)) + alpha*g(:);
%Hx     = A2x2Tmult(A,A2x2mult) + alpha*x;
r      = b-Hx;
r = r(:);
rho(1) = r.'*r;
figure(15); % You can monitor proceeding of the iteration
% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;%tilapäisesti muuttuva rekonstruktio on sama kuin b-HX
        p1 = r(1:1600);% Yritys jakaa p komponentteihinsa, koska funktio vaatii ne erikseen
        p2 = r(1600+1:end);
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
        p1   = r(1:1600) + beta*p1;
        p2   = r(1600+1:end) + beta*p2;
    end
    %p1        = r(1:1600);
    %p2        = r(1601:end);%r(N/2+1:N);
    w          = A2x2mult(A,c11,c12,c21,c22,p1,p2)+ alpha*p; % Tämän funktion takia pitää erotella toisistaan p1 ja p2. Mutta eikös idea ole, että saadaaan ulos vain yksi vektori
    a          = rho(kkk)/(p.'*w);
    g          = g + a*p;
    r          = r - a*w;
    rho(kkk+1) = r.'*r;
    g1 = g(1:1600);
    g1 = reshape(g1,N,N);
    %pause;
    imshow(g1,[]);
    g2 = g(1601:3200);
    g2 = reshape(g2,N,N);
    %pause;
    
    
    imshow(g2,[]);
    
    disp([kkk K])
end
%recn = g;
%recn = reshape(g,N,N); %x on kuva!!!! Nyt ei voi tehdä näin, koska 2 kuvaa

% Determine computation time
comptime = toc;

% % Compute relative errors
% err_sup = max(max(abs(target-recn)))/max(max(abs(target)));
% err_squ = norm(target(:)-recn(:))/norm(target(:));

% Save result to file
%eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% % Here we calculated the error separately for each phantom
% % square norm errors:
% % Square Error in Tikhonov reconstruction 1
% err_squ1 = norm(M1(:)-reco1(:))/norm(M1(:));
% disp(['Square norm relative error for first reconstruction: ', num2str(err_squ1)]);
% % Square Error in Tikhonov reconstruction 2
% err_squ2 = norm(M2(:)-reco2(:))/norm(M2(:));
% disp(['Square norm relative error for second reconstruction: ', num2str(err_squ2)]);

% % Sup norm errors:
% % Sup error in reco1
% err_sup1 = max(max(abs(M1(:)-reco1(:))))/max(max(abs(reco1)));
% disp(['Sup relative error for first reconstruction: ', num2str(err_sup1)]);
% % Sup error in reco2
% err_sup2 = max(max(abs(M2(:)-reco2(:))))/max(max(abs(reco2)));
% disp(['Sup relative error for second reconstruction: ', num2str(err_sup2)]);

%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(3);
% Originbal M1
subplot(2,2,1);
imagesc(reshape(target1,N,N));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of M1
subplot(2,2,2)
g1=reshape(g1,N,N);
imagesc(g1);
colormap gray;
axis square;
axis off;
title(['M1 BB reco1, \alpha_1 = ' num2str(alpha), ', \alpha_2 = ' num2str(alpha)]);
% Original M2
subplot(2,2,3)
imagesc(reshape(target2,N,N));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of M2
subplot(2,2,4)
g2=reshape(g2,N,N);
imagesc(g2);
colormap gray;
axis square;
axis off;
title(['M2 BB reco2, iter=' num2str(K)]);
toc