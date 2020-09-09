% I try to make this work with the functions and matrices!
%% Level set method, HY example for one slice! case n=2
% This modified level set method allows target to change in time.
% Tomographic reconstruction is found as a minimizer of a nonlinear
% functional, which contains a regularization term penalizing the L^2 norms
% up to n derivatives of the reconstruction. For the case n=2 we aim to 
% minimize the functional F2 but drop for simplicity the mixed derivatives from the functional. 

% Salla Latva-Äijö 22.11.2019

clear all;
tic

%% Choises for the user
% Choose the size of the unknown. The image has size MxM.
M = 40;

% Choose the angles for tomographic projections
Nang = 40; % Odd number is preferred

%% Define attenuation coefficients c_ij of the two materials
% Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

%% Construct target ****************************ADD c*************************************
% Here we use simulated phantoms for both materials.
% Define overlapping materials: HY phantoms!
M1 = imresize(double(imread('HY_Al.bmp')), [M M]);
M2 = imresize(double(imread('HY_square_inv.jpg')), [M M]);

% Vektorize
g1 = M1(:);
g2 = M2(:);
% Combine
g=[g1;g2];
%% Definitions and initializations
% Some definitions
n = M*M;
ang = [0:(Nang-1)]*360/Nang;
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
a=A;

%% Start reconstruction
% Simulate noisy measurements; here including inverse crime
m = A2x2mult(a,c11,c12,c21,c22,g);
m = m+0.01*max(abs(m))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter
b = A2x2Tmult(a,c11,c12,c21,c22,m);

% Initialize g (the image vector containing both reconstuctions one after
% another)
%old version: g = zeros(2*M*M,1);
g = b; % initial iterate is the backprojected data
alpha1 = 100;% For better results alpha_1 >= 1
alpha2 = 20; % better results with small values!
%%
% Initial guess for u, u = kuva (eli g)
tt=1;
u = zeros(size(a,2)*tt,1);%u = zeros(size(A,2),1);
ss = sqrt(size(u,1)/tt); 
u = reshape(u,[ss,ss,tt]);

alpha = 0.5; % regularization parameter

ht = 1; % adjusts the amount of regularization in temporal direction

% First iteration
graddu2 = [];
graddu2=(8+4/ht)*u-2*([u(2:end,:,:);zeros(1,ss,tt)]+[u(:,2:end,:),...
zeros(ss,1,tt)]+[zeros(1,ss,tt);u(1:end-1,:,:)]+[zeros(ss,1,tt),...
u(:,1:end-1,:)])-(2/ht)*(cat(3,u(:,:,2:end),zeros(ss,ss,1))+...
cat(3,zeros(ss,ss,1),u(:,:,1:end-1)));
disp('Creating gradient function...');
graddu2 = graddu2(:);
u = u(:);

% New part, building delta function for replacing F1 with F2
delta = 10^-4;
f_delta = @(tau) (tau>0).*sqrt(tau.^2+delta);
df_delta = @(tau) (tau>0).*tau./sqrt(tau.^2+delta);
u = u(:);
% Make a diagonal matrix
%Df_delta = spdiags(df_delta(u),0,size(u,1),size(u,1));
%
gradF2 = 2*blkmulti(a',blkmulti(a,f_delta(u)))-2*blkmulti(a',m)+alpha*graddu2;
%gradF2 = 2*blkmulti(df_delta(u),blkmulti(A0',blkmulti(A0,f_delta(u))))-2*blkmulti(df_delta(u),blkmulti(A0',m))+alpha*graddu2;%gradF2 = 2*Df_delta*(A'*(A*f_delta(u)))-2*Df_delta*(A'*m)+alpha*graddu2;
disp('Gradient function ready!');

lam = .0001;

oldu = u;
u = u-lam*gradF2;

%% Iterate
disp('Iterating... ' );
for l = 1:14
graddu2 = [];
u = reshape(u,[ss,ss,tt]);
graddu2 = (32+4/ht+12/ht^2)*u-10*([u(2:end,:,:);-ones(1,ss,tt)]+...
[u(:,2:end,:),-ones(ss,1,tt)]+[-ones(1,ss,tt);u(1:end-1,:,:)]+...
[-ones(ss,1,tt),u(:,1:end-1,:)])+2*([u(3:end,:,:);-ones(2,ss,tt)]+...
[u(:,3:end,:),-ones(ss,2,tt)]+[-ones(2,ss,tt);u(1:end-2,:,:)]+...
[-ones(ss,2,tt),u(:,1:end-2,:)])-((2/ht)+8/ht^2)*(cat(3,u(:,:,2:end),...
-ones(ss,ss,1))+cat(3,-ones(ss,ss,1),u(:,:,1:end-1)))+(2/ht^2)*...
(cat(3,u(:,:,3:end),-ones(ss,ss,2))+cat(3,-ones(ss,ss,2),u(:,:,1:end-2)));

graddu2 = graddu2(:);
u = u(:);
 
%Df_delta = spdiags(df_delta(u),0,size(u,1),size(u,1));
oldgradF2 = gradF2;
%gradF2 = 2*A'*(A*u)-2*A'*m+alpha*graddu2;
gradF2=2*blkmulti(a',blkmulti(a,f_delta(u)))-2*blkmulti(a',m)+alpha*graddu2;
%gradF2 = 2*blkmulti(df_delta(u),blkmulti(A0',blkmulti(A0,f_delta(u))))-2*blkmulti(df_delta(u),blkmulti(A0',m))+alpha*graddu2;%gradF2 = 2*(A'*(A*f_delta(u)))-2*(A'*m)+alpha*graddu2;
lam = (u-oldu)'*(u-oldu)/((u-oldu)'*(gradF2-oldgradF2));
oldu = u;
u = u-lam*gradF2;
disp([num2str(l),'/7']);
end
%% Show reconstruction
u = reshape(u,[ss,ss,tt]);
u2 = f_delta(u);
for ii = 1:tt
figure(1);
imagesc(u2(:,:,ii));
title(['Frame ',num2str(ii)]);
axis square;
colormap gray;
%colorbar
pause(0.1);
end