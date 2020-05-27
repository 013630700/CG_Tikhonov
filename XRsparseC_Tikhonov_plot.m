% Plot the results of routine XRsparseC_Tikhonov_comp.m
%
% Jennifer Mueller and Samuli Siltanen, October 2012

% Plot parameters
fsize     = 12;
thinline  = .5;
thickline = 2;

% Load reconstruction
load XRsparseTikhonov recn alpha target

% Compute relative error
err_squ = norm(target(:)-recn(:))/norm(target(:));

% Plot reconstruction image
figure(1)
clf
imagesc([target,recn],[0,1])
colormap gray
axis equal
axis off
title(['Tikhonov: error ', num2str(round(err_squ*100)), '%'])

% Plot profile of reconstruction
figure(2)
clf
plot(target(end/2,:),'k','linewidth',thinline)
hold on
plot(recn(end/2,:),'k','linewidth',thickline)
xlim([1 size(recn,1)])
axis square
box off
title('Profile of Tikhonov regularized reconstruction')






