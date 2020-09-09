path(path, [pwd '/CoShREM v11'])
path(path, [pwd '/CoShREM v11/Continuous Shearlet Transform'])
path(path, [pwd '/CoShREM v11/GUI'])
path(path, [pwd '/CoShREM v11/Mex'])
path(path, [pwd '/CoShREM v11/Util'])

% path(path, [pwd '/ShearLab3Dv11'])
% path(path, [pwd '/ShearLab3Dv11/Util'])
% path(path, [pwd '/ShearLab3Dv11/2D'])

path(path, [pwd '/images'])
%% Load image
%N = 512;
%im = zeros(N);
%im(1:N,128) = 1;
%im(128,1:N) = 1;
%im(129:N,312) = 0.2;
%imagesc(im);
%axis equal
%axis off
%colormap gray
%%
%im = FBP;%
im = phantom('Modified Shepp-Logan', N);
%im = f;
%im = max(0,recon2);

if ndims(im)>2
    im = im(:,:,1);
end
im = double(im);
[row,col,tmp] = size(im);

%% Create shearlet system
scales = 4;
disp('Creating shearlet system...')
tic
sh = SLgetShearletSystem2D(0,row,col,scales);
disp(['Done. Took: ' num2str(toc) ' seconds'])

% Calculate shearlet coefficients
disp('Calculating shearlet coefficients...')
tic
coeffs = SLsheardec2D(im,sh);
disp(['Done. Took: ' num2str(toc) ' seconds'])

%% Construction of the cleaner system
sc = shearletCleaner(im);
% NOTE: sc = shearletCleaner(zeros(size(im))) didn't work
% so shearlet cleaner needs to have im with information about object

%% Cleaning of the shearlet coefficients
coeffs_ori_cleaned1 = orientationCleaner(coeffs,sh,sc,1);
coeffs_ori_cleaned2 = orientationCleaner(coeffs,sh,sc);

%% convolution
scales = 'all';
nPSF = 3;
PSF = ones(nPSF);
PSF = conv2(PSF,PSF);
PSF = conv2(PSF,PSF);
PSF = PSF./sum(sum(PSF));

coeffs_convoluted1 = coeffs_conv(coeffs_ori_cleaned1,scales,PSF, sh.shearletIdxs);
coeffs_convoluted2 = coeffs_conv(coeffs_ori_cleaned2,scales,PSF, sh.shearletIdxs);
%% Plot coeffs
scale = 4;
for i = 1:size(coeffs,3)
    if sh.shearletIdxs(i,2) == scale
        txt = [sh.shearletIdxs(i,1) sh.shearletIdxs(i,2) sh.shearletIdxs(i,3)];
        figure(1)
        subplot(1,2,1)
        imagesc(abs(coeffs(:,:,i)))
        colorbar
        title(num2str(txt))

        subplot(1,2,2)
        imagesc(abs(coeffs_ori_cleaned2(:,:,i)))
        colorbar
        title(num2str(txt))
        pause
    end
end

%% Threshold function visualization

vec = linspace(0,pi,1000);
figure(2)
clf
plot(vec,sc.thresholdFunction(vec))

%% orientationThicken visualization
o = thickenOrientations(sc.orientations,1);
figure(9)
clf
subplot(1,2,1)
imagesc(sc.orientations)
title('Original')
subplot(1,2,2)
imagesc(o)
title('Thickened')
%% isocube
scale = 4;
indScale = sh.shearletIdxs(:,2) == scale;
cube = coeffs_convoluted1(:,:,indScale);
cube_nc = coeffs(:,:,indScale);
threshold = max(abs(cube(:)))/13;

figure(90)
clf
isosurface(cube,threshold)
xlim([1 N])
ylim([1 N])
axis square
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])
text(20,-35,0,'$x_1$','FontSize',16,'Interpreter','latex')
text(540,490,0,'$x_2$','FontSize',16,'Interpreter','latex')
text(0,-30,23,'$\theta$','FontSize',16,'Interpreter','latex')
view(70,75)