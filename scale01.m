% Scale an image (or any matrix) linearly to (a sub-interval of) the 
% interval [0,1].
%
% Arguments:
% im       image to be scaled
% minmax   optional vector of size 1x2 giving min and max value for scaling
%
% Returns:
% res      scaled image
%
% Samuli Siltanen May 2016

function res = scale01(im,minmax)

if nargin<2
    res = im-min(im(:));
    res = res/max(res(:));
else
    MIN = minmax(1);
    MAX = minmax(2);
    if MAX<=MIN
        error('minmax(2) must be strictly greater than minmax(1)')
    end
    if MAX>1
        error('minmax(2) cannot be smaller than max(im(:))')
    end
    if MIN>min(im(:))
        error('minmax(1) cannot be greater than min(im(:))')
    end
    res = im-MIN;
    res = res/(MAX-MIN);
end