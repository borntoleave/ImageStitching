function R = cornerDetection(f, sigma, tau, radius, figId)
%
%   R = cornerDetection(f, sigma, tau, radius, figId)
%
% Implementation of corner detection using Harris method
%
% Input
%   f		: input image in grayscale
%   sigma	: standard deviation of Gaussian, 1 - 3 typically
%   tau		: (optional) threshold between 0 and 1, percentage of max of R
%   radius	: (optional) radius of region for non-maximal suppression, 1 - 3
%   figID	: (optional) number of figure where to show results
%
% Output
%   R		: Corner detection response matrix. If tau and radius are given
%		  then non-maximal suppression and threshold are applied to R.

% Fall 2010

% Input arguments must be at least the first two
error(nargchk(2, 5, nargin));

%
% Compute image gradient at each point in the image
%
G = fspecial('gaussian', max(1,fix(6*sigma)), sigma);
[Gx,Gy] = gradient(G);
fx = conv2(f, Gx, 'same');
fy = conv2(f, Gy, 'same');

%
% Compute the 3 terms of the second moment matrices M at each pixel
% The window size is the same as the Gaussian previously used
%
fx2 = conv2(fx.^2, ones(fix(6*sigma)), 'same');
fy2 = conv2(fy.^2, ones(fix(6*sigma)), 'same');
fxy = conv2(fx .* fy, ones(fix(6*sigma)), 'same');

%
% Compute the measure R = det(M) - alpha trace(M) for each pixel's M matrix
%
alpha = 0.04;
R = (fx2.*fy2 - fxy.^2) - alpha*(fx2 + fy2).^2; 

% If no optional parameters are given then return R untouched
if nargin == 2
   return;
end

% 
% Apply non-maximal suppression and thresholding using gray scale morphological
% dilation 
%
if nargin >= 4
   threshold = tau * max(max(R));

   % Dilate the image using the 2-D ordered statistic method
   Wsize = 2*radius + 1;
   Rdilated = ordfilt2(R, Wsize^2, true(Wsize));  

   R = (R == Rdilated) & (R > threshold);
end

if nargin == 5
   figure(figId);
   clf
   subplot(1,2,1)
   imagesc(f); axis image; colormap gray
   hold on
   [I,J] = find(R == 1);
   plot(J,I,'b.')

   subplot(1,2,2)
   imagesc(R); axis image; colormap gray
   title('Thresholded response')
end


