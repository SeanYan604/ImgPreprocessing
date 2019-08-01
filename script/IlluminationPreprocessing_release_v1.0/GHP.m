function resImg = GHP(srcImg)

% face illumination preprocessing using GHP method
% GHP: Gaussian high-pass
% Basic principle: Gradient or edge extraction
% References: A.Fitzgibbon, A.Zisserman, On affine invariant clustering and automatic cast
%             listing in movies, in: Proceedings of the European Conference on Computer Vision, 
%             2002,pp.304¨C320.
%
% INPUT:
% srcImg: the input image
%
% OUTPUT:
% resImg: the GHP image



%% default para
% hsize = 7;
% sigma = (hsize/2 - 1)*0.3 + 0.8;

%% 64x80
%sigma = 5.2;
%sigma = 3;
sigma = 2;
hsize = ((sigma - 0.8)/0.3 + 1) * 2;
hsize = double( round(hsize) );

h = fspecial('gaussian', hsize, sigma);
smoImg = imfilter(srcImg, h);
resImg = double(srcImg) - double(smoImg);
resImg = uint8( mat2gray(resImg) * 255 );
end
% eof