
% The released code is for academyic use purpose only, and WITHOUT ANY
% WARRANTY; For any commercial use, please contact HanHu.CAS@gmail.com.
% 
% 
% The LTV method demostrated here used a Total Variation package provided by Prof. Wotao Yin (http://www.caam.rice.edu/~optimization/L1/2007/09/software_08.html, GNU General Public License)
% 
% 
% 
% To get the details of the 13 methods, please refer to our paper:
% 
% Hu Han, Shiguang Shan, Xilin Chen, and Wen Gao. A Comparative Study on Illumination Preprocessing in Face Recognition. Pattern Recognition (P.R.), vol. 46, no. 6, pp. 1691-1699, Jun. 2013.
% 
% Please cite the above paper, if our code is used in your research.


% function IlluminationNormalization4OneImg()

% A comparative study on illumination preprocessing for one image
%
% 12 different illumination preprocessing methods
%   1 : Histogram equalization(HE)
%   2 : logarithmic transform(LT)
%   3 : gamma intensity correction(GIC)
%   4 : directional gray-scale derivative(DGDx and DGDy)
%   5 : Laplacian of Gaussian(LoG)
%   6 : single-scale Retinex(SSR)
%   7 : Gaussian high-pass(GHP)
%   8 : self-quotient image(SQI)
%   9 : logarithmic discrete cosine transform(LDCT)
%   10: logarithmic total variation(LTV)
%   11: local normalization(LN)
%   12: X.Tan,B.Triggs,Enhanced local texture feature sets for face recognition
%       under difficult lighting conditions,IEEE Transactions on Image Processing 19
%       (2010) 1635�C1650.


localMode = 0;

% input image
name_file = '00039_P00A-110E-20';
img_tes = imread(['./ImagesForDebug/' name_file '.bmp']);

% num = 27;
% for index = 1:num
%     path = '/home/seanyan/Desktop/july_projects/ImgPreprocessing/data/sample_0/006/';
%     wpath = '/home/seanyan/Desktop/july_projects/ImgPreprocessing/data/sample_0/006_IN/';
%     img_tes = imread(strcat(path, num2str(index),'.png'));
%     img_tes = imread('region_00.png');
if size(img_tes, 3) > 1
    img_tes = rgb2gray(img_tes);
end

%img_tes = imresize(img_tes, 0.8);

%img_tes = img_tes(:, 1 : size(img_tes, 2) / 2);
%img_tes = img_tes(:, size(img_tes, 2) / 2 + 1 : end);

% get image size
[row_img, col_img] = size(img_tes);

%% patch size
row_pat = 35;
col_pat = 35;
fprintf(['Patch size = ' num2str(row_pat) 'x' num2str(col_pat) '\n']);
%% overlap
step_pat = 9;
fprintf(['Patch step = ' num2str(step_pat) '\n']);

%% divide patch        
[numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat] = OverlappedBlock(row_img, col_img, row_pat, step_pat);
fprintf('Finish dividing patches\n');

figure;

% show original image
h = subplot(2, 7, 1);
set(h, 'position', [0.01 0.5 0.125 0.5]);
imshow(img_tes);
title('ORI');    


%% use different illumination preprocessing methods

% HE: histogram equalization
img_rst = HE(img_tes, localMode, numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat);

% show HE image
h = subplot(2, 7, 2);
set(h, 'position', [0.145 0.5 0.125 0.5]);
imshow(img_rst);
title('HE');

% write_path = strcat(wpath,'HE/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% LT: logarithmic transform
img_rst =  LT(img_tes);

% show LT image
h = subplot(2, 7, 3);
set(h, 'position', [0.28 0.5 0.125 0.5]);
imshow(img_rst);
title('LT')

% write_path = strcat(wpath,'LT/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% GIC: gamma intensity correction
% img_rst = GIC(img_tes, localMode, numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat);

% show GIC image
%     h = subplot(2, 7, 4);
%     set(h, 'position', [0.415 0.5 0.125 0.5]);
%     imshow(img_rst);
%     title('GIC');

%   DGD  
[FX, FY] = gradient(double(img_tes));
img_rst = uint8( mat2gray((FX+FY)*0.5) * 255 );

h = subplot(2, 7, 4);
set(h, 'position', [0.415 0.5 0.125 0.5]);
imshow(img_rst);
title('DGD');

% write_path = strcat(wpath,'DGD/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% DGDx: directional gray-scale derivative in X direction
img_rst = DX(img_tes);

% show DGDx image
h = subplot(2, 7, 5);
set(h, 'position', [0.55 0.5 0.125 0.5]);
imshow(img_rst);
title('DGDx');

% DGDy: directional gray-scale derivative
img_rst = DY(img_tes);

% show DGDy image
h = subplot(2, 7, 6);
set(h, 'position', [0.685 0.5 0.125 0.5]);
imshow(img_rst);
title('DGDy');

% LoG: Laplacian of Gaussian
img_rst = LoG(img_tes);

% show LoG image
h = subplot(2, 7, 7);
set(h, 'position', [0.82 0.5 0.125 0.5]);
imshow(img_rst);
title('LoG');

% write_path = strcat(wpath,'LOG/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% SSR: Single scale retinex
type_img     = 'GRAY8';  % 'GRAY8' or 'RGB24'
type_retinex = 2;        % which type of retinex, 2 is recommended
dGamma       = 2.0;      % gamma value for 'postlut' postprocessing
nIterations  = 4;        % number of iterations
img_rst = RETINEX(img_tes, type_img, type_retinex, dGamma, nIterations);

% show SSR image
h = subplot(2, 7, 8);
set(h, 'position', [0.01 0.2 0.125 0.5]);
imshow(img_rst);
title('SSR');

% write_path = strcat(wpath,'SSR/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% GHP: Gaussian high-pass
% I' = I - G_1.5 * I;
img_rst = GHP(img_tes);

% show GHP image
h = subplot(2, 7, 9);
set(h, 'position', [0.145 0.2 0.125 0.5]);
imshow(img_rst);
title('GHP');

% write_path = strcat(wpath,'GHP/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% SQI: Self-quotient image
img_rst = SQI(img_tes);

% show SQI image
h = subplot(2, 7, 10);
set(h, 'position', [0.28 0.2 0.125 0.5]);
imshow(img_rst);
title('SQI');

% write_path = strcat(wpath,'SQI/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% LDCT: Logarithmic discrete cosine transforms normalization
% 105x120: Ddis = 18--25
% 64x80:   Ddis = 11--16
blkPattern = 0;
Cov2LOG = true;
img_rst = DCTN(img_tes, Cov2LOG, blkPattern, numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat);

% show LDCT image
h = subplot(2, 7, 11);
set(h, 'position', [0.415 0.2 0.125 0.5]);
imshow(img_rst);
title('LDCT');

% write_path = strcat(wpath,'LDCT/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

% LTV: Logarithm total variation
nNeighbors = 4;
% Some suggestions in choosing lambda:
%lambda = 0.4;   %200x200
%lambda = 0.8;   %100x100
%lambda = 1.0;   %64x80

lambda = 0.4;
% img_rst = LTV(img_tes, lambda, nNeighbors);               

% show LTV image
h = subplot(2, 7, 12);
set(h, 'position', [0.55 0.2 0.125 0.5]);
imshow(img_rst);
title('LTV');

% LN: Local normalization
img_rst = LN(img_tes, localMode, numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat);

% show LN image
h = subplot(2, 7, 13);
set(h, 'position', [0.685 0.2 0.125 0.5]);
imshow(img_rst);
title('LN');

% References: X.Tan, B.Triggs, Enhanced local texture feature sets for face recognition
% under difficult lighting conditions, IEEE Transactions on Image Processing 19
% (2010) 1635�C1650.

cases = [...
    %0  1    0    0  0  0  0 0  0.0  0 0    % no normalization
    1   0.2  1   -2  0  0  0 10 0.09 1 6;   % default setting
    %2  1    1   -2  0  0  0 10 0.09 1 6;   % no gamma
    %3  0.2  0    0  0  0  0 10 0.09 1 6;   % no DoG
    %4  0.2  1   -2  0  0  0 0  0.09 1 6;   % no equalization
    %5  0.2  1   -2  0  0  0 -10  0.09 1 6; % no tanh compression
    ];
inx_case = 1;
c = cases(inx_case, :);
% parameter setting for preprocessing
gamma = c(2);    % gamma parameter
sigma0 = c(3);   % inner Gaussian size
sigma1 = c(4);   % outer Gaussian size
sx = c(5);       % x offset of centres of inner and outer filter
sy = c(6);       % y offset of centres of inner and outer filter
do_norm = c(8);  % Normalize the spread of output values
mask = [];

%% to ensure this program run successfully, you should add preproc2.m, gauss.m and gaussianfilter.m 
% from http://lear.inrialpes.fr/people/triggs/src/amfg07-demo-v1.tar.gz to current workplace
path = 'amfg07-demo-v1';
cd(path);
img_rst = preproc2( double(img_tes), gamma, sigma0, sigma1, [sx, sy], mask, do_norm);
img_rst = uint8( mat2gray(img_rst) * 255 );    

% show TT image
h = subplot(2, 7, 14);
set(h, 'position', [0.82 0.2 0.125 0.5]);
imshow(img_rst);
title('TT');

% write_path = strcat(wpath,'TT/',sprintf('%02d',index),'.png');
% imwrite(img_rst, write_path);

cd('..');
fprintf('Finish\n');
    
% end
% eof