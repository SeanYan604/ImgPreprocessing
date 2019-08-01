clear; clc; close all;

% Each line contains an ID and a list of settings for normalization
% and descriptor parameters. Here we only test default settings.
% For the meaning of each parameter, see below.
cases = [...
    %0  1    0    0  0  0  0 0  0.0  0 0    % no normalization
    1   0.2  1   -2  0  0  0 10 0.09 1 6;   % default setting
    %2  1    1   -2  0  0  0 10 0.09 1 6;   % no gamma
    %3  0.2  0    0  0  0  0 10 0.09 1 6;   % no DoG
    %4  0.2  1   -2  0  0  0 0  0.09 1 6;   % no equalization
    %5  0.2  1   -2  0  0  0 -10  0.09 1 6; % no tanh compression
    ];

for i = 1:size(cases,1)

    %*****************************************************
    c = cases(i,:);
    casenum = c(1);  % type of the parameter settings

    % parameter setting for preprocessing
    gamma = c(2);    % gamma parameter
    sigma0 = c(3);   % inner Gaussian size
    sigma1 = c(4);   % outer Gaussian size
    sx = c(5);       % x offset of centres of inner and outer filter
    sy = c(6);       % y offset of centres of inner and outer filter
    mask = c(7);     % mask
    do_norm = c(8);  % Normalize the spread of output values

    % parameter setting for LTP code and Distance Transform-based similarity
    % metric calculation
    thresh= c(9);       % threshold for LTP code
    dtalpha =c(10);     % alpha parameter for DT based distance
    dtthresh = c(11);   % threshold for truncating DT distance
    %*****************************************************

    if mask
       load('mask.mat');
       mask = double(mask1);
    else
       mask = [];
    end

    % preprocessing face images.......
    im1=double(imread('02463d282.pgm'));

    im1n = preproc2(im1,gamma,sigma0,sigma1,[sx,sy],mask,do_norm);
    subplot(221); imshow(im1,[]); title('before normalization');
    subplot(222); imshow(im1n,[]);   title('after normalization');

    im2=double(imread('02463d254.pgm'));

    im2n = preproc2(im2,gamma,sigma0,sigma1,[sx,sy],mask,do_norm);
    subplot(223); imshow(im2,[]); title('before normalization');
    subplot(224); imshow(im2n,[]);   title('after normalization');

    % preprocessing face images end.......

    [imgwidth,imglen]=size(im2);
    targetset=im1n(:);
    queryset=im2n(:);

    fprintf('\n\n ******** LBP features with different similarity metrics ******');
    % extract LBP features and compute the chi^2 distance between them
    subrow=18; subcol=21; % best setting for LBP with 130x150 images according to Ahonen's paper
    ta = lbp_image_set2(targetset,[imgwidth,imglen],[subrow,subcol]);
    te = lbp_image_set2(queryset,[imgwidth,imglen],[subrow,subcol]);
    mat_dist = dist_chi2(ta,te);
    fprintf('\n The chi2 distance ===> %3f',mat_dist);
    % extract LBP features and compute the distance transform-based distance between them
    ta0 = lbp_image_set0(targetset,[imgwidth,imglen]);
    te0 = lbp_image_set0(queryset,[imgwidth,imglen]);
    mat_dist = dist_lbp_disttrans(ta0,te0,[imgwidth,imglen],-1,dtalpha,dtthresh);
    if dtalpha<=0, mat_dist=-mat_dist; end
    fprintf('\n The DT distance ===> %3f',mat_dist);

    fprintf('\n\n ******** LTP features with different similarity metrics ******');
    % extract LTP features and compute the chi^2 distance between them
    [ta,ta0] = ltp_image_set2(targetset,[imgwidth,imglen],[subrow,subcol],thresh);
    [te,te0] = ltp_image_set2(queryset,[imgwidth,imglen],[subrow,subcol],thresh);
    mat_dist = dist_chi2(ta,te);
    fprintf('\n The chi2 distance ===> %3f',mat_dist);

    % extract LTP features and compute the distance transform based distance between them
    [tah,tal] = ltp_image_set0(targetset,[imgwidth,imglen],thresh); 
    [teh,tel] = ltp_image_set0(queryset,[imgwidth,imglen],thresh);

    mat_dist  =  dist_lbp_disttrans(tah,teh,[imgwidth,imglen],-1,dtalpha,dtthresh);
    mat_dist2 =  dist_lbp_disttrans(tal,tel,[imgwidth,imglen],-1,dtalpha,dtthresh);

    mat_dist=mat_dist+mat_dist2;

    if dtalpha<=0, mat_dist=-mat_dist; end

    fprintf('\n The DT distance ===> %3f\n',mat_dist);

end
