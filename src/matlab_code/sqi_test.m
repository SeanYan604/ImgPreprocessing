clear;
close all;

num = 0;
exposure = 6;
pic_num = 1;
img_file = strcat(sprintf('../data/sample_%d',num), sprintf('/%03d/%d.png', exposure, pic_num));
src_img = imread(img_file);

tic;
out_img = SQI(src_img);
toc

imshow(out_img); 