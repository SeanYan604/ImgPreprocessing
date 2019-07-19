clc
clear
close all

for k = 0:5:100
    img = imread(strcat('/home/seanyan/dataset/LINEMOD/ape/JPEGImages/',sprintf('%.6d',k),'.jpg'));
    img = rgb2gray(img);
    img = imresize(img,[256,256]);
    [m,n] = size(img);
    sobel_mag = zeros(m,n);
    sobel_angle = zeros(m,n);
    quantized_angle = zeros(m,n);
    sobel_mask = zeros(m,n);
    D = double(img);
    thresh = 150;

    w = fspecial('gaussian',[5,5],1);
    %replicate:图像大小通过赋值外边界的值来扩展
    %symmetric 图像大小通过沿自身的边界进行镜像映射扩展
    img = imfilter(img,w,'replicate');
    BW = edge(img,'canny',0.9);

    for i = 2:m - 1   %sobel边缘检测
        for j = 2:n - 1
            Gx = - (D(i+1,j-1) + D(i+1,j) + D(i+1,j+1)) + (D(i-1,j-1) + D(i-1,j) + D(i-1,j+1));
            Gy = (D(i-1,j+1) + D(i,j+1) + D(i+1,j+1)) - (D(i-1,j-1) + D(i,j-1) + D(i+1,j-1));
            sobel_mag(i,j) = sqrt(Gx^2+Gy^2); 
            sobel_angle(i,j) = atan2(Gy,Gx)*180/pi;                         % sobel梯度方向map
            quantized_angle(i,j) = quantizeAngle(atan2(Gy,Gx)*180/pi);    % sobel梯度方向量化map
            if(sqrt(Gx^2+Gy^2) >= thresh)
                sobel_mask(i,j) = 1;
            end
        end
    end  

    sobel_mask = medfilt2(sobel_mask);      % 中值滤波
    [filter_output,strong_angle]=angle_filter(sobel_mask, quantized_angle);

    figure;
    imshow(filter_output);
    imwrite(255*uint8(filter_output), strcat('../background/background_',num2str(k),'.png'));
end