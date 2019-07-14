clc;
clear;
close all;

for k = 10:18
    img = imread(strcat('./pic/',num2str(k),'.png'));
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

    % se = strel('disk',8,8);  

    % sobel_mask = imdilate(sobel_mask, se);
    % sobel_mask = imerode(sobel_mask, se);
    % sobel_mask = medfilt2(sobel_mask);
    figure;
    imshow(255*uint8(sobel_mask));
    
    figure;
    imshow(BW);

    sobel_mask = medfilt2(sobel_mask);      % 中值滤波
    [filter_output,strong_angle]=angle_filter(sobel_mask, quantized_angle);
%     figure;
%     imshow(uint8(255*filter_output));
    
%     imwrite(255*uint8(filter_output), strcat('./sobel_mask/sobel_mask_',num2str(k),'.png'));
    % kernal_size = 3;
    % mask_pooling_1 = imgDownSample(filter_output, kernal_size, m, n, 'average');
    % angle_pooling_1 = imgDownSample(sobel_angle, kernal_size, m, n, 'average');
    % [m_,n_] = size(mask_pooling_1);
    % mask_pooling_2 = imgDownSample(mask_pooling_1, kernal_size, m_, n_, 'average');
    % angle_pooling_2 = imgDownSample(angle_pooling_1, kernal_size, m_, n_, 'average');

    % mask_level2 = imbinarize(mask_pooling_2);
    % figure;
    % imshow(uint8(255*mask_level2));
    % figure;
    % subplot(1,2,1);
    % imshow(uint8(255*mask_pooling_1));
    % subplot(1,2,2);
    % imshow(imbinarize(uint8(255*mask_pooling_2)));
end



