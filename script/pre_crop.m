clc;
close all;
clear;

model = 'resize';

if(strcmp(model, 'crop'))
    for i =1:27
        img = imread(strcat('./pic/',num2str(i),'.jpg'));
        img = rgb2gray(img);
        [m,n] = size(img);
        img = img(1:n,:);
        imwrite(img,strcat('./pic/',num2str(i),'.png'));
    end
elseif(strcmp(model, 'resize'))
    for i =1:27
        img = imread(strcat('./pic/',num2str(i),'.png'));
        img_temp = zeros(256,256);
        [m,n] = size(img);
        img_temp(4:n+3,4:n+3) = img(:,:);  
        imwrite(uint8(img_temp),strcat('./pic/',num2str(i),'.png'));
    end
end