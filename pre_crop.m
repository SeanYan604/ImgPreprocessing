clc;
close all;
clear;

model = "resize";

if(model == "crop")
    for i =1:9
        img = imread(strcat('./pic/',num2str(i),'.jpg'));
        img = rgb2gray(img);
        [m,n] = size(img);
        img = img(1:n,:);
        imwrite(img,strcat('./pic/',num2str(i),'.png'));
    end
elseif(model == "resize")
    for i =1:9
        img = imread(strcat('./pic/',num2str(i),'.jpg'));
        img = imresize(img,[256,256]);
        imwrite(img,strcat('./pic/',num2str(i),'.png'));
    end
end