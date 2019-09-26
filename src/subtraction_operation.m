function [D] = subtraction_operation(initial_template, img_compressed, distribution)
% test_img = imread('test_image_5.png');
img_compressed = uint8(img_compressed);
load('para.mat','mu','sigma','alpha');
% load('fitresult.mat','fitresult');

% sorting is canceled
% [test_img_sort,sort_index] = sort(img_compressed, 1);

test_img_sort = img_compressed;
figure('name','Result_1');
% subplot(1,2,1);
imshow(img_compressed);
% subplot(2,2,2);
% imshow(test_img_sort);
L=0;R=256;

% initial_template = imread('initial_template.png');
[m,n] = size(initial_template);
gap_pix = 20;
Vl = initial_template(gap_pix,1);
Vh = initial_template(m-gap_pix,1);
for k = 1:n-1
    if(test_img_sort(m,k)==0 && test_img_sort(m,k+1)>0)
        L=k;
    end
    if(test_img_sort(m,k)>0 && test_img_sort(m,k+1)==0)
        R=k+1;
    end
end
syms t xi
% y_f = normpdf(t, mu, sigma);
mask = zeros(m,n);
Qj = zeros(1,n);
C  = zeros(m,n);
Xjv = zeros(m,n);
for j = 1:n
    for i = 1:m
        if(test_img_sort(i,j)>=Vl && test_img_sort(i,j) <=Vh)
            mask(i,j) = 1;
            Qj(j)= Qj(j) + 1;
        end
    end
%     Xjv(1:Qj(j),j) = calculateXi_(distribution,Qj(j));
%     Xjv(1:Qj(j),j) = calculateXi(mu,sigma,Qj(j));
    Xjv(1:Qj(j),j) = calculateXi_tri(mu,sigma,alpha,Qj(j));
end
% subplot(2,2,3);
% imshow(uint8(255*mask));

Q_index = 1;
for j = 1:n
    for i = 1:m
        if(mask(i,j) == 1)
            C(i,j) = Xjv(Q_index, j);
            Q_index = Q_index + 1;
        elseif(j>L && j<R)
            C(i,j) = initial_template(i,j);
        end
    end
    Q_index = 1;
end
figure('name','Result_2');
% subplot(1,2,2);
imshow(uint8(C));

DrawCurve(test_img_sort, uint8(C));
D = C - double(test_img_sort);

end


