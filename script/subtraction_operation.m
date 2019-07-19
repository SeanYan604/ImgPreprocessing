function [defect_mask, defect_rgb] = subtraction_operation(initial_template, img_compressed, distribution)
% test_img = imread('test_image_5.png');
img_compressed = uint8(img_compressed);
load('para.mat','mu','sigma');
[test_img_sort,sort_index] = sort(img_compressed, 1);
figure('name','Result');
subplot(2,3,1);
imshow(img_compressed);
subplot(2,3,2);
imshow(test_img_sort);

% initial_template = imread('initial_template.png');
[m,n] = size(initial_template);
gap_pix = 7;
Vl = initial_template(gap_pix,1);
Vh = initial_template(m-gap_pix,1);

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
    Xjv(1:Qj(j),j) = calculateXi_(distribution,Qj(j));
end
subplot(2,3,3);
imshow(uint8(255*mask));

Q_index = 1;
for j = 1:n
    for i = 1:m
        if(mask(i,j) == 1)
            C(i,j) = Xjv(Q_index, j);
            Q_index = Q_index + 1;
        else
            C(i,j) = initial_template(i,j);
        end
    end
    Q_index = 1;
end
subplot(2,3,4);
imshow(uint8(C));

D = C - double(test_img_sort);

% reverse sort
R = zeros(m,n);
for i = 1:m
    for j = 1:n
        R(sort_index(i,j),j) = D(i,j);
    end
end
%-----------------------Defect Location byAdaptive Threshold
% L = (min(min(abs(D))));
% W = 30; H = 10;
% THl = W*log(sigma)/log(mu);
% % THl = max(THl, L);
% THh = THl + H;
THl = 20;
THh = 200;

defect_rgb = zeros(m,n,3);
defect_rgb(:,:,1) = img_compressed;
defect_rgb(:,:,2) = img_compressed;
defect_rgb(:,:,3) = img_compressed;
defect_mask = zeros(m,n);

for i =1:m
    for j = 1:n
        if(R(i,j) > THl || R(i,j) < -THh)
            defect_mask(i,j) = 255;
            defect_rgb(i,j,:) = [255,0,0];
        end
    end
end

subplot(2,3,5);
imshow(uint8(defect_mask));
subplot(2,3,6);
imshow(uint8(defect_rgb));
end


