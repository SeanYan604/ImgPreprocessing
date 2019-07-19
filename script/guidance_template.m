clear;
clc;
close all;

temp_num = 5;
background_sum = [];

for i=1:temp_num
    background = imread(strcat('../roi/region_',num2str(i),'.png'));
    pix_vect = pixExtraction(background);
    background_sum = [ background_sum; pix_vect];
end
    
% background_1 = imread('../roi/region_1.png');
% background_2 = imread('../roi/region_2.png');
% background_3 = imread('../roi/region_3.png');
% background_4 = imread('../roi/region_4.png');
% background_5 = imread('../roi/region_5.png');
% 
% 
% % imhist(background);
% pix_vect_1 = pixExtraction(background_1);
% pix_vect_2 = pixExtraction(background_2);
% pix_vect_3 = pixExtraction(background_3);
% pix_vect_4 = pixExtraction(background_4);
% pix_vect_5 = pixExtraction(background_5);

% figure;
% h = histogram(pix_vect_1,'EdgeColor','none','BinWidth',0.5,'FaceColor','m');
% axis([210,255,0,5000]);
% hold on;
% values = h.Values;
% edges = h.BinEdges;
% binwidth = h.BinWidth;
% centerpoints(1,:) = edges(1:end-1)+0.5*binwidth;
% centerpoints(2,:) = values;
% centerpoints(:,values(:)==0) = [];
% val = spcrv(centerpoints,4);
% plot(val(1,:),val(2,:),'-m','LineWidth',2);
% hold on;
% 
% 
% h = histogram(pix_vect_2,'EdgeColor','none','BinWidth',0.5,'FaceColor','#77AC30');
% axis([210,255,0,5000]);
% hold on;
% values = h.Values;
% edges = h.BinEdges;
% binwidth = h.BinWidth;
% centerpoints2(1,:) = edges(1:end-1)+0.5*binwidth;
% centerpoints2(2,:) = values;
% centerpoints2(:,values(:)==0) = [];
% val = spcrv(centerpoints2,4);
% plot(val(1,:),val(2,:),'-','Color','#77AC30','LineWidth',2);
% hold on;
% 
% h = histogram(pix_vect_3,'EdgeColor','none','BinWidth',0.5,'FaceColor','#D95319');
% axis([210,255,0,5000]);
% hold on;
% values = h.Values;
% edges = h.BinEdges;
% binwidth = h.BinWidth;
% centerpoints3(1,:) = edges(1:end-1)+0.5*binwidth;
% centerpoints3(2,:) = values;
% centerpoints3(:,values(:)==0) = [];
% val = spcrv(centerpoints3,4);
% plot(val(1,:),val(2,:),'-','Color','#D95319','LineWidth',2);
% hold on;
% 
% h = histogram(pix_vect_4,'EdgeColor','none','BinWidth',0.5,'FaceColor','b');
% axis([210,255,0,5000]);
% hold on;
% values = h.Values;
% edges = h.BinEdges;
% binwidth = h.BinWidth;
% centerpoints4(1,:) = edges(1:end-1)+0.5*binwidth;
% centerpoints4(2,:) = values;
% centerpoints4(:,values(:)==0) = [];
% val = spcrv(centerpoints4,4);
% plot(val(1,:),val(2,:),'-b','LineWidth',2);


figure(1);
% background_sum = [pix_vect_1;pix_vect_2;pix_vect_3;pix_vect_4;pix_vect_5];
%------------------------------------------------------------------
% avg = 250;
% [len,]=size(background_sum);
% for i=1:len
%     if(background_sum(i) < avg-5)
%         background_sum = [background_sum;background_sum(i)+2*(avg - background_sum(i))];
%     end
% end
%------------------------------------------------------------------
h = histogram(background_sum,'EdgeColor','none','BinWidth',0.5,'FaceColor','#0072BD');
axis([150,255,0,10000]);
hold on;
values = h.Values;
edges = h.BinEdges;
binwidth = h.BinWidth;
centerpoints_(1,:) = edges(1:end-1)+0.5*binwidth;
centerpoints_(2,:) = values;
centerpoints_(:,values(:)==0) = [];
% val = spcrv(centerpoints5,4);
% plot(val(1,:),val(2,:),'-','Color','#0072BD','LineWidth',2);

centerpoints_ = double(centerpoints_); 

x = centerpoints_(1,:);
y = centerpoints_(2,:);
[fitresult, gof] = createNormal_H(x, y);
x = edges(1:end-1)+0.5*binwidth;
% y = fitresult.a1 * normpdf(x, fitresult.b1, fitresult.c1);
y = fitresult.a1 * gaussmf(x, [fitresult.c1 / 1.414, fitresult.b1]);  %  得到fH

figure(1);
plot(x,y,'-','Color','#0072BD','LineWidth',2);

y2 = normpdf(x, fitresult.b1, fitresult.c1 / 1.414); % 得到f(t) = 1/c*fH

test_num = 27;
for o=1:test_num
    target_img = imread(strcat('../roi/region_',num2str(o),'.png'));
    img_compressed = CompressImg(target_img);
    [m,n] = size(img_compressed);

    syms t xi
    y_f = normpdf(t, fitresult.b1, fitresult.c1 / 1.414);
    %-------------------------------
    mu = fitresult.b1;
    sigma = fitresult.c1 / 1.414;
    save('para.mat','mu','sigma'); %保存func的参数mu和sigma
    %-------------------------------
    initial_template = zeros(m,n);
    Xi = calculateXi_(centerpoints_(2,:),m);


    for i = 1:m
        initial_template(i,1) = floor(Xi(i) + 0.5);
    end
    for i = 2:n
        initial_template(:,i) = initial_template(:,1);
    end

    % figure(4);
    % initial_template = uint8(initial_template);
    % imshow(initial_template);

    [defect_mask, defect_rgb] = subtraction_operation(initial_template, img_compressed, centerpoints_(2, :));

    [result, mask] = Uncompressing(defect_mask, target_img);
    figure;
    subplot(1,3,1);
    imshow(target_img);
    subplot(1,3,2);
    imshow(mask);
    subplot(1,3,3);
    imshow(result);

% imwrite(initial_template, 'initial_template.png');
end

