function [] = DrawCurve( sorted_image,template)

column_num = [122, 218];
[a,num] = size(column_num);
% [m,n] = size(sorted_image);
x = linspace(1,256,256);
for i = 1:num
    y_sort = sorted_image(:, column_num(i));
    y_temp = template(:,column_num(i));
    figure('name','ladder');
    plot(x,y_sort);
%     grid on;
    hold on;
    plot(x,y_temp);
    hold on; 
    
    y_result  = double(y_temp)-double(y_sort);
    figure('name','subresult');
    plot(x,y_result);
    axis([0,256,-50,150]);
    
end

