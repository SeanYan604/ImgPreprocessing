clc
clear 
close all

loss = load('loss.mat');
data = loss.loss;
epoch = data(:,1);
G_loss = data(:,3);

figure(1);
plot(epoch, G_loss);
grid on;

% D_loss = data(:,2);
% epoch_ = data(:,1);
% real_score = data(:, 4);
% fake_score = data(:, 5);
% 
% figure(2);
% plot(epoch_, D_loss);
% grid on;
% figure(3);
% plot(epoch_, real_score);
% hold on;
% plot(epoch_, fake_score);
% grid on;

