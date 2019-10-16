clc
clear 
close all

% =========================  plot tranning loss =======
% loss = load('loss.mat');
% data = loss.loss;
% epoch = data(1:350,1);
% G_loss = data(1:350,3);
% 
% loss_ = load('loss_ae.mat');
% data_ = loss_.loss_ae;
% epoch_ = data_(1:350,1);
% G_loss_ = data_(1:350,2);
% 
% figure(1);
% plot(epoch, G_loss);
% hold on;
% plot(epoch_, G_loss_);
% grid on;

% =========================  plot evaluate Wh =========
load('Wl_evaluate_data.mat');
Precise_eval = data(1,:);
Recall_eval = data(2,:);
F1_eval = data(3,:);

load('Wl_evaluate_data_.mat');
F1_eval_ = data(3,:);

[m,n] = size(Precise_eval);
step = 0.8/(n-1);
index = 0.1:step:0.9;
values = spcrv([[index(1) index index(end)];[F1_eval(1) F1_eval F1_eval(end)]],10);
values_ = spcrv([[index(1) index index(end)];[F1_eval_(1) F1_eval_ F1_eval_(end)]],10);
figure(1);
% plot(index,Precise_eval);
% hold on;
% plot(index,Recall_eval);
% hold on;
plot(values(1,:),values(2,:));
hold on;
plot(values_(1,:),values_(2,:));
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

