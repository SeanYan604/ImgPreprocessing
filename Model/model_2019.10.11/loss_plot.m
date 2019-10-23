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
load('Wh_evaluate_data.mat');
Precise_eval = data(1,:);
Recall_eval = data(2,:);
F1_eval = data(3,:);

load('Wl_evaluate_data.mat');
F1_eval_ = data(3,:);

[m,n] = size(Precise_eval);
step = 0.8/(n/11-1);
index = 0.1:step:0.9;
% values = spcrv([[index(1) index index(end)];[F1_eval(1) F1_eval F1_eval(end)]],10);
% values_ = spcrv([[index(1) index index(end)];[F1_eval_(1) F1_eval_ F1_eval_(end)]],10);

mean_f1_wh = zeros([m,n/11]);
mean_f1_wl = zeros([m,n/11]);
for i = 1: n/11
    mean_f1_wh(i) = max(F1_eval((i*11-10):i*11));
    mean_f1_wl(i) = max(F1_eval_((i*11-10):i*11));
end

% figure(1);
% plot(index,Precise_eval);
% hold on;
% plot(index,Recall_eval);
% hold on;
W_plot(index,mean_f1_wh);
W_plot(index,mean_f1_wl);
% hold on;
% plot(values_(1,:),values_(2,:));
% grid on;

% ============== plot evaluate data of DAAE and DAE =====
% data_daae = load('evaluate_data_daae.mat');
% data_dae = load('evaluate_data_dae.mat');
% 
% daae_matrix = data_daae.data;
% dae_matrix = data_dae.data;
% 
% num = 1:27;
% num = spcrv([num(1) num num(end)], 2);
% % Y_data = zeros(4,27);
% iou_daae = daae_matrix(1,:);
% F1_daae = daae_matrix(4,:);
% iou_dae = dae_matrix(1,:);
% F1_dae = dae_matrix(4,:);
% Y_data = spcrv([[iou_daae(1) iou_daae iou_daae(end)];[iou_dae(1) iou_dae iou_dae(end)];
%     [F1_daae(1) F1_daae F1_daae(end)];[F1_dae(1) F1_dae F1_dae(end)]], 2);
% 
% mean_daae = mean(daae_matrix, 2);
% mean_dae = mean(dae_matrix, 2);
% 
% createfigure(num, Y_data);


