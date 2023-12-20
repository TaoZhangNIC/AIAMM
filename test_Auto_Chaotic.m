%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-07-01
% Remark : test Auto chaotic system with IAMM (integral alternating minimization method)
%          Case1: Lorenz 63 system                [-5; 10; 30];
%          Case2: R{\"o}ssler system              [3; 5; 0]; 
%          Case3: Mean field model                [0.1; 0.1; 20]; 
%          Case4: Moore-Spiegel system            [-0.1; 0.1; 0.6];
%          Case5: Belousov-Zhabotinsky reaction   [0.5; 0.3; 0.8];
%          Case6: jerk circuit system             [0; 0; 0.5];
%          Case7: Chua circuit system             [0.1; 0.1; 0.1];
%          Case8: 5D autonomous system            [0.5; -0.5; 0.1; -1; 2]; 

clear; clc; close all;

addpath('./LIB','./utils','./ODE');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Algorithm parameters settings
caseNumber = 1;  % 1-8

% SmoothorNot = 'sgolay';
SmoothorNot = 'OriginNoise';
% SmoothorNot = 'TrueData';

% Model settings
switch caseNumber
    case 1
        dt = 0.001;      % Time step 0.0005;
        t = 0:dt:15;    % Time span
        odex =[-8.0; 7.0; 27.0];    % Initial conditions [-8.0; 7.0; 27.0];  [-5; 10; 30];
        systemName = 'Lorenz system';
    case 2
        dt = 0.001;      % Time step
        t = 0:dt:15;    % Time span
        odex = [3; 5; 0];    % Initial conditions
        systemName = 'Rossler system';
    case 3
        dt = 0.001;      % Time step
        t = 0:dt:10;    % Time span
        odex = [.1; .1; 20];     % Initial conditions
        systemName = 'Mean field model';
    case 4
        dt = 0.001;      % Time step
        t = 0:dt:15;    % Time span
        odex = [-0.1; 0.1; 0.6];     % Initial conditions
        systemName = 'Moore-Spiegel system';
    case 5
        dt = 0.001;      % Time step
        t = 0:dt:15;    % Time span
        odex = [0.5; 0.3; 0.8];     % Initial conditions
        systemName = 'Belousov-Zhabotinsky reaction';
    case 6
        dt = 0.001;      % Time step
        t = 0:dt:30;    % Time span
        odex = [0; 0; 0.5];     % Initial conditions
        systemName = 'Jerk circuit';
    case 7
        dt = 0.0005;      % Time step
        t = 0:dt:1;    % Time span
        odex = [0.1; 0.1; 0.1];     % Initial conditions
        systemName = 'Chua circuit';
    case 8
        dt = 0.0005;      % Time step
        t = 0:dt:20;    % Time span
%         odex = [1; 0; 0; 0; 4.9];     % Initial conditions
        odex = [2.8; 1; 20; 16.7; 55];     % Initial conditions
        systemName = '5D autonomous disc dynamo';        
    otherwise
        error('Invalid case number');
end

tE = 0:0.001:20;  % predicted time

% Corrputed data settings
data.ratio_corrupted = 0.05; 
data.sigma_corrupted = 0.1; 
data.min_blocklength = 5; 
data.max_blocklength = 50; 
data.sigma_noise = 0.05; 

% hyper-parameter settings
params_alg.tol = 5e-3 ;  
params_alg.maxit = 100; 
mu = 0.01; 

% Threshold parameter Settings
numlambda = 20; 
lambdastart = -2; 
lambdaend = 0;
lambdaseq = logspace(lambdastart,lambdaend, numlambda);

% The order setting of the library
X_OrderMax=3;
Trig_OrderMax=0;
nonsmooth_OrderMax=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True data by ODE
options = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[t, xr] = ode45(@(t,y)odeAutoChaos(t, y, systemName), t, odex, options);
[DataN, Dimension] = size(xr);
%% noise data
[xn,index_mislead,opts_data] = corrupted_data(xr,dt,data);

%% Denoising by  sgolayfilt
sgolayON = 1;
if sgolayON 
    order = Dimension;
    framelen = 5;
    xG = sgolayfilt(xn,order,framelen);
end

%% Data selection
% % SmoothorNot = 'sgolay' or 'OriginNoise' or 'TrueData';
switch SmoothorNot
    case 'sgolay'
        X=xG;
    case 'OriginNoise'
        X=xn;
    case 'TrueData'
        X=xr;
end

%% Construct Library
[Theta,Sym]=LIBA(X,X_OrderMax,Trig_OrderMax,nonsmooth_OrderMax);  % caseNumber
% [Theta,Sym]=LIBB(X,X_OrderMax,Trig_OrderMax,nonsmooth_OrderMax);  % caseNumber
%% IHT_Solve
Xi_IHT = cell(1, numel(lambdaseq));
ObFu1 = zeros(1, numel(lambdaseq));
ObFu2 = zeros(1, numel(lambdaseq));
for i=1:size(lambdaseq,2)
    lambda=lambdaseq(i);
    [Xi_IHT{i},outlier,outputs,options] = IAMM(X, dt, Theta, params_alg, lambda, mu);
    ObFu1(i)=outputs.ObFu1;
    ObFu2(i)=outputs.ObFu2;
end

% Calculate the distance between ObFu2 and ObFu1
distances = sqrt((min(ObFu2) - ObFu2).^2 + (min(ObFu1) - ObFu1).^2);
[min_distance, min_index] = min(distances);
lambda_min_distance = lambdaseq(min_index);
disp("minimum distance: " + min_distance);
disp("The serial number of the corresponding lambda: " + min_index);
disp("The corresponding lambda value: " + lambda_min_distance);
fprintf('Current system：%s\n', systemName);
disp("Initial Condition:  " +num2str(Xi_IHT{min_index}(1,:)));
disp("Dictionary Index: " + Sym' + "     |      Coefficient Value " + num2str(Xi_IHT{min_index}(2:end,:)));
% pause
fprintf('The percentage of corrupting data %2.2f\n',100*length(index_mislead)/size(xr,1));
%% Optimal parameter setting
% % mu_seq = [10 20 30 40 50 60 70 80 90 100];
% % mu_seq = [0.3 0.4 0.5 1 2 3 4 5 10 20 30 40 50 60 70 80 90 100];
mu_seq = logspace(-2,2, 50);
Xi_can = cell(1, numel(mu_seq));
ObFu1_can = zeros(1, numel(mu_seq));
ObFu2_can = zeros(1, numel(mu_seq));
for i=1:size(mu_seq,2)
    mu_can=mu_seq(i);
    [Xi_can{i},outlier_can,outputs_can,options_can] = IAMM(X, dt, Theta, params_alg, lambda_min_distance, mu_can);
    ObFu1_can(i)=outputs_can.ObFu1;
    ObFu2_can(i)=outputs_can.ObFu2;
end
% Calculate the distance between ObFu2 and ObFu1
distances_can = sqrt((min(ObFu2_can) - ObFu2_can).^2 + (min(ObFu1_can) - ObFu1_can).^2);
[min_distance_can, min_index_can] = min(distances_can);
mu_distance = mu_seq(min_index_can);
disp("minimum distance of mu: " + mu_distance);
disp("The serial number of the corresponding mu: " + min_index_can);
disp("The corresponding mu value: " + mu_distance);
fprintf('Current system：%s\n', systemName);
disp("Initial Condition:  " +num2str(Xi_can{min_index_can}(1,:)));
disp("Dictionary Index: " + Sym' + "     |      Coefficient Value " + num2str(Xi_can{min_index_can}(2:end,:)));


% %% Generate recovery data
% Xi=Xi_IHT{min_index};
Xi=Xi_can{min_index_can};
Xi_0 = Xi(1,:);         % Initial condition
% Xi_0 = XR(1,:); 
Xi_E = Xi(2:end,:)';    % Sparse coefficient
[~, xR] = ode45(@(t,y)odeAutoChaos(t, y, systemName), tE, odex, options);
% [~, xR] = ode45(@(t,y)odeAutoChaos(t, y, systemName), tE, XR(1,:), options);
[~, xiden] = ode45(@(t,y)RecoveryModel(t,y,Xi_E,systemName), tE, odex, options);
% [~, xiden] = ode45(@(t,y)RecoveryModel(t,y,Xi_E,systemName), tE, Xi_0, options);

%% P=3 Lorenz
Xi_true=[-10, 28, 0; 10, -1, 0; 0, 0, -8/3; 0, 0, 0; 0, 0, 1; 0, -1, 0; 0, 0, 0; 0, 0, 0;...
     0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0;...
      0, 0, 0]';
RE = (Xi_E(abs(Xi_E)>0) - Xi_true(abs(Xi_E)>0))./Xi_true(abs(Xi_E)>0);
MRE = max(abs(RE))*100;
fprintf('The maximum relative error %2.2f\n',MRE);
%% P=4 Lorenz
% Xi_true=[-10, 28, 0; 10, -1, 0; 0, 0, -8/3; 0, 0, 0; 0, 0, 1; 0, -1, 0; 0, 0, 0; 0, 0, 0;...
%      0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0;...
%       0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0;...
%        0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0]';
% RE = (Xi_E(abs(Xi_E)>0) - Xi_true(abs(Xi_E)>0))./Xi_true(abs(Xi_E)>0);
% MRE = max(abs(RE))*100;
% fprintf('The maximum relative error %2.2f\n',MRE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% graphic plotting
% True data and Contaminated data - Phase diagram
figure(1);
plot3(xr(1:end,1),xr(1:end,2),xr(1:end,3),'r-', 'LineWidth', 1.5);
hold on;
plot3(X(1:end,1),X(1:end,2),X(1:end,3),'k.', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 20);
zlabel('$z$', 'Interpreter', 'latex', 'FontSize', 20);
h1=legend('Exact','Measured','FontSize',15);
set(h1,'FontName', 'Times New Roman');
 view(27,16)
hold off;

% True data and Contaminated data - Time series diagram
figure(2);
subplot(3, 1, 1);
plot(t(1:end,1),xn(1:end,1),'k.', 'LineWidth', 1.5);
hold on;
plot(t(1:end,1),xr(1:end,1),'r-', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
hold on;
% axis tight; 
subplot(3, 1, 2);
plot(t(1:end,1),xn(1:end,2),'k.', 'LineWidth', 1.5);
hold on;
plot(t(1:end,1),xr(1:end,2),'r-', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 20);
% axis tight; 
subplot(3, 1, 3);
plot(t(1:end,1),xn(1:end,3),'k.', 'LineWidth', 1.5);
hold on;
plot(t(1:end,1),xr(1:end,3),'r-', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$z$', 'Interpreter', 'latex', 'FontSize', 20);
% axis tight; 
hold off;

% Threshold lambda Selection graph 3D: gray
figure(3);
plot3(lambdaseq, ObFu2, ObFu1,'k-o', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;
plot3(lambdaseq(min_index), ObFu2(min_index), ObFu1(min_index), 'r*', 'LineWidth', 1.5, 'MarkerSize', 10);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$|| \mathbf{\Xi} ||_0$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$|| \mathbf{X}  -\mathbf{\Theta}\mathbf{\Xi} - \mathbf{Outlier}  ||_2$',...
    'Interpreter', 'latex', 'FontSize', 15);

% Threshold lambda Selection graph 3D: Color
figure(4);
colormap('jet'); 
color_values = distances;  
scatter3(lambdaseq, ObFu2, ObFu1, 80, color_values, 'filled'); 
colorbar; 
hold on;
scatter3(lambdaseq(min_index), ObFu2(min_index), ObFu1(min_index), 300, min(distances), 'p', 'filled');
ax = gca; 
ax.GridColor = 'blue'; 
hold off;

% Recover data and real data  - Phase diagram of Origin data
figure(5);
plot3(xR(1:end,1),xR(1:end,2),xR(1:end,3),'r-', 'LineWidth', 1.5);
hold on;
plot3(xiden(1:end,1),xiden(1:end,2),xiden(1:end,3),'b--', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 20);
zlabel('$z$', 'Interpreter', 'latex', 'FontSize', 20);
h1=legend('Exact','AIAMM','FontSize',15);
set(h1,'FontName', 'Times New Roman');
hold off;

% Recover data and real data - Time series diagram
figure(6);
subplot(3, 1, 1);
plot(tE,xR(1:end,1),'r-', 'LineWidth', 1.5);
hold on;
plot(tE,xiden(1:end,1),'b--', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
hold on;
subplot(3, 1, 2);
plot(tE,xR(1:end,2),'r-', 'LineWidth', 1.5);
hold on;
plot(tE,xiden(1:end,2),'b--', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 20); 
subplot(3, 1, 3);
plot(tE,xR(1:end,3),'r-', 'LineWidth', 1.5);
hold on;
plot(tE,xiden(1:end,3),'b--', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$z$', 'Interpreter', 'latex', 'FontSize', 20);
hold off;
sgtitle('Exact vs AIAMM', 'Interpreter', 'latex', 'FontSize', 15);

% Threshold lambda Selection graph 3D: Color
figure(7);
colormap('jet');
scatter3(lambdaseq, ObFu2, ObFu1, 80, distances, 'filled');
colorbar;
clim([min(distances), max(distances)]);
hold on;

min_index = find(distances == min(distances));
scatter3(lambdaseq(min_index), ObFu2(min_index), ObFu1(min_index), 300, 'p', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'm');
xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$G_2$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$G_1$', 'Interpreter', 'latex', 'FontSize', 15);
hold off;

% Threshold mu Selection graph 3D: Color
figure(8);
colormap('jet');
scatter3(mu_seq, ObFu2_can, ObFu1_can, 80, distances_can, 'filled');
colorbar;
clim([min(distances_can), max(distances_can)]);
hold on;

min_index = find(distances_can == min(distances_can));
scatter3(mu_seq(min_index_can), ObFu2_can(min_index_can), ObFu1_can(min_index_can), 300, 'p', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'm');
xlabel('$\gamma$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$G_2$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$G_1$', 'Interpreter', 'latex', 'FontSize', 15);
hold off;

% Recover data and real data and measured data - Phase diagram of Predicted data
figure(9);
plot3(xR(1:end,1),xR(1:end,2),xR(1:end,3),'r-', 'LineWidth', 1.5);
hold on;
plot3(X(1:end,1),X(1:end,2),X(1:end,3),'k.', 'LineWidth', 1.5);
hold on;
plot3(xiden(1:end,1),xiden(1:end,2),xiden(1:end,3),'b--', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 20);
zlabel('$z$', 'Interpreter', 'latex', 'FontSize', 20);
h1=legend('Exact','Measured','AIAMM','FontSize',15);
set(h1,'FontName', 'Times New Roman');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%