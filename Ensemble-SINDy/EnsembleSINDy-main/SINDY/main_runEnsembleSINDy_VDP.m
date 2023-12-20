%%%%%%%%%%%%%%%%%%%
% 
% run sims for one noise level and data length to plot ensemble forecasting
% and UQ  of van der Pol
%
%

clear all
close all
clc

%% sweep over a set of noise levels and data length to generate heatmap plots

% simulation time
tEnd = 15;

% at each noise level and simulation time, nTest different instantiations of noise are run (model errors and success rate are then averaged for plotting)
nTest1 = 1; % generate models nTest1 times for SINDy
nTest2 = 1; % generate models nTest times for ensemble SINDy


%% hyperparameters
% SINDy sparsifying hyperparameters
lambda = 0.2;

% ensemble hyperparameters
% data ensembling
nEnsembles = 100; % number of bootstraps (SINDy models using sampled data) in ensemble
ensembleT = 0.6; % Threshold model coefficient inclusion probability: set ensemble SINDy model coefficient to zero if inclusion probability is below ensembleT

% library
nEnsemble1P = 0.9; % percentage of full library that is sampled without replacement for library bagging
nEnsemble2 = 100; % number of bootstraps (SINDy models using sampled library terms) in ensemble
ensT = 0.4; % Threshold library term inclusion probabilty: cut library entries that occur less than ensT

% double bagging
nEnsemblesDD = 100; % number of models in ensemble for data bagging after library bagging


%% common parameters, true Lorenz system, signal power for noise calculation

% generate synthetic van der Pol system data
ode_params = {2}; 
x0 = [0 1]';
n = length(x0); 

% set common params
polys = 1:3;
trigs = [];
common_params = {polys,trigs};
gamma = 0;
tol_ode = 1e-10;         % set tolerance (abs and rel) of ode45
options = odeset('RelTol',tol_ode,'AbsTol',tol_ode*ones(1,length(x0)));
Beta = cell2mat(ode_params);

% time step
dt = 0.001;
tspan = 0:dt:tEnd;

% get true van der Pol system for comparison

true_nz_weights=[
0	-1
1	2
0	0
0	0
0	0
0	0
0	-2
0	0
0	0
    ];


%% general parameters

% smooth data using golay filter 
sgolayON = 1;

% generate data
[t,x]=ode45(@(t,x) vanderpol(t,x,Beta),tspan,x0,options);

% Corrputed data settings
params_data.ratio_corrupted = 0.00; % 损坏率 默认= 0.008
params_data.sigma_corrupted = 0.05; % 高斯噪声的标准偏差，添加在离群点(损坏)位置 默认= 50*dt
params_data.min_blocklength = 5; % 损坏块的最小长度 默认= 5
params_data.max_blocklength = 50; % 损坏块的最大长度 默认= 50
params_data.sigma_noise = 0.0; % 除了离群值外，噪声水平在每次测量时都增加 默认= 0.00
[xobs,index_mislead,opts_data] = corrupted_data(x,dt,params_data);

% data before smoothing for plotting
xobsPlotE = xobs;

% smooth data
if sgolayON 
    order = 3;
    framelen = 5;
    xobs = sgolayfilt(xobs,order,framelen);
end

% build library
Theta_0 = build_theta(xobs,common_params);


%% SINDy
% sindy with central difference differentiation
sindy = sindy_cd(xobs,Theta_0,n,lambda,gamma,dt);

% store outputs
nWrongTermsS = sum(sum(abs((true_nz_weights~=0) - (sindy~=0))));
modelErrorS = norm(sindy-true_nz_weights)/norm(true_nz_weights);
successS = norm((true_nz_weights~=0) - (sindy~=0))==0;


%% ENSEMBLES SINDY

%% calculate derivatives
% finite difference differentiation
dxobs_0 = zeros(size(x));
dxobs_0(1,:)=(-11/6*xobs(1,:) + 3*xobs(2,:) -3/2*xobs(3,:) + xobs(4,:)/3)/dt;
dxobs_0(2:size(xobs,1)-1,:) = (xobs(3:end,:)-xobs(1:end-2,:))/(2*dt);
dxobs_0(size(xobs,1),:) = (11/6*xobs(end,:) - 3*xobs(end-1,:) + 3/2*xobs(end-2,:) - xobs(end-3,:)/3)/dt;
            

%% double bagging SINDy
            
%% Bagging SINDy library
% randomly sample library terms without replacement and throw away terms
% with low inclusion probability
nEnsemble1 = round(nEnsemble1P*size(Theta_0,2));
mOutBS = zeros(nEnsemble1,n,nEnsemble2);
libOutBS = zeros(nEnsemble1,nEnsemble2);
for iii = 1:nEnsemble2
    rs = RandStream('mlfg6331_64','Seed',iii); 
    libOutBS(:,iii) = datasample(rs,1:size(Theta_0,2),nEnsemble1,'Replace',false)';
    mOutBS(:,:,iii) = sparsifyDynamics(Theta_0(:,libOutBS(:,iii)),dxobs_0,lambda,n,gamma);
end

inclProbBS = zeros(size(Theta_0,2),n);
for iii = 1:nEnsemble2
    for jjj = 1:n
        for kkk = 1:nEnsemble1
            if mOutBS(kkk,jjj,iii) ~= 0
                inclProbBS(libOutBS(kkk,iii),jjj) = inclProbBS(libOutBS(kkk,iii),jjj) + 1;
            end
        end
    end
end
inclProbBS = inclProbBS/nEnsemble2*size(Theta_0,2)/nEnsemble1;

XiD = zeros(size(Theta_0,2),n);
for iii = 1:n
    libEntry = inclProbBS(:,iii)>ensT;
    XiBias = sparsifyDynamics(Theta_0(:,libEntry),dxobs_0(:,iii),lambda,1,gamma);
    XiD(libEntry,iii) = XiBias;
end

                
%% Double bagging SINDy 
% randomly sample library terms without replacement and throw away terms
% with low inclusion probability
% then on smaller library, do bagging

XiDB = zeros(size(Theta_0,2),n);
XiDBmed = zeros(size(Theta_0,2),n);
XiDBs = zeros(size(Theta_0,2),n);
XiDBeOut = zeros(size(Theta_0,2),n,nEnsemblesDD);
inclProbDB = zeros(size(Theta_0,2),n);
for iii = 1:n
    libEntry = inclProbBS(:,iii)>ensT;

    bootstatDD = bootstrp(nEnsemblesDD,@(Theta,dx)sparsifyDynamics(Theta,dx,lambda,1,gamma),Theta_0(:,libEntry),dxobs_0(:,iii)); 
    
    XiDBe = [];
    XiDBnz = [];
    for iE = 1:nEnsemblesDD
        XiDBe(:,iE) = reshape(bootstatDD(iE,:),size(Theta_0(:,libEntry),2),1);
        XiDBnz(:,iE) = XiDBe(:,iE)~=0;
        
        XiDBeOut(libEntry,iii,iE) = XiDBe(:,iE);
    end

    % Thresholded bootstrap aggregating (bagging, from bootstrap aggregating)
    XiDBnzM = mean(XiDBnz,2); % mean of non-zero values in ensemble
    inclProbDB(libEntry,iii) = XiDBnzM;
    XiDBnzM(XiDBnzM<ensembleT) = 0; % threshold: set all parameters that have an inclusion probability below threshold to zero

    XiDBmean = mean(XiDBe,2);
    XiDBmedian = median(XiDBe,2);
    XiDBstd = std(XiDBe')';

    XiDBmean(XiDBnzM==0)=0; 
    XiDBmedian(XiDBnzM==0)=0; 
    XiDBstd(XiDBnzM==0)=0; 
    
    XiDB(libEntry,iii) = XiDBmean;
    XiDBmed(libEntry,iii) = XiDBmedian;
    XiDBs(libEntry,iii) = XiDBstd;
    
end


%% model error and success rates

nWrongTermsDE = sum(sum(abs((true_nz_weights~=0) - (XiD~=0))));
modelErrorDE = norm(XiD-true_nz_weights)/norm(true_nz_weights);
successDE = norm((true_nz_weights~=0) - (XiD~=0))==0;

nWrongTermsDDE = sum(sum(abs((true_nz_weights~=0) - (XiDB~=0))));
nWrongTermsDDE2 = sum(sum(abs((true_nz_weights~=0) - (XiDBmed~=0))));
modelErrorDDE = norm(XiDB-true_nz_weights)/norm(true_nz_weights);
modelErrorDDE2 = norm(XiDBmed-true_nz_weights)/norm(true_nz_weights);
successDDE = norm((true_nz_weights~=0) - (XiDB~=0))==0;
successDDE2 = norm((true_nz_weights~=0) - (XiDBmed~=0))==0;

% figure(4);
% plot(x(1:end,1),x(1:end,2),'r-', 'LineWidth', 1.5);
% hold on;
% plot(xobs(1:end,1),xobs(1:end,2),'k.', 'LineWidth', 1.5);
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
% xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
% ylabel('$\dot{x}$', 'Interpreter', 'latex', 'FontSize', 20);
% % xlim([-2.5, 2.5]); 
% % ylim([-4.5, 4.5]); 
% % axis tight; 
% h1=legend('Exact','Measured','FontSize',15);
% set(h1,'FontName', 'Times New Roman');
% hold off;
%% model error and success rate: SINDy, library bagging, double library bagging
modelError = [modelErrorS, modelErrorDE, modelErrorDDE]
successRate = [successS, successDE, successDDE]

RE_E = (XiDB(abs(true_nz_weights)>0) - true_nz_weights(abs(true_nz_weights)>0))./true_nz_weights(abs(true_nz_weights)>0);
MRE_E = max(abs(RE_E))*100

NF_E=nnz(XiDB)-nnz(true_nz_weights)
