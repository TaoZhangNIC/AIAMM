%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy: script for recoverying various ODE systems
%%%%%%%%%%%% 
%%%%%%%%%%%% ode_num selects an ODE system from the list ode_names
%%%%%%%%%%%% tol_ode sets the tolerance (abs and rel) of ode45 
%%%%%%%%%%%% noise_ratio sets the signal-to-noise ratio (L2 sense)
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz
clear all, close all, clc

addpath('./data_usage_lorenz');

load('wsindy100.mat')
% load('0.05sigmar_0.008eta_0.05sigma_19.84eta%.mat')
load('0.05sigmar_0.008eta_0sigma_19.96eta%.mat')

xobs=xn;
tobs = t;
%% Time usage
% Define the percentage you want to extract (e.g., 5%, 10%, etc.)
percentages = 50; % You can adjust this list as needed [5, 10, 15, 20, 25, 50, 75, 100];

for i = 1:length(percentages)
    % Calculate the number of data points to extract based on the percentage
    percentage = percentages(i);
    num_points = round(percentage / 100 * length(xobs));    
    % Extract data starting from the beginning
    xobs = xobs(1:num_points, :);
    tobs = tobs(1:num_points, :);
end

%%% recover dynamics

tic;
common_params = {polys,trigs,lambda_mult,scale_Theta,gamma};
wsindy_params = {s, K, p, tau};

[Theta_0, tags, true_nz_weights, M_diag, lambda] = build_theta(xobs,weights,common_params,n);

w_sparse = zeros(size(Theta_0,2),n);
mats = cell(n,1);
ps_all = [];
ts_grids = cell(n,1);
RTs = cell(n,1);
Ys = cell(n,1);
Gs = cell(n,1);
bs = cell(n,1);

for i=1:n
    [Y,grid_i] = adaptive_grid(tobs,xobs(:,i),wsindy_params);
    [V,Vp,ab_grid,ps] = VVp_build_adaptive_whm(tobs,grid_i, r_whm, {0,inf,0});  %{pow,nrm,ord}. ord=0, ||phi||, ord=1, ||phi'|| 
    ps_all = [ps_all;ps];
    mats{i} = {V,Vp};
    ts_grids{i} = ab_grid;
    Ys{i} = Y;
    
    if useGLS == 1
        Cov = Vp*Vp'+10^-12*eye(size(V,1));
        [RT,flag] = chol(Cov);        
        RT = RT';
        G = RT \ (V*Theta_0);
        b = RT \ (Vp*xobs(:,i));
    else
        RT = (1./vecnorm(Vp,2,2));
        G = V*Theta_0.*RT;
        b = Vp*xobs(:,i).*RT;
    end        
   RTs{i} = RT;
   Gs{i} = G;
   bs{i} = b;
 
    if scale_Theta > 0
        w_sparse_temp = sparsifyDynamics(G.*(1./M_diag'),b,lambda,1,gamma);
        w_sparse(:,i) = (1./M_diag).*w_sparse_temp;
    else
        w_sparse(:,i) = sparsifyDynamics(G,b,lambda,1,gamma);
    end
end
ET = toc;

%%% get SINDy model

useFD = 0;                      % finite difference differentiation order, if =0, then uses TVdiff
w_sparse_sindy = standard_sindy(tobs,xobs,Theta_0,M_diag, useFD,n,lambda,gamma);
err_sindy = [norm(w_sparse_sindy(:)-true_nz_weights(:));norm(w_sparse_sindy(true_nz_weights~=0)-true_nz_weights(true_nz_weights~=0))]/norm(true_nz_weights(:));        

%%% visualize basis and covariance, display error analysis in command window

figure(3); clf
set(gcf, 'position', [1250 10 700 450])
for d=1:n
subplot(3,n,d) 
plot(tobs,xobs(:,d),'r-',tobs(floor(mean(ts_grids{d},2))),mean(xobs(:,d))*ones(size(ts_grids{d})),'.k')
subplot(3,n,n+d)
plot(tobs,mats{d}{1}')
subplot(3,n,2*n+d)
spy(RTs{d})
end
 
supp_range = [];
for i=1:n    
    supp_range = [supp_range; ts_grids{i}*[-1;1]];
end

clc;
disp(['log10 2norm err (all weights) (WSINDy)=',num2str(log10(norm(true_nz_weights(:)-w_sparse(:))/norm(true_nz_weights(:))))])
disp(['log10 2norm err (all weights) (SINDy)=',num2str(log10(err_sindy(1)))])
disp(['log10 2norm err (true nz weights) (WSINDy)=',num2str(log10(norm(true_nz_weights(true_nz_weights~=0)-w_sparse(true_nz_weights~=0))/norm(true_nz_weights(:))))])
disp(['log10 2norm err (true nz weights) (SINDy)=',num2str(log10(err_sindy(2)))])
disp(['min/mean/max deg=',num2str([min(ps_all) mean(ps_all) max(ps_all)])])
disp(['min/mean/max supp=',num2str([min(supp_range) mean(supp_range) max(supp_range)])])
disp(['Num Basis elts=',num2str(length(ps_all))])
disp(['Run time=',num2str(ET)])

figure(4);
plot(x(1:end,1),x(1:end,2),'r-', 'LineWidth', 1.5);
hold on;
plot(xobs(1:end,1),xobs(1:end,2),'k.', 'LineWidth', 1.5);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15, 'LineWidth', 1.5);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$\dot{x}$', 'Interpreter', 'latex', 'FontSize', 20);
% xlim([-2.5, 2.5]); 
% ylim([-4.5, 4.5]); 
% axis tight; 
h1=legend('Exact','Measured','FontSize',15);
set(h1,'FontName', 'Times New Roman');
hold off;

RE_W = (w_sparse(abs(true_nz_weights)>0) - true_nz_weights(abs(true_nz_weights)>0))./true_nz_weights(abs(true_nz_weights)>0);
MRE_W = max(abs(RE_W))*100

RE_S = (w_sparse_sindy(abs(true_nz_weights)>0) - true_nz_weights(abs(true_nz_weights)>0))./true_nz_weights(abs(true_nz_weights)>0);
MRE_S = max(abs(RE_S))*100

NF_W=nnz(w_sparse)-nnz(true_nz_weights)
NF_S=nnz(w_sparse_sindy)-nnz(true_nz_weights)
