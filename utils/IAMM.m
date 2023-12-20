%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-06-01
% Remark : IAMM to solve the outlier and noise problem

function [Xi,outlier,outputs,options] = IAMM(X, dt, Theta, options, lambda, mu)

% Set default values if input arguments are missing or options is empty
if nargin < 3 || isempty(options) 
    options.maxit = 100;
    options.tol = 1e-2;
end
options = setDefaultField(options, 'maxit', 100);
options = setDefaultField(options, 'tol', 1e-2);
maxit = options.maxit;
tol = options.tol;

% Calculate Trapezoidal integral
[Da,~] = size(Theta);  
Itheta = cumtrapz(Theta) * dt;
Itheta = [ones(Da-1, 1), Itheta(2:end,:)];
X=X(2:end,:);

%% Operators    
shrink2 = @(w,gamma) repmat(max(1-1./(gamma*sqrt(sum(w.^2,2))),0), 1, size(w,2)).*w;

%% Algorithm
% Initialization
outlier = zeros(size(X));
b = randn(size(X));
Xi = Itheta\(X - outlier - b);

iter = 0;
error_outlier = 1.0; 
while ((iter <= maxit) && (error_outlier > tol) )
    iter = iter+1;
    Xi_TS = Xi; 
    Xi = Itheta\(X - outlier - b );
    for iter_inner = 1:10 
        smallinds = (abs(Xi)<lambda);
        Xi(smallinds) = 0; 
        for ind = 1:size(X,2)
            biginds = ~smallinds(:,ind);
            Xi(biginds,ind) = Itheta(:,biginds)\(X(:,ind) - outlier(:,ind) - b(:,ind));
        end
    end

    rel_err_Xi(iter) = norm(Xi_TS - Xi,'fro');    
    outlier_TS = outlier;
    outlier = shrink2(X - Itheta*Xi - b , mu); 
    error_outlier = norm(outlier - outlier_TS,'fro')/norm(outlier_TS,'fro');
    rel_err_outlier(iter) = error_outlier;
    len_outlier(iter) = nnz(outlier);

    b = b + Itheta*Xi + outlier - X;
end

%% Objective function
ObFu1 = norm(X- Itheta*Xi - outlier, 2);
ObFu2 = nnz(Xi);
% ObFu2 = norm(Xi,1);

%%  outputs and options
outputs.iter = iter;
outputs.relerr_Xi = rel_err_Xi;
outputs.relerr_outlier = rel_err_outlier;
outputs.len_outlier = len_outlier;
outputs.ObFu1 = ObFu1;
outputs.ObFu2 = ObFu2;
%
options.maxit = maxit;
options.tol = tol;
options.coef_thres = lambda;
options.mu = mu;

end

function s = setDefaultField(s, field, defaultValue)
    if ~isfield(s, field)
        s.(field) = defaultValue;
    end
end