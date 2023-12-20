%==========================================================================
% corrupted_data: add bandwith corruption to clean data X
% Input: 
%       X(m x n): Clean, time-varying measurements, kth row is the measurement value at time k*dt 
%                         where n = dimension of the ODE system
%                               m = # measurements
%       opts.: An optional struct of options
%               ratio_corrupted: corruption ratio
%               sigma_corrupted: standard deviation of the Gaussian noise, added at the outlier(corrupted) location
%               min_blocklength: min length of the corrupted block
%               max_blocklength: max length of the corrupted block
%               sigma_noise: noise level added at every measurement, in addition to the outlier
%
%
% Output:
%       U(m x n): data with corruption
%       index_mislead: all misled indices
%       opts: A complete struct of options, containing all the values that were used.                        
%
% Note: sigma_noise < dt and length of each corrupted block ranges in min_blocklength to max_blocklength
%
% Authors: Giang Tran and Rachel Ward
% Institution: The University of Texas at Austin
% Version: 1.0, Year: 2016
% More information can be found at: 
%     G. Tran and R. Ward, "Exact Recovery of Chaotic Sysmtems from Highly
%     Corrupted Data", https://arxiv.org/abs/1607.01067 
%==========================================================================
function [U,index_mislead,opts] = corrupted_data(X,dt,params_data)

if nargin < 2 || isempty(params_data) % 如果输入参数个数小于2或opts为空，则使用默认参数
    params_data.ratio_corrupted = 0.008;
    params_data.sigma_corrupted = 50*dt;
    params_data.min_blocklength = 5;
    params_data.max_blocklength = 50;
    params_data.sigma_noise = 0.00;
end

% 检查opts结构体中各字段是否存在，如果存在则使用输入的值，否则使用默认值
if isfield(params_data, 'ratio_corrupted')
    ratio_corrupted = params_data.ratio_corrupted;
else
    ratio_corrupted = 0.008;
end
if isfield(params_data, 'sigma_corrupted')
    sigma_corrupted = params_data.sigma_corrupted;
else
    sigma_corrupted = 50*dt;
end
if isfield(params_data, 'min_blocklength')
    min_blocklength = params_data.min_blocklength;
else
    min_blocklength = 5;
end
if isfield(params_data, 'max_blocklength')
    max_blocklength = params_data.max_blocklength;
else
    max_blocklength = 50;
end
if isfield(params_data, 'sigma_noise')
    sigma_noise = params_data.sigma_noise;
else
    sigma_noise = 0.00;
end

U = X;
N = size(X,1);% number of samples

%% Add noise + corrupted data
index_mislead = sort(randperm(N-1,round(N*ratio_corrupted)));% not include the last index

length_mislead = randi([min_blocklength max_blocklength],length(index_mislead),1);
% Random block size
index_matrix = zeros(max_blocklength, length(index_mislead));
for i=1:max_blocklength
    for j=1:length(index_mislead)
        if ((i<=length_mislead(j)) && ((index_mislead(j) + i-1) <= N))
            index_matrix(i,j) = index_mislead(j) + (i-1);
        end
    end
end
index_mislead = unique(nonzeros(index_matrix));
U(index_mislead,1:size(X,2)) = U(index_mislead,1:size(X,2)).*(1+ sigma_corrupted*randn(length(index_mislead),size(X,2)));
% U(index_mislead,1:size(X,2)) = awgn(U(index_mislead,1:size(X,2)),sigma_corrupted,'measured');


% U = U.*(1 + sigma_noise*randn(size(X)));
% U = awgn(U,10,'measured');
U = awgn(U,10);
% U = U + sigma_noise*randn(size(X));
opts.ratio_corrupted = ratio_corrupted; 
opts.sigma_corrupted = sigma_corrupted; 
opts.min_blocklength = min_blocklength; 
opts.max_blocklength = max_blocklength;
opts.sigma_noise = sigma_noise;
return