%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-07-01
% Source code source: 
%     G. Tran and R. Ward, "Exact Recovery of Chaotic Sysmtems from Highly Corrupted Data", https://arxiv.org/abs/1607.01067 


function [U,index_mislead,opts] = corrupted_data(X,dt,data)

if nargin < 2 || isempty(data) 
    data.ratio_corrupted = 0.008;
    data.sigma_corrupted = 50*dt;
    data.min_blocklength = 5;
    data.max_blocklength = 50;
    data.sigma_noise = 0.00;
end

if isfield(data, 'ratio_corrupted')
    ratio_corrupted = data.ratio_corrupted;
else
    ratio_corrupted = 0.008;
end
if isfield(data, 'sigma_corrupted')
    sigma_corrupted = data.sigma_corrupted;
else
    sigma_corrupted = 50*dt;
end
if isfield(data, 'min_blocklength')
    min_blocklength = data.min_blocklength;
else
    min_blocklength = 5;
end
if isfield(data, 'max_blocklength')
    max_blocklength = data.max_blocklength;
else
    max_blocklength = 50;
end
if isfield(data, 'sigma_noise')
    sigma_noise = data.sigma_noise;
else
    sigma_noise = 0.00;
end

U = X;
N = size(X,1);

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
%% AWGN noise
% U(index_mislead,1:size(X,2)) = awgn(U(index_mislead,1:size(X,2)),sigma_corrupted,'measured');
% % U = awgn(U,10);

U = U + sigma_noise*randn(size(X));
opts.ratio_corrupted = ratio_corrupted; 
opts.sigma_corrupted = sigma_corrupted; 
opts.min_blocklength = min_blocklength; 
opts.max_blocklength = max_blocklength;
opts.sigma_noise = sigma_noise;
return