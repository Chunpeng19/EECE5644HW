function [x,label,Nc] = mixGaussian(N,mu,Sigma,p)
% function generates a mixture ofGaussians with speci?ed prior probabilities
% for each Gaussian class conditional pdf, as well as respective mean vectors
% and covariance matrices
%
% N: number of samples
% mu: mean matrix with n dimesions and c classes
% Sigma: covariance matrix
% p: prior probability vector for each class

[n,c] = size(mu); % number dimensions and classes

label = rand(1,N);
for i = 1:N
    sum_p = 0;
    for j = 1:c
        sum_p = sum_p + p(j);
        if label(i) <= sum_p
            label(i) = j-1;
            break;
        end
    end
end
Nc = zeros(1,c); % number of samples from each class
for i = 1:c
    Nc(i) = length(find(label==i-1));
end

x = zeros(n,N); % data
for l = 0:c-1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

