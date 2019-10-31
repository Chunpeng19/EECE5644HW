% EECE5644 HW3 Question 1
clear all, close all

delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
K = 10; % 10-fold cross validation
MM = 6; % model orders from 1 - 6

% Generate samples from a 4-component GMM
alpha_true = [0.1,0.2,0.3,0.4];
mu_true = [-10 -10 10 10;-10 10 10 -10];
Sigma_true(:,:,1) = [6 1;1 20];
Sigma_true(:,:,2) = [14 1;1 8];
Sigma_true(:,:,3) = [15 1;1 20];
Sigma_true(:,:,4) = [24 1;1 11];

N = 1000;
x = randGMM(N,alpha_true,mu_true,Sigma_true);
[d,~] = size(mu_true); % determine dimensionality of samples and number of GMM components

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Allocate space
GMMtrain = zeros(K,MM); GMMvalidate = zeros(K,MM); 
% AverageGMMtrain = zeros(1,MM); AverageGMMvalidate = zeros(1,MM);

% K-fold cross validation
for k = 1:K
    indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
    xValidate = x(:,indValidate); % Using folk k as validation set
    if k == 1
        indTrain = [indPartitionLimits(k,2)+1:N];
    elseif k == K
        indTrain = [1:indPartitionLimits(k,1)-1];
    else
        indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
    end
    xTrain = x(:,indTrain); % using all other folds as training set
    Ntrain = length(indTrain); Nvalidate = length(indValidate);
    % Initialize the GMM to randomly selected samples and train model parameters
    for M = 1:MM
        alpha = ones(1,M)/M;
        shuffledIndices = randperm(Ntrain);
        mu = xTrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
        [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); % assign each sample to the nearest mean
        for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
            Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        end
        Converged = 0; % Not converged at the beginning
        while ~Converged
            for l = 1:M
                temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            clear temp
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
            muNew = xTrain*w';
            for l = 1:M
                v = xTrain-repmat(muNew(:,l),1,Ntrain);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
        end
        GMMtrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)))/Ntrain;
        GMMvalidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)))/Nvalidate;
    end
end
AverageGMMtrain = mean(GMMtrain,1); % average training MSE over folds
AverageGMMvalidate = mean(GMMvalidate,1); % average validation MSE over folds

% semilogy([1:MM],AverageGMMtrain,'bo')
plot([1:MM],AverageGMMtrain,'bo')
hold on
% semilogy([1:MM],AverageGMMvalidate,'r*')
plot([1:MM],AverageGMMvalidate,'r*')
xlabel('Gaussian Model Components')
ylabel('Mean of Log-likelihood Probability')
legend('Training Data Performance','Validation Data Performance','Location','northwest')
title(['10-fold Cross-validation for 4 Compnents-GMM with ',num2str(N),' Samples'])

function x = randGMM(N,alpha,mu,Sigma)
    d = size(mu,1); % dimensionality of samples
    cum_alpha = [0,cumsum(alpha)];
    u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
    for m = 1:length(alpha)
        ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
        x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    end
end

function gmm = evalGMM(x,alpha,mu,Sigma)
    gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end