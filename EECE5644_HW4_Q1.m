% EECE5644 HW4 Question 1
clear all, close all

delta = 1e-5; % tolerance for k-mean and EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
% pixel reading
% A = imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg');
A = imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg');
B = im2double(A);
[r,c,n] = size(A);
N = r*c; % number of samples
AClassified = zeros(r,c,n);

% feature vector
x = zeros(5,N); % x, y, R, G, B
xClassified = zeros(3,N);
for j = 1:c
    for i = 1:r
        x(1,i+(j-1)*r) = (j-1)/c; % x
        x(2,i+(j-1)*r) = (i-1)/r; % y
        x(3,i+(j-1)*r) = B(i,j,1); % R
        x(4,i+(j-1)*r) = B(i,j,2); % G
        x(5,i+(j-1)*r) = B(i,j,3); % B
    end
end

%% k-mean clustering algorithm
for K = 2:5
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:K)); % pick K random samples as initial mean estimates
    muNew = mu;
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    Converged = 0;
    while ~Converged
        [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
        for k = 1:K
            muNew(:,k)= mean(x(:,find(assignedCentroidLabels == k)),2);
        end
        Dmu = sum(sum(abs(muNew-mu)));
        Converged = (Dmu < delta); % Check if converged
        mu = muNew;
    end
    for k = 1:K
        xClassified(:,find(assignedCentroidLabels == k)) = repmat(mu(3:5,k),1,length(find(assignedCentroidLabels == k))); % assign mu value to classified x
    end
    for j = 1:c
        for i = 1:r
            for ii = 1:3
                AClassified(i,j,ii) = xClassified(ii,i+(j-1)*r);
            end
        end
    end
    figure(K-1)
    image(AClassified)
    title(['K-mean clustering classification using ',num2str(K),' segments'])
end

%% GMM-based clustering
d = 5;
for K = 2:5
    % GMM
    alpha = ones(1,K)/K;
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:K)); % pick K random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    for k = 1:K % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,k) = cov(x(:,find(assignedCentroidLabels == k))') + regWeight*eye(d,d);
    end
    Converged = 0;
    temp = zeros(K,N);
    while ~Converged
        for l = 1:K
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = x*w';
        for l = 1:K
            v = x-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(sum(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    end
    % MAP
    evalResult = zeros(K,N);
    for k = 1:K
        evalResult(k,:) = log(evalGaussian(x,mu(:,k),Sigma(:,:,k)))+log(alpha(k));
    end
    [Max,D] = max(evalResult);
    for k = 1:K
        xClassified(:,find(D == k)) = repmat(mu(3:5,k),1,length(find(D == k))); % assign mu value to classified x
    end
    for j = 1:c
        for i = 1:r
            for ii = 1:3
                AClassified(i,j,ii) = xClassified(ii,i+(j-1)*r);
            end
        end
    end
    figure(K+10-1)
    image(AClassified)
    title([num2str(K) '-GMM clustering and MAP classification'])
end