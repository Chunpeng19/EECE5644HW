% EECE5644 HW3 Question 1
clear all, close all

% Generate samples from a 2-component GMM
alpha_true = [0.3 0.7]; thr = [0,cumsum(alpha_true)];
mu_true = [-2 1;-1 2];
[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components
Sigma_true(:,:,1) = [3 1;1 3];
Sigma_true(:,:,2) = [5 1;1 2];
N = 999;
error_count = zeros(1,3);
p_error = zeros(1,3);

u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
Nc = zeros(1,2); % sample numbers
for l = 1:2
    indices = find(thr(l)<u & u<=thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices)); Nc(l) = length(indices);
    x(:,indices) = mvnrnd(mu_true(:,l),Sigma_true(:,:,l),length(indices))';
end

%% MAP classifier
evalResult = [log(evalGaussian(x,mu_true(:,1),Sigma_true(:,:,1)))+log(alpha_true(1));
    log(evalGaussian(x,mu_true(:,2),Sigma_true(:,:,2)))+log(alpha_true(2))];
[Min,D] = max(evalResult);

ind11 = find(D==1 & L==1); p11 = length(ind11)/Nc(1);
ind21 = find(D==2 & L==1); p21 = length(ind21)/Nc(1);

ind12 = find(D==1 & L==2); p12 = length(ind12)/Nc(2);
ind22 = find(D==2 & L==2); p22 = length(ind22)/Nc(2);

confusionMatrix = [length(ind11) length(ind12); length(ind21) length(ind22)];

error_count(1) = length(ind21)+length(ind12);
p_error(1) = [p21,p12]*Nc'/N;

figure(1)
plot(x(1,ind11),x(2,ind11),'og'), hold on
plot(x(1,ind22),x(2,ind22),'+g')

plot(x(1,ind21),x(2,ind21),'or')
plot(x(1,ind12),x(2,ind12),'+r')

axis equal
legend('Class - with D -','Class + with D +','Class - with D +','Class + with D -')
title('Classified Data and Their Decision Using MAP')
xlabel('x_1')
ylabel('x_2')
hold off

%% Fisher LDA
% parameters estimation instead of using true parameters
delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates`
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end

Converged = 0; % Not converged at the beginning
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
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

if alpha(1) >= alpha(2)
    alpha = fliplr(alpha);
    mu = fliplr(mu);
    temp = Sigma(:,:,1);
    Sigma(:,:,1) = Sigma(:,:,2);
    Sigma(:,:,2) = temp;
end
clear temp

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
LLDA = L-ones(1,N);
wLDA = sign(mean(yLDA(find(LLDA==1)))-mean(yLDA(find(LLDA==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(LLDA==1)))-mean(yLDA(find(LLDA==0))))*yLDA; % flip yLDA accordingly

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * alpha(1)/alpha(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
[~,boundInd] = sort(abs(discriminantScore));
bLDA = (yLDA(boundInd(1))+yLDA(boundInd(2)))/2;
DLDA = (discriminantScore >= 0);
ind00 = find(DLDA==0 & LLDA==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(DLDA==1 & LLDA==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(DLDA==0 & LLDA==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(DLDA==1 & LLDA==1); p11 = length(ind11)/Nc(2); % probability of true positive

error_count(2) = length(ind10) + length(ind01);
p_error(2) = error_count(2)/N;

figure(2)
plot(x(1,ind00),x(2,ind00),'og'), hold on
plot(x(1,ind11),x(2,ind11),'+g')
plot(x(1,ind10),x(2,ind10),'or')
plot(x(1,ind01),x(2,ind01),'+r')

xb = linspace(min(x(1,:)),max(x(1,:)));
yb = -wLDA(1)/wLDA(2)*xb+bLDA/wLDA(2);
plot(xb,yb,'b')

axis equal,
legend('Class - with D -','Class + with D +','Class - with D +','Class + with D -','Decision Boundary')
title('Classified Data and Their Decision Using Fisher LDA')
xlabel('x_1')
ylabel('x_2')
hold off

%% logistic-linear classifier
fun = @(var) -1/N*(sum(log(1-1./(1+exp([var(1) var(2)]*x(:,find(L==1))+var(3)))))+sum(log(1./(1+exp([var(1) var(2)]*x(:,find(L==2))+var(3))))));
var0 = [wLDA' bLDA];
varm = fminsearch(fun,var0);

DLL = (1-1./(1+exp([varm(1) varm(2)]*x+varm(3))) < 1./(1+exp([varm(1) varm(2)]*x+varm(3))));
LLL = LLDA;
ind00 = find(DLL==0 & LLL==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(DLL==1 & LLL==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(DLL==0 & LLL==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(DLL==1 & LLL==1); p11 = length(ind11)/Nc(2); % probability of true positive

error_count(3) = length(ind10) + length(ind01);
p_error(3) = error_count(3)/N;

figure(3)
plot(x(1,ind00),x(2,ind00),'og'), hold on
plot(x(1,ind11),x(2,ind11),'+g')
plot(x(1,ind10),x(2,ind10),'or')
plot(x(1,ind01),x(2,ind01),'+r')

axis equal,
legend('Class - with D -','Class + with D +','Class - with D +','Class + with D -')
title('Classified Data and Their Decision Using Fisher Logistic-linear Classifier')
xlabel('x_1')
ylabel('x_2')
hold off

%%
error_count
p_error

%% function
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end