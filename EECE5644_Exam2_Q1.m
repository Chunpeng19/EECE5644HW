% EECE5644 Exam2 Question 1
clear all, close all

% Generate samples from a 4-component GMM
alpha = [0.1,0.2,0.3,0.4]; thr = [0,cumsum(alpha)];
mu = [2 -5 6 -2;3 -4 -3 4;-2 -4 4 2];
[d,M] = size(mu); % determine dimensionality of samples and number of GMM components
Sigma(:,:,1) = [6 1 1;1 10 1;1 1 8];
Sigma(:,:,2) = [5 1 1;1 8 1;1 1 9];
Sigma(:,:,3) = [8 1 1;1 4 1;1 1 8];
Sigma(:,:,4) = [11 1 1;1 3 1;1 1 5];

N = 1000; % number of samples
u = rand(1,N); L = zeros(1,N); x = zeros(d,N);
Nc = zeros(1,M); % sample numbers
for l = 1:4
    indices = find(thr(l)<u & u<=thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices)); Nc(l) = length(indices);
    x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
end

figure(1)
subplot(2,2,1)
plot3(x(1,L==1),x(2,L==1),x(3,L==1),'bo'), hold on
plot3(x(1,L==2),x(2,L==2),x(3,L==2),'r+')
plot3(x(1,L==3),x(2,L==3),x(3,L==3),'c*')
plot3(x(1,L==4),x(2,L==4),x(3,L==4),'ms'), axis equal
xlabel('x_1'), ylabel('x_2'), zlabel('x_3')
% legend('class 1 data','class 2 data','class 3 data','class 4 data')
title('Training set data')
hold off

subplot(2,2,2)
plot(x(1,L==1),x(2,L==1),'bo'), hold on
plot(x(1,L==2),x(2,L==2),'r+')
plot(x(1,L==3),x(2,L==3),'c*')
plot(x(1,L==4),x(2,L==4),'ms'), axis equal
xlabel('x_1'), ylabel('x_2')
legend('class 1 data','class 2 data','class 3 data','class 4 data')
title('Training set data')
hold off

subplot(2,2,3)
plot(x(1,L==1),x(3,L==1),'bo'), hold on
plot(x(1,L==2),x(3,L==2),'r+')
plot(x(1,L==3),x(3,L==3),'c*')
plot(x(1,L==4),x(3,L==4),'ms'), axis equal
xlabel('x_1'), ylabel('x_3')
legend('class 1 data','class 2 data','class 3 data','class 4 data')
title('Training set data')
hold off

subplot(2,2,4)
plot(x(2,L==1),x(3,L==1),'bo'), hold on
plot(x(2,L==2),x(3,L==2),'r+')
plot(x(2,L==3),x(3,L==3),'c*')
plot(x(2,L==4),x(3,L==4),'ms'), axis equal
xlabel('x_2'), ylabel('x_3')
legend('class 1 data','class 2 data','class 3 data','class 4 data')
title('Training set data')
hold off

%% MAP classifier
NMAP = 10000; % number of samples
u = rand(1,NMAP); LMAP = zeros(1,NMAP); xMAP = zeros(d,NMAP);
Nc = zeros(1,M); % sample numbers
for l = 1:4
    indices = find(thr(l)<u & u<=thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    LMAP(1,indices) = l*ones(1,length(indices)); Nc(l) = length(indices);
    xMAP(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
end

evalResult = [log(evalGaussian(xMAP,mu(:,1),Sigma(:,:,1)))+log(alpha(1));
    log(evalGaussian(xMAP,mu(:,2),Sigma(:,:,2)))+log(alpha(2));
    log(evalGaussian(xMAP,mu(:,3),Sigma(:,:,3)))+log(alpha(3));
    log(evalGaussian(xMAP,mu(:,4),Sigma(:,:,4)))+log(alpha(4))];
[Max,D] = max(evalResult);

ind11 = find(D==1 & LMAP==1); p11 = length(ind11)/Nc(1);
ind21 = find(D==2 & LMAP==1); p21 = length(ind21)/Nc(1);
ind31 = find(D==3 & LMAP==1); p31 = length(ind31)/Nc(1);
ind41 = find(D==4 & LMAP==1); p41 = length(ind41)/Nc(1);

ind12 = find(D==1 & LMAP==2); p12 = length(ind12)/Nc(2);
ind22 = find(D==2 & LMAP==2); p22 = length(ind22)/Nc(2);
ind32 = find(D==3 & LMAP==2); p32 = length(ind32)/Nc(2);
ind42 = find(D==4 & LMAP==2); p42 = length(ind42)/Nc(2);

ind13 = find(D==1 & LMAP==3); p13 = length(ind13)/Nc(3);
ind23 = find(D==2 & LMAP==3); p23 = length(ind23)/Nc(3);
ind33 = find(D==3 & LMAP==3); p33 = length(ind33)/Nc(3);
ind43 = find(D==4 & LMAP==3); p43 = length(ind43)/Nc(3);

ind14 = find(D==1 & LMAP==4); p14 = length(ind14)/Nc(4);
ind24 = find(D==2 & LMAP==4); p24 = length(ind24)/Nc(4);
ind34 = find(D==3 & LMAP==4); p34 = length(ind34)/Nc(4);
ind44 = find(D==4 & LMAP==4); p44 = length(ind44)/Nc(4);

p_error = [(p21+p31+p41) (p12+p32+p42) (p13+p23+p43) (p14+p24+p34)]*Nc'/NMAP;

figure(2)
subplot(2,2,1)
plot3(xMAP(1,ind11),xMAP(2,ind11),xMAP(3,ind11),'og'), hold on
plot3(xMAP(1,ind22),xMAP(2,ind22),xMAP(3,ind22),'+g')
plot3(xMAP(1,ind33),xMAP(2,ind33),xMAP(3,ind33),'*g')
plot3(xMAP(1,ind44),xMAP(2,ind44),xMAP(3,ind44),'sg')

plot3(xMAP(1,ind21),xMAP(2,ind21),xMAP(3,ind21),'or')
plot3(xMAP(1,ind31),xMAP(2,ind31),xMAP(3,ind31),'oc')
plot3(xMAP(1,ind41),xMAP(2,ind41),xMAP(3,ind41),'om')

plot3(xMAP(1,ind12),xMAP(2,ind12),xMAP(3,ind12),'+b')
plot3(xMAP(1,ind32),xMAP(2,ind32),xMAP(3,ind32),'+c')
plot3(xMAP(1,ind42),xMAP(2,ind42),xMAP(3,ind42),'+m')

plot3(xMAP(1,ind13),xMAP(2,ind13),xMAP(3,ind13),'*b')
plot3(xMAP(1,ind23),xMAP(2,ind23),xMAP(3,ind23),'*r')
plot3(xMAP(1,ind43),xMAP(2,ind43),xMAP(3,ind43),'*m')

plot3(xMAP(1,ind14),xMAP(2,ind14),xMAP(3,ind14),'sb')
plot3(xMAP(1,ind24),xMAP(2,ind24),xMAP(3,ind24),'sr')
plot3(xMAP(1,ind34),xMAP(2,ind34),xMAP(3,ind34),'sc')

axis equal
legend('L = 1 with D = 1','L = 2 with D = 2','L = 3 with D = 3','L = 4 with D = 4','L = 1 with D = 2','L = 1 with D = 3','L = 1 with D = 4','L = 2 with D = 1','L = 2 with D = 3','L = 2 with D = 4','L = 3 with D = 1','L = 3 with D = 2','L = 3 with D = 4','L = 4 with D = 1','L = 4 with D = 2','L = 4 with D = 3')
title('Classified Data and Their Decision Using MAP')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
hold off

subplot(2,2,2)
plot(xMAP(1,ind11),xMAP(2,ind11),'og'), hold on
plot(xMAP(1,ind22),xMAP(2,ind22),'+g')
plot(xMAP(1,ind33),xMAP(2,ind33),'*g')
plot(xMAP(1,ind44),xMAP(2,ind44),'sg')

plot(xMAP(1,ind21),xMAP(2,ind21),'or')
plot(xMAP(1,ind31),xMAP(2,ind31),'oc')
plot(xMAP(1,ind41),xMAP(2,ind41),'om')

plot(xMAP(1,ind12),xMAP(2,ind12),'+b')
plot(xMAP(1,ind32),xMAP(2,ind32),'+c')
plot(xMAP(1,ind42),xMAP(2,ind42),'+m')

plot(xMAP(1,ind13),xMAP(2,ind13),'*b')
plot(xMAP(1,ind23),xMAP(2,ind23),'*r')
plot(xMAP(1,ind43),xMAP(2,ind43),'*m')

plot(xMAP(1,ind14),xMAP(2,ind14),'sb')
plot(xMAP(1,ind24),xMAP(2,ind24),'sr')
plot(xMAP(1,ind34),xMAP(2,ind34),'sc')

axis equal
% legend('L = 1 with D = 1','L = 2 with D = 2','L = 3 with D = 3','L = 4 with D = 4','L = 1 with D = 2','L = 1 with D = 3','L = 1 with D = 4','L = 2 with D = 1','L = 2 with D = 3','L = 2 with D = 4','L = 3 with D = 1','L = 3 with D = 2','L = 3 with D = 4','L = 4 with D = 1','L = 4 with D = 2','L = 4 with D = 3')
title('Classified Data and Their Decision Using MAP')
xlabel('x_1')
ylabel('x_2')
hold off

subplot(2,2,3)
plot(xMAP(1,ind11),xMAP(3,ind11),'og'), hold on
plot(xMAP(1,ind22),xMAP(3,ind22),'+g')
plot(xMAP(1,ind33),xMAP(3,ind33),'*g')
plot(xMAP(1,ind44),xMAP(3,ind44),'sg')

plot(xMAP(1,ind21),xMAP(3,ind21),'or')
plot(xMAP(1,ind31),xMAP(3,ind31),'oc')
plot(xMAP(1,ind41),xMAP(3,ind41),'om')

plot(xMAP(1,ind12),xMAP(3,ind12),'+b')
plot(xMAP(1,ind32),xMAP(3,ind32),'+c')
plot(xMAP(1,ind42),xMAP(3,ind42),'+m')

plot(xMAP(1,ind13),xMAP(3,ind13),'*b')
plot(xMAP(1,ind23),xMAP(3,ind23),'*r')
plot(xMAP(1,ind43),xMAP(3,ind43),'*m')

plot(xMAP(1,ind14),xMAP(3,ind14),'sb')
plot(xMAP(1,ind24),xMAP(3,ind24),'sr')
plot(xMAP(1,ind34),xMAP(3,ind34),'sc')

axis equal
% legend('L = 1 with D = 1','L = 2 with D = 2','L = 3 with D = 3','L = 4 with D = 4','L = 1 with D = 2','L = 1 with D = 3','L = 1 with D = 4','L = 2 with D = 1','L = 2 with D = 3','L = 2 with D = 4','L = 3 with D = 1','L = 3 with D = 2','L = 3 with D = 4','L = 4 with D = 1','L = 4 with D = 2','L = 4 with D = 3')
title('Classified Data and Their Decision Using MAP')
xlabel('x_1')
ylabel('x_3')
hold off

subplot(2,2,4)
plot(xMAP(2,ind11),xMAP(3,ind11),'og'), hold on
plot(xMAP(2,ind22),xMAP(3,ind22),'+g')
plot(xMAP(2,ind33),xMAP(3,ind33),'*g')
plot(xMAP(2,ind44),xMAP(3,ind44),'sg')

plot(xMAP(2,ind21),xMAP(3,ind21),'or')
plot(xMAP(2,ind31),xMAP(3,ind31),'oc')
plot(xMAP(2,ind41),xMAP(3,ind41),'om')

plot(xMAP(2,ind12),xMAP(3,ind12),'+b')
plot(xMAP(2,ind32),xMAP(3,ind32),'+c')
plot(xMAP(2,ind42),xMAP(3,ind42),'+m')

plot(xMAP(2,ind13),xMAP(3,ind13),'*b')
plot(xMAP(2,ind23),xMAP(3,ind23),'*r')
plot(xMAP(2,ind43),xMAP(3,ind43),'*m')

plot(xMAP(2,ind14),xMAP(3,ind14),'sb')
plot(xMAP(2,ind24),xMAP(3,ind24),'sr')
plot(xMAP(2,ind34),xMAP(3,ind34),'sc')

axis equal
% legend('L = 1 with D = 1','L = 2 with D = 2','L = 3 with D = 3','L = 4 with D = 4','L = 1 with D = 2','L = 1 with D = 3','L = 1 with D = 4','L = 2 with D = 1','L = 2 with D = 3','L = 2 with D = 4','L = 3 with D = 1','L = 3 with D = 2','L = 3 with D = 4','L = 4 with D = 1','L = 4 with D = 2','L = 4 with D = 3')
title('Classified Data and Their Decision Using MAP')
xlabel('x_2')
ylabel('x_3')
hold off

%% Neural network train
N = 100; % number of samples 100, 1000, 10000
u = rand(1,N); L = zeros(1,N); x = zeros(d,N);
Nc = zeros(1,M); % sample numbers
for l = 1:4
    indices = find(thr(l)<u & u<=thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices)); Nc(l) = length(indices);
    x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
end

targets = zeros(4,N);
for l = 1:4
    targets(l,find(L==l)) = ones(1,Nc(l));
end

% Divide the data set into K approximately-equal-sized partitions
K = 10;
nnValidateError = zeros(1,K);
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% neural network definition
net = network;
net.numInputs = 1;
net.numLayers = 2;
net.biasConnect = [1;1];
net.inputConnect = [1;0];
net.layerConnect = [0 0;1 0];
net.outputConnect = [0 1];
        
net.inputs{1}.size = d;
        
net.layers{1}.size = 1;
net.layers{1}.transferFcn = 'logsig';
        
net.layers{2}.size = 4;
net.layers{2}.transferFcn = 'softmax';    
net.biases{1}.initFcn = 'rands';
net.biases{2}.initFcn = 'rands';      
net.inputWeights{1,1}.initFcn = 'rands';
net.layerWeights{2,1}.initFcn = 'rands';
        
net.performFcn = 'crossentropy';
net.trainFcn = 'trainscg';

KK = 20; % Number of perceptrons
nnAverageValidateError = zeros(1,KK);
for kk = 1:KK
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        targetsValidate = targets(:,indValidate);
        LValidate = L(:,indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        xTrain = x(:,indTrain); % using all other folds as training set
        targetsTrain = targets(:,indTrain);
        Ntrain = length(indTrain); Nvalidate = length(indValidate);
      
        net.layers{1}.size = kk;       
        net = init(net);
        net = train(net,xTrain,targetsTrain);
        outputs = sim(net,xValidate);

        [~,DValidate] = max(outputs);
        nnValidateError(k) = 1-length(find(LValidate==DValidate))/Nvalidate;  
    end
    nnAverageValidateError(kk) = mean(nnValidateError);
end
[~,Index] = min(nnAverageValidateError);

figure(3)
plot(nnAverageValidateError,'b-','Linewidth',1.5), hold on
plot(nnAverageValidateError,'ro','Linewidth',1.5)
title('Average validate error of each node number')
xlabel('Node number')
ylabel('Error')
hold off

%% final train
% net = init(net);
net.biases{1}.initFcn = 'rands';
net.biases{2}.initFcn = 'rands';      
net.inputWeights{1,1}.initFcn = 'rands';
net.layerWeights{2,1}.initFcn = 'rands';
net.layers{1}.size = Index;
net = train(net,x,targets);

outputs = sim(net,x);
[~,D] = max(outputs);
nnError = 1-length(find(L==D))/N;

outputsMAP = sim(net,xMAP);
[~,DMAP] = max(outputsMAP);
nnErrorMAP = 1-length(find(LMAP==DMAP))/NMAP;

%% error comparison
j = log10(N)-1; % 1 2 3
nntrainingError = zeros(1,3);
nnMAPError = zeros(1,3);
nnTrainingError(j) = nnError*100;
nnMAPError(j) = nnErrorMAP*100;

figure(4)
% nnTrainingError = [0 5.11 6.96];
% nnMAPError = [10.56 7.71 7.02];
samples = [100 1000 10000];
semilogx(samples, nnTrainingError,'bo-','Linewidth',1.5)
hold on
semilogx(samples, nnMAPError,'ro-','Linewidth',1.5)
title('Errors Using Neural Network')
xlabel('sample number')
ylabel('Error(%)')
legend('Training set error', 'MAP data set error')
hold off

%% function
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end