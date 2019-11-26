clear all, close all,

plotData = 1;
n = 2; Ntrain = 1000; Ntest = 10000; 
alpha = [0.33,0.34,0.33]; % must add to 1.0
meanVectors = [-18 0 18;-8 0 8];
covEvalues = [3.2^2 0;0 0.6^2];
covEvectors(:,:,1) = [1 -1;1 1]/sqrt(2);
covEvectors(:,:,2) = [1 0;0 1];
covEvectors(:,:,3) = [1 -1;1 1]/sqrt(2);

t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

t = rand(1,Ntest);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtest = zeros(n,Ntrain);
Xtest(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtest(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtest(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

if plotData == 1
    figure(1), subplot(1,2,1),
    plot(Xtrain(1,:),Xtrain(2,:),'.')
    title('Training Data'), axis equal, xlabel('x_1'), ylabel('x_2')
    subplot(1,2,2),
    plot(Xtest(1,:),Xtest(2,:),'.')
    title('Testing Data'), axis equal, xlabel('x_1'), ylabel('x_2')
end

%% Training
inputs = Xtrain(1,:);
targets = Xtrain(2,:);

% Divide the data set into K approximately-equal-sized partitions
K = 10;
dummy = ceil(linspace(0,Ntrain,K+1));
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
        
net.inputs{1}.size = 1;
net.layers{1}.size = 1;
net.layers{1}.transferFcn = 'logsig';  
% net.layers{1}.transferFcn = 'softplus';        
net.layers{2}.size = 1;
net.layers{2}.transferFcn = 'purelin';    
net.biases{1}.initFcn = 'rands';
net.biases{2}.initFcn = 'rands';      
net.inputWeights{1,1}.initFcn = 'rands';
net.layerWeights{2,1}.initFcn = 'rands';
        
net.performFcn = 'mse';
net.trainFcn = 'trainscg';

KK = 10; % Number of perceptrons
nnValidateError = zeros(K,KK);
for kk = 1:KK
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        inputsValidate = inputs(:,indValidate); % Using folk k as validation set
        targetsValidate = targets(:,indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:Ntrain];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):Ntrain];
        end
        inputsTrain = inputs(:,indTrain); % using all other folds as training set
        targetsTrain = targets(:,indTrain);
        NNtrain = length(indTrain); NNvalidate = length(indValidate);
      
        net.layers{1}.size = kk;       
        net = init(net);
        net = train(net,inputsTrain,targetsTrain);
        outputs = sim(net,inputsValidate);
      
        nnValidateError(k,kk) = (targetsValidate-outputs)*(targetsValidate-outputs)'/NNvalidate ;  
    end  
end
nnAverageValidateError = mean(nnValidateError);

%% node number selction
figure(2)
subplot(1,2,1)
plot([1:KK],nnAverageValidateError,'bo-','linewidth',1.5)
title('10-fold cross validation average validate error')
ylabel('error')
xlabel('node number')

% [~,Index] = min(nnAverageValidateError);
% error changing rate
errorChangingRate = zeros(1,KK-1);
delta = 0.01; % stop criterion
Index = 0;
for i = 1:KK-1
    errorChangingRate(i) = abs(nnAverageValidateError(i+1)-nnAverageValidateError(i))/nnAverageValidateError(i);
    if (errorChangingRate(i) < delta) && (Index == 0)
        Index = i+1;
    end
end

subplot(1,2,2)
plot([2:KK],errorChangingRate,'ro-','linewidth',1.5)
title('10-fold cross validation average validate error changing rate')
ylabel('error changing rate')
xlabel('node number')

%% final train
net = init(net);
net.layers{1}.size = Index;
net = train(net,inputs,targets);
outputs = sim(net,inputs);
nnError = (targets-outputs)*(targets-outputs)'/Ntrain;

figure(3)
plot(inputs,targets,'b.')
hold on
plot(inputs,outputs,'r.')
hold off
title('Comparison of estimated x_2 and original x_2 for training set')
ylabel('x2')
xlabel('x1')
legend('Original x_2','Estimated x_2')

%% test
testInputs = Xtest(1,:);
testTargets = Xtest(2,:);
testOutputs = sim(net,testInputs);
nnTestError = (testTargets-testOutputs)*(testTargets-testOutputs)'/Ntest;

figure(4)
plot(testInputs,testTargets,'b.')
hold on
plot(testInputs,testOutputs,'r.')
hold off
title('Comparison of estimated x_2 and original x_2 for testing set')
ylabel('x2')
xlabel('x1')
legend('Original x_2','Estimated x_2')