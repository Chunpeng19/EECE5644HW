% EECE5644 Exam1 Question 1
clear all, close all

mu(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4 5]; % mean and covariance of data pdf conditioned on label 3
mu(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0 2]; % mean and covariance of data pdf conditioned on label 2
mu(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
Nc = zeros(1,3); % sample numbers

figure(1),clf, colorList = 'rbg';
subplot(1,2,1)
for l = 1:3
    indices = find(thr(l)<u & u<=thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices)); Nc(l) = length(indices);
    x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
    plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
hold off
legend('Class 1','Class 2','Class 3')
title('Gaussian Distribution 2 Dimision Data of 3 Classes')
xlabel('x_1')
ylabel('x_2')

subplot(1,2,2)

evalResult = -[log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))+log(classPriors(1));
    log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))+log(classPriors(2));
    log(evalGaussian(x,mu(:,3),Sigma(:,:,3)))+log(classPriors(3))];
[M,D] = min(evalResult);

ind11 = find(D==1 & L==1); p11 = length(ind11)/Nc(1);
ind21 = find(D==2 & L==1); p21 = length(ind21)/Nc(1);
ind31 = find(D==3 & L==1); p31 = length(ind31)/Nc(1);

ind12 = find(D==1 & L==2); p12 = length(ind12)/Nc(2);
ind22 = find(D==2 & L==2); p22 = length(ind22)/Nc(2);
ind32 = find(D==3 & L==2); p32 = length(ind32)/Nc(2);

ind13 = find(D==1 & L==3); p13 = length(ind13)/Nc(3);
ind23 = find(D==2 & L==3); p23 = length(ind23)/Nc(3);
ind33 = find(D==3 & L==3); p33 = length(ind33)/Nc(3);

confusionMatrix = [length(ind11) length(ind12) length(ind13);
    length(ind21) length(ind22) length(ind23);
    length(ind31) length(ind32) length(ind33)];

misclassifiedNumber = length(ind21)+length(ind31)+length(ind12)+length(ind32)+length(ind13)+length(ind23);
p_error = [p21+p31,p12+p32,p13+p23]*Nc'/N;

plot(x(1,ind11),x(2,ind11),'og'), hold on
plot(x(1,ind22),x(2,ind22),'+g')
plot(x(1,ind33),x(2,ind33),'*g')

plot(x(1,ind21),x(2,ind21),'o','Color',[0.85 0.33 0.1])
plot(x(1,ind31),x(2,ind31),'o','Color',[0.93 0.69 0.13])

plot(x(1,ind12),x(2,ind12),'+','Color',[0 0.45 0.74])
plot(x(1,ind32),x(2,ind32),'+','Color',[0.93 0.69 0.13])

plot(x(1,ind13),x(2,ind13),'*','Color',[0 0.45 0.74])
plot(x(1,ind23),x(2,ind23),'*','Color',[0.85 0.33 0.1])

axis equal
legend('Class 1 with D = 1','Class 2 with D = 2','Class 3 with D = 3','Class 1 with D = 2','Class 1 with D = 3','Class 2 with D = 1','Class 2 with D = 3','Class 3 with D = 1','Class 3 with D = 2')
title('Classified Data and Their Decision')
xlabel('x_1')
ylabel('x_2')
hold off

Nc
confusionMatrix
misclassifiedNumber
p_error

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
