% EECE5644 HW2 Question 2
clear all, close all

p_error = zeros(1,6);
N = 400;

%% case 1
figure(1)
mu = [0 0;3 3]';
Sigma(:,:,1) = [1 0; 0 1];
Sigma(:,:,2) = [1 0; 0 1];
p = [0.5 0.5]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(1) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(11)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

%% case 2
figure(2)
mu = [0 0; 3 3]';
Sigma(:,:,1) = [3 1; 1 0.8];
Sigma(:,:,2) = [3 1; 1 0.8];
p = [0.5 0.5]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(2) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(12)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

%% case 3
figure(3)
mu = [0 0;2 2]';
Sigma(:,:,1) = [2 0.5; 0.5 1];
Sigma(:,:,2) = [2 -1.9; -1.9 5];
p = [0.5 0.5]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(3) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(13)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

%% case 4
figure(4)
mu = [0 0;3 3]';
Sigma(:,:,1) = [1 0; 0 1];
Sigma(:,:,2) = [1 0; 0 1];
p = [0.05 0.95]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(4) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(14)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

%% case 5
figure(5)
mu = [0 0; 3 3]';
Sigma(:,:,1) = [3 1; 1 0.8];
Sigma(:,:,2) = [3 1; 1 0.8];
p = [0.05 0.95]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(5) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(15)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

%% case 6
figure(6)
mu = [0 0;2 2]';
Sigma(:,:,1) = [2 0.5; 0.5 1];
Sigma(:,:,2) = [2 -1.9; -1.9 5];
p = [0.05 0.95]';
[x,label,Nc] = mixGaussian(N,mu,Sigma,p);

% plot of generatered true data
subplot(1,2,1)
plot(x(1,label==0),x(2,label==0),'o')
set(gcf,'Position',[100 100 1000 500])
hold on
plot(x(1,label==1),x(2,label==1),'+')
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')
hold off

% plot of MAP classification
subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error(6) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
axis equal
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

% Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

figure(16)
subplot(1,2,1)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
set(gcf,'Position',[100 100 1000 500])
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2')
hold off

subplot(1,2,2)
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
muLDA = wLDA'*mu;
SigmaLDA1 = wLDA'*Sigma(:,:,1)*wLDA;
SigmaLDA2 = wLDA'*Sigma(:,:,2)*wLDA;
discriminantScore = log(evalGaussian(yLDA,muLDA(:,2),SigmaLDA2))-log(evalGaussian(yLDA,muLDA(:,1),SigmaLDA1))- log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
axis equal,
legend('Class 0 with D = 0','Class 0 with D = 1','Class 1 with D = 1','Class 1 with D = 0')
title('Classified LDA projection data and their decision')
xlabel('x_1')
ylabel('x_2')
hold off

p_error