% EECE5644 HW1 Question 4_2
clear all, close all

% parameters
n = 2; % number of feature dimensions
N = 1000; % number of iid samples
mu = [1;2]; % desired average vector
Sigma = [5 1;1 3]/10; % desired covariance matrix

% calculate A and b
[V,D] = eigs(Sigma); 
A = V*D^0.5;
b = mu;

% plot
z = mvnrnd([0;0],[1 0;0 1],N).'; % generate N samples data using N(0,I)
x = A*z+b; % transfer N(0,I) data to N(mu,Sigma)
plot(x(1,:),x(2,:),'+')
% hold on
% y = mvnrnd(mu,Sigma,N).';
% plot(y(1,:),y(2,:),'o')
xlabel('x_{i1} (i=1,2,...N)')
ylabel('x_{i2} (i=1,2,...N)')
title('n=2 dimensional randmom vectors plot')
