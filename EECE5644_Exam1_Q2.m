% EECE5644 Exam1 Question 3
clear all, close all

% parameters
sigma = 0.3;
sigmax = 0.25;
sigmay = 0.25;

% contour mesh
n = 100;
x = linspace(-2,2,n);
y = linspace(-2,2,n);
[X,Y] = meshgrid(x,y);

% true position
Sigma = [sigmax^2 0; 0 sigmay^2];
while(1)
    Theta_T = mvnrnd(zeros(2,1),Sigma)';
    if vecnorm(Theta_T)<1
        break;
    end
end

figure(2)
% case K = 1
K = 1; % # number of reference
subplot(2,2,1)
contourPlot(X,Y,K,sigma,sigmax,sigmay,n,Theta_T)

% case K = 2
K = 2; % # number of reference
subplot(2,2,2)
contourPlot(X,Y,K,sigma,sigmax,sigmay,n,Theta_T)

% case K = 3
K = 3; % # number of reference
subplot(2,2,3)
contourPlot(X,Y,K,sigma,sigmax,sigmay,n,Theta_T)

% case K = 4
K = 4; % # number of reference
subplot(2,2,4)
contourPlot(X,Y,K,sigma,sigmax,sigmay,n,Theta_T)

function contourPlot(X,Y,K,sigma,sigmax,sigmay,n,Theta_T)

% reference position
xr = zeros(1,K);
yr = zeros(1,K);
for i = 1:K
    xr(i) = cos(2*pi/K*(i-1));
    yr(i) = sin(2*pi/K*(i-1));
end
Theta_r = [xr; yr];

% measurements
mu = vecnorm(Theta_r-Theta_T*ones(1,K)); % dTi
while(1)
    r = mvnrnd(mu,sigma^2*eye(K)); % check wheter positive or not
    if min(r)>=0
        break;
    end
end

% MAP estimated function f
f1 = zeros(n);
for i = 1:K
    f1 = f1 + (r(i)*ones(n)-((X-Theta_r(1,i)*ones(n)).^2+(Y-Theta_r(2,i)*ones(n)).^2).^0.5).^2/sigma^2;
end
f = f1 + sigmax^(-2)*X.^2+sigmay^(-2)*Y.^2;

% plot
level = 0:2:200;
contour(X,Y,f,level)
% level = 0:2:20;
% contour(X,Y,f,level,'ShowText','on')
hold on
plot(Theta_T(1),Theta_T(2),'r+','LineWidth',2.0)
plot(Theta_r(1,:),Theta_r(2,:),'bo','LineWidth',2.0)
hold off
xlabel('x')
ylabel('y')
title(['Contour Plot of Estimated Position for Case ' num2str(K)])
legend('Contour plot','Ture position','Reference position')
end

