% EECE5644 HW1 Question 4_2
clear all, close all

% parameters
mu = 1;
sigma = sqrt(2);

% likelihood
x = -10:.1:10;
pdf1 = normpdf(x,0,1);
pdf2 = normpdf(x,mu,sigma);

% boundary
b1 = (mu+sigma*sqrt(mu^2+2*log(sigma)*(sigma^2-1)))/(1-sigma^2);
b2 = (mu-sigma*sqrt(mu^2+2*log(sigma)*(sigma^2-1)))/(1-sigma^2);

% posterior
post1 = pdf1./(pdf1+pdf2);
post2 = pdf2./(pdf1+pdf2);

% plot
figure(1)
plot(x,pdf1,'-b','LineWidth',1.5)
hold on
plot(x,pdf2,'-r','LineWidth',1.5)
line([b1 b1], get(gca, 'ylim'),'Color','black','LineStyle','--','LineWidth',1.5);
line([b2 b2], get(gca, 'ylim'),'Color','black','LineStyle','--','LineWidth',1.5);
xlabel('x')
ylabel('probability')
title('plot of class-conditional pdfs')
legend('N(0,1)','N(1,2)','Decision Boundary')
hold off

figure(2)
plot(x,post1,'-b','LineWidth',1.5)
hold on
plot(x,post2,'-r','LineWidth',1.5)
line([b1 b1], get(gca, 'ylim'),'Color','black','LineStyle','--','LineWidth',1.5);
line([b2 b2], get(gca, 'ylim'),'Color','black','LineStyle','--','LineWidth',1.5);
xlabel('x')
ylabel('probability')
title('plot of posterior probabilities')
legend('N(0,1)','N(1,2)','Decision Boundary')
hold off

% probability error
p = normcdf(b2,mu,sigma)-normcdf(b1,mu,sigma)+normcdf(b1,0,1)+1-normcdf(b2,0,1);