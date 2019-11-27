function d = da_dn(n,a,param)
%LOGSIG.DA_DN Derivative of outputs with respect to inputs

% Copyright 2012-2015 The MathWorks, Inc.

  d = 1./(1+exp(n)).*exp(n);
end
