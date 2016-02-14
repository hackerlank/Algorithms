% One implementation of the Mahdavi's algorithm
% Copyright (c) 2016 Zilong Tan (eric.zltan@gmail.com)

% Permission is hereby granted, free of charge, to any person
% obtaining a copy of this software and associated documentation
% files (the "Software"), to deal in the Software without
% restriction, subject to the conditions listed in the Colossal
% LICENSE file. These conditions include: you must preserve this
% copyright notice, and you cannot mention the copyright holders in
% advertising related to the Software without their permission.  The
% Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
% notice is a summary of the Colossal LICENSE file; the license in
% that file is legally binding.

function [ Fval, Xval ] = mspd(func, X0, niter, bs, beta, alpha, lambda, tau, Ub)
Fval0 = func(X0);             % Objective at the initial point                  
n     = length(X0);           % Number of decision variables
k     = length(Fval0);        % Number of objective functions
Fval  = zeros(k, niter);      % Objective matrix
Xval  = zeros(niter, n);      % Decision variable matrix

Xval(1,:) = X0;
Fval(:,1) = Fval0;
Weight    = zeros(k-1,1);     % the first weight is always 1

DeltaX    = zeros(bs, n);     % Perturbation matrix
DeltaS    = zeros(bs, 1);     % Changes in the proxy model objective

for t = 2:niter 
    for j = 1:bs
        pertRange      = (t-1)^(-1/3);
        DeltaX(j,:)    = beta * ( 2*rand(1,n) - 1) * pertRange;
        DeltaS(j)      = [1;Weight]'*(func( Xval(t-1,:)+DeltaX(j,:) ) - Fval(:,t-1));
    end
    D = diag(exp(sum(-DeltaX.^2, 2) / 2 / tau^2));
    g = pinv(DeltaX'*D*DeltaX + lambda*eye(n))*DeltaX'*D*DeltaS;
    step = alpha / (1+alpha*lambda*(t-1));
    Xval(t,:) = Xval(t-1,:) -  step * g';
    Weight = Weight + step * (Fval(2:end,t-1) - Ub);
    Fval(:,t) = func(Xval(t,:));
end

end
