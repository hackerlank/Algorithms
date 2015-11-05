% The PAreto Local Descent (PALD) Algorithm
% Copyright (c) 2015 Zilong Tan (eric.zltan@gmail.com)

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


function [ Fval, Xval ] = pald(func, X0, niter, bs, beta, alpha, lambda, tau, Ub)
%  @func    -- the multi-objective function to optimize
%  @X0      -- initial point
%  @niter   -- number of iterations
%  @bs      -- minibatch size
%  @beta    -- perturbation scaling factor
%  @alpha   -- step size
%  @lambda  -- regularization factor
%  @tau     -- bandwidth
%  @Ub      -- Upper bounding constraints

Fval0 = func(X0);             % Objective at the initial point                  
n     = length(X0);           % Number of decision variables
k     = length(Fval0);        % Number of objective functions
Fval  = zeros(k, niter);      % Objective matrix
Xval  = zeros(niter, n);      % Decision variable matrix
C     = 10;                   % z limit
UbM   = repmat(Ub,1,bs);

Xval(1,:) = X0;
Fval(:,1) = Fval0;

for j = 2:niter
    [NewY, DeltaX] = perturb(func, Xval(j-1,:), j-1, beta, bs, n, k);
    P      = ortho_proj(DeltaX, lambda, tau);
    YM     = repmat(Fval(:,j-1), 1, bs);
    JT     = jacob_trans(P, NewY, YM);
    Ni     = Fval(:,j-1) >= Ub;
    Weight = comp_weight(JT, Ni, k, C);
    rho    = comp_rho(JT, Ni, Weight);
    Ds     = proxy_grad(P, NewY, YM, Weight, rho, UbM);
    Xval(j,:) = Xval(j-1,:) - alpha/(1+alpha*lambda*(j-1))*Ds';
    Fval(:,j) = func(Xval(j,:));
end
end

function [ NewY, DeltaX ] = perturb(func, X, iter, beta, bs, n, k )

NewY   = zeros(k, bs);
DeltaX = zeros(bs, n);

for j = 1:bs
    DeltaX(j,:) = (2*rand(1,n) - 1) * beta * iter^(-1/3);
    NewY(:,j)   = func(X + DeltaX(j,:));
end

end

function [ JT ] = jacob_trans( P, NewY, YM )

JT = P * (NewY - YM)';

end

function [ Ds ] = proxy_grad( P, NewY, YM, Weight, rho, UbM )

DeltaS = (NewY - YM - rho*(max(NewY,UbM) - max(YM,UbM)))' * Weight;
Ds     = P*DeltaS;

end

function [ P ] = ortho_proj( DeltaX, lambda, tau )

W = diag(exp(-sum(DeltaX.^2,2) / 2 / tau^2));
P = pinv(DeltaX'*W*DeltaX + lambda*eye(size(DeltaX,2)))*DeltaX'*W;

end

function [ Weight ] = comp_weight( JT, Ni, k, C )

if any(Ni)
    JND    = JT(:,Ni);
    lp_M   = JND'*JT;
    lp_f   = [-1; zeros(k,1)];
    lp_A   = [ones(size(lp_M,1),1) -lp_M
              0 -ones(1,k) 
              1 zeros(1,k)];
    lp_b   = [zeros(size(lp_M,1)+1,1); C];
    options= optimset('Display','none');
    lp_x   = linprog(lp_f, lp_A, lp_b,[],[],[],[],[],options);
    if (size(lp_x,1) == k + 1)
        Weight = lp_x(2:end);
        Weight = Weight / norm(Weight);
    else
        Weight = rand(k,1);
        Weight = Weight / norm(Weight);
    end
else
    Weight = rand(k,1);
    Weight = Weight / norm(Weight);
end

end

function [ rho ] = comp_rho(JT, Ni, Weight)

Ri = Ni' & any(JT); % non-zero gradient and violated constraint
if any(Ri)
    JND = JT(:,Ni);
    JNN = JT(:,Ri);
    RM  = JNN'*JT;
    RMP = JNN'*JND;
    Td  = RM  * Weight;
    Tn  = RMP * Weight(Ni);
    %%% Case 1: rho >= 0
    [~, id] = min(Td-Tn);
    id = id(1);
    if (Tn(id) > 0)
        rho1 = 0;
    else
        rho1 = min( Td ./ (RM.*(RM>0)*Weight) );
    end
    imp1 = Td(id) - rho1*Tn(id);

    %%% Case 2: rho < 0
    [~, id] = min(Td+Tn);
    id = id(1);
    if (Tn(id) > 0)
        NegIdx   = any((RM<0),2);
        if any(NegIdx)
            RM   = RM(NegIdx,:);
            rho2 = max( Td(NegIdx) ./ (RM.*(RM<0)*Weight) );
        else
            rho2 = 0;
        end
    else
        rho2 = 0;
    end
    imp2 = Td(id) - rho2*Tn(id);

    if (imp1 > imp2)
        rho = rho1;
    else
        rho = rho2;
    end
    %%%%%%%%%
else
    rho = 0;
end        
end
