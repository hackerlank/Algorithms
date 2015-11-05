function [ Fval, Xval ] = min_pald(func, X0, niter, bs, beta, alpha, lambda, tau, R)
%  PAreto Local Descent (PALD) Algorithm
%  @func    -- the multi-objective function to optimize
%  @X0      -- initial point
%  @niter   -- number of iterations
%  @bs      -- minibatch size
%  @beta    -- perturbation scaling factor
%  @alpha   -- step size
%  @lambda  -- regularization factor
%  @tau     -- bandwidth
%  @R       -- constraint, or the preference point

Fval0      = func(X0);                 % Objective at the initial point                  
decValNum  = length(X0);               % Number of decision variables
objNum     = length(Fval0);            % Number of objective functions
Fval       = zeros(objNum, niter);     % Objective matrix
Xval       = zeros(niter, decValNum);  % Decision variable matrix
DeltaX     = zeros(bs, decValNum);     % Perturbation matrix
DeltaS     = zeros(bs, 1);             % Changes in the proxy model objective
DELTA_Y    = zeros(objNum, bs);        % Changes in the objectives
FvalBatch  = zeros(objNum, bs);        % Objective values for the batch
C          = 1;                        % z limit

Xval(1,:) = X0;
Fval(:,1) = Fval0;

for t = 2:niter
    for j = 1:bs
        pertRange      = (t-1)^(-1/3);
        DeltaX(j,:)    = beta * ( 2*rand(1,decValNum) - 1) * pertRange;
        FvalBatch(:,j) = func( Xval(t-1,:)+DeltaX(j,:) );
        DELTA_Y(:,j)   = FvalBatch(:,j) - Fval(:,t-1);
    end
    D = diag(exp(-sum(DeltaX.^2, 2) / 2 / tau^2));
    M = pinv(DeltaX'*D*DeltaX + lambda*eye(decValNum))*DeltaX'*D*DELTA_Y';   % \nabla_x func, Jacobian transpose
    
    %%% Solve for the Weight %%%
    Ni = Fval(:,t-1) >= R;
    if any(Ni)
        JND    = M(:,Ni);
        lp_M   = JND'*M;  % f_i x f_j, i:f_i > r_i
        lp_f   = [-1; zeros(objNum,1)];
        lp_A   = [ones(size(lp_M,1),1) -lp_M
                  0 -ones(1,objNum) 
                  1 zeros(1,objNum)];
        lp_b   = [zeros(size(lp_M,1)+1,1); C];
        options= optimset('Display','none');
        lp_x   = linprog(lp_f, lp_A, lp_b,[],[],[],[],[],options);
        if (size(lp_x,1) == objNum + 1)
            Weight = lp_x(2:end);
            Weight = Weight / norm(Weight);
        end
        % y  = lp_M * Weight; % correctness check
    else
        Weight = rand(objNum,1);
        Weight = Weight / norm(Weight);
    end
    
    if ~exist('Weight','var') || size(Weight,1) ~= objNum
        Weight = rand(objNum,1);
        Weight = Weight / norm(Weight);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Solve for the rho %%%
    RhoIdx = Ni' & any(M); % non-zero gradient and violated constraint
    if any(RhoIdx)
        JNN = M(:,RhoIdx);
        RM  = JNN'*M;
        RMP = JNN'*JND; % JND must have been initialized
        td  = RM  * Weight;
        tn  = RMP * Weight(Ni);
        %%% Case 1: rho >= 0
        [~, id] = min(td-tn);
        id = id(1);
        if (tn(id) > 0)
            rho1 = 0;
        else
            rho1 = min( td ./ (RM.*(RM>0)*Weight) );
        end
        imp1 = td(id) - rho1*tn(id);
        
        %%% Case 2: rho < 0
        [~, id] = min(td+tn);
        id = id(1);
        if (tn(id) > 0)
            NegIdx   = any((RM<0),2);
            if any(NegIdx)
                RM   = RM(NegIdx,:);
                rho2 = max( td(NegIdx) ./ (RM.*(RM<0)*Weight) );
            else
                rho2 = 0;
            end
        else
            rho2 = 0;
        end
        imp2 = td(id) - rho2*tn(id);
        
        if (imp1 > imp2)
            rho = rho1;
        else
            rho = rho2;
        end
        %%%%%%%%%
    else
        rho = 0;
    end        
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    for j = 1:bs
        tp   = FvalBatch(:,j) - rho*max(FvalBatch(:,j),  R);
        tm   = Fval(:,t-1) - rho*max(Fval(:,t-1),R);
        DeltaS(j) = Weight'*(tp-tm);
    end
    
    g = pinv(DeltaX'*D*DeltaX + lambda*eye(decValNum))*DeltaX'*D*DeltaS;
    Xval(t,:) = Xval(t-1,:) - alpha * g'/(1+alpha*lambda*(t-1));
    Fval(:,t) = func(Xval(t,:));
end
