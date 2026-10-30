function W=cvx_solve_W_for_noRIS(M,K,G,Theta,V,A,W,Ps_max)

temp=(G'*Theta)*(Theta'*G);

D=kron(eye(K),temp);
D=0.5*(D+D');

A=0.5*(A+A');

A=A+10^(-50);

cvx_begin quiet
    variable W(M*K,1) complex;
    minimize((W')*A*W-2*real((V')*W))
    subject to
    W'*W<=Ps_max;

cvx_end

end