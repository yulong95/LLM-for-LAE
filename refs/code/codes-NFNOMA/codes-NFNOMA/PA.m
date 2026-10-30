function [SE,EE,ite,power] = PA(H,Fi,setf,K,rf_num,sigma2,P,Imax)


%% 功率初始化

power0 = zeros(rf_num,K);
for n = 1:rf_num
    usern = setf(n,:);  % 第n个beam的用户
    usern(usern==0) = [];
    for m = 1:length(usern)
        power0(n,m) = m*(norm(Fi(:,usern(m))))^2;
    end
end
power0 = power0*(P/sum(sum(power0)));

p0 = [];
for n = 1:rf_num
    usern = setf(n,:);  % 第n个beam的用户
    usern(usern==0) = [];
    for m = 1:length(usern)
        p0 = [p0;power0(n,m)];
    end
end

%% 等效信道

for k = 1:K
    Fi(:,k) = Fi(:,k)/norm(Fi(:,k));
end
H_eq = H'*Fi;

%% 迭代
  
ite = 1:Imax;
SE = zeros(1,Imax);
EE = zeros(1,Imax);

for it = 1:Imax

    A = zeros(rf_num,K);
    C = zeros(rf_num,K);  % MMSE均衡系数
    E = zeros(rf_num,K);  % MSE

    x1 = zeros(rf_num,K);
    x2 = zeros(rf_num,K);
    
    temp = 0;

    for n = 1:rf_num
        usern = setf(n,:);  % 第n个beam的用户
        usern(usern==0) = [];
        for m = 1:length(usern)
            bet = (norm(H_eq(usern(m),usern(m))))^2*sum(power0(n,1:m-1))+sigma2; % 干扰加噪声项
            for j = 1:rf_num
                if j~=n
                    bet = bet+(norm(H_eq(usern(m),setf(j,1))))^2*sum(power0(j,:));
                end
            end
            C(n,m) = (sqrt(power0(n,m))*H_eq(usern(m),usern(m)))'/(power0(n,m)*(norm(H_eq(usern(m),usern(m))))^2+bet);
            E(n,m) = 1-power0(n,m)*(norm(H_eq(usern(m),usern(m))))^2/(power0(n,m)*(norm(H_eq(usern(m),usern(m))))^2+bet);
            A(n,m) = 1/E(n,m);
        end
    end
    
    
    A0 = ones(1,K);
    b0 = P;
    Rmin = 0.01;
    ita = 2^Rmin-1;
    %Ptol = P/K/10; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ind = zeros(1,rf_num);
    usel = 0;
    for n = 1:rf_num
        usern = setf(n,:);  % 第n个beam的用户
        usern(usern==0) = [];
        ind(n) = usel+length(usern);
        usel = ind(n);
    end
    ind = [0,ind];
    for n = 1:rf_num
        usern = setf(n,:);  % 第n个beam的用户
        usern(usern==0) = [];
        for m = 1:length(usern)
            a0 = zeros(1,K);
            a0(ind(n)+m) = abs(H_eq(usern(m),usern(m)))^2;
            a0(ind(n)+1:ind(n)+m-1) = -ita*abs(H_eq(usern(m),usern(m)))^2;
            for j = 1:rf_num
                if j ~=  n
                    userj = setf(j,:);  % 第n个beam的用户
                    userj(userj==0) = [];
                    a0(ind(j)+1:ind(j+1)) = -ita*abs(H_eq(usern(m),userj(1)))^2;
                end
            end
            A0 = [A0;-a0];
            b0 = [b0;-ita*sigma2];           
        end
    end

    lb = zeros(K,1);
    ub = zeros(K,1);
    ub(1:K) = P;
    options = optimset('LargeScale','off','display','iter');

    [power,sume] = fmincon(@(power)(myfun(power,H,Fi,A,C,setf,K,rf_num,sigma2)),p0,A0,b0,[],[],lb,ub,[],options);
    
    rmax = -sume;

    for n = 1:rf_num
        usern = setf(n,:);  % 第n个beam的用户
        usern(usern==0) = [];
        power0(n,1:length(usern)) = power(ind(n)+1:ind(n+1));
    end

    p0 = [];
    for n = 1:rf_num
        usern = setf(n,:);  % 第n个beam的用户
        usern(usern==0) = [];
        for m = 1:length(usern)
            p0 = [p0;power0(n,m)];
        end
    end
    
    SE(it) = rmax;
    EE(it) = rmax/(P+rf_num*305+200)*10^3;

end
