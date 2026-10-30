function sume = myfun_NF(power,H,Fi,A,C,setf,K,rf_num,sigma2)

H_eq = H'*Fi;

ind = zeros(1,rf_num);
usel = 0;
for n = 1:rf_num
    usern = setf(n,:);  % 第n个beam的用户
    usern(usern==0) = [];
    ind(n) = usel+length(usern);
    usel = ind(n);
end
ind = [0,ind];

bet = zeros(rf_num,K);
E = zeros(rf_num,K);
for n = 1:rf_num
    usern = setf(n,:);  % 第n个beam的用户
    usern(usern==0) = [];
    for m = 1:length(usern)
        bet(n,m) = (norm(H_eq(usern(m),usern(m))))^2*sum(power(ind(n)+1:ind(n)+m-1))+sigma2; % 干扰加噪声项
        for j = 1:rf_num
            if j~=n
                bet(n,m) = bet(n,m)+(norm(H_eq(usern(m),setf(j,1))))^2*sum(power(ind(j)+1:ind(j+1)));
            end
        end
        E(n,m) = (abs(1-C(n,m)*sqrt(power(ind(n)+m))*H_eq(usern(m),usern(m))))^2+abs(C(n,m))^2*bet(n,m);
    end
end

sume = 0;
for n = 1:rf_num
    usern = setf(n,:);  % 第n个beam的用户
    usern(usern==0) = [];
    for m = 1:length(usern)
        sume = sume+A(n,m)*E(n,m)/log(2)-log2(A(n,m))-1/log(2);
    end
end
        

