function [Hr, F, cluster] = NF_group_6(H_beam, K , rf_num , n ,d,r_circle_min, r_circle_max,fc,sigma_aod, L, kappa, w_near_reshape,max_K)
% 在NF_group_4的基础上，参照select.m函数考虑max_K



% H_mul_user_max_K = generate_user_in_line_multipath(n, d, r_circle_min, r_circle_max, fc, max_K , sigma_aod, L, kappa);
% H_to_group_6 = H_mul_user_max_K*conj(w_near_reshape); % max_K*Q
% H_to_group_6 = H_to_group_6.'; % Q*max_K
% 

H_mod = abs(H_beam).^2;
H_sum = sum(H_mod,1);
[~,orderr] = sort(H_sum,'descend');
H_beam = H_beam(:,orderr);
H_mod = abs(H_beam).^2;

Hr=[];
% index of beams
index=zeros(rf_num,1);
beam_num = 0;

for k=1:max_K
   [~,max_beam] = max(H_mod(:,k)); 
   if ~ismember(max_beam,index)
       beam_num = beam_num+1;
       index(beam_num) = max_beam;
       Hr = [Hr,H_beam(:,k)];
   end
   
   if beam_num == rf_num
       break;
   end
   
end

while beam_num < rf_num
%     h_beam = U*beamspace_channel(n,1,L,lambda,d,sigma_square_LOS,sigma_square_NLOS);
    H_mul_user = generate_user_in_line_multipath(n, d, r_circle_min, r_circle_max,fc,1,sigma_aod,L,kappa); % K*n(K=1)(1*n)
    h_beam = H_mul_user*conj(w_near_reshape); % K*Q(K=1)(1*Q)
    h_beam = h_beam.'; % Q*K(K=1)(Q*1)
    [~,max_beam] = max(abs(h_beam).^2);
    if ~ismember(max_beam,index)
        beam_num = beam_num+1;
        index(beam_num) = max_beam;
        Hr = [Hr,h_beam];
    end
end

interval = k;
index=sort(index,'ascend');


for k=interval+1:max_K
    [~,I]=max(H_mod(:,k));
    if ismember(I,index)
        Hr=[Hr,H_beam(:,k)];
    end
end

num_serve=size(Hr,2);

if num_serve >= K
    Hr=Hr(:,1:K);
else 
    while num_serve < K
%         h_beam = U*beamspace_channel(n,1,L,lambda,d,sigma_square_LOS,sigma_square_NLOS);
        H_mul_user = generate_user_in_line_multipath(n, d, r_circle_min, r_circle_max,fc,1,sigma_aod,L,kappa); % K*n(K=1)(1*n)
        h_beam = H_mul_user*conj(w_near_reshape); % K*Q(K=1)(1*Q)
        h_beam = h_beam.'; % Q*K(K=1)(Q*1)
        [~,max_index]=max(abs(h_beam).^2);
        if ismember(max_index,index)
            Hr = [Hr,h_beam];
            num_serve = num_serve+1;
        end
    end
end

Hr=Hr(index,:); %(rf_num*K)

cluster= zeros (rf_num,K); % This cluster records the users in every beam in descending order of channel gain 
H_reduce = zeros(rf_num,rf_num); % This matrix is for zero-forcing
setr = zeros(1,rf_num); % 强用户的集合


Hr_mod = abs(Hr).^2;%(rf_num*K)
set = Hr_mod*0;%(rf_num*K)

for k = 1:K
   [~,I]=max(Hr_mod(:,k));
   set(I,k)=1;
end 

for i=1:rf_num
   %find each user in a certain beam 
   temp = Hr(:,set(i,:)>0);
   temp_sum = sum(abs(temp).^2,1);
   [~,order] = sort(temp_sum,'descend');
   temp1 = find(set(i,:)>0);
   temp1 = temp1(order);
   setr(1,i) = temp1(1);
   for j = 1:length(temp1)
       cluster(i,j) = temp1(j);    
   end 
   H_reduce(:,i) = Hr(:,temp1(1));    
end

% 参考Bichai_Wang的reduce_RF写预编码矩阵F

F = zeros(rf_num,K); %(rf_num*K)
Ft = H_reduce*inv(H_reduce'*H_reduce);  % ZF precoding
F(:,setr) = Ft;
%.................A....................
for i=1:rf_num
   temp = Hr(:,set(i,:)>0);
   temp_sum = sum(abs(temp).^2,1);
   [~,order] = sort(temp_sum,'descend');
   temp1 = find(set(i,:)>0);
   temp1 = temp1(order);
   for k = 2:length(order)
   F(:,temp1(k)) = F(:,temp1(1));
   end
end
%.................A....................
% 效果不好可以尝试去掉A这一段
