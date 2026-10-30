function [Hr,H_reduce,H_se,cluster,index]=select(H_beam,rf_num,n,L,lambda,d,sigma_square_LOS,sigma_square_NLOS,K,max_K,U)
% In this file, we will choose rf_num largest beams
% H_mod=abs(H_beam).^2;
% H_sum=sum(H_mod,1);
% [~,order]=sort(H_sum,'descend');
% H_beam=H_beam(:,order);
H_mod=abs(H_beam).^2;

Hr=[];
% index of beams
index=zeros(rf_num,1);
beam_num=0;

for k=1:max_K
   [~,max_beam]=max(H_mod(:,k)); 
   if ~ismember(max_beam,index)
       beam_num=beam_num+1;
       index(beam_num)=max_beam;
       Hr=[Hr,H_beam(:,k)];
   end
   
   if beam_num==rf_num
       break;
   end
   
end

while beam_num<rf_num
    h_beam = U*beamspace_channel(n,1,L,lambda,d,sigma_square_LOS,sigma_square_NLOS);
    [~,max_beam]=max(abs(h_beam).^2);
    if ~ismember(max_beam,index)
        beam_num=beam_num+1;
        index(beam_num)=max_beam;
        Hr=[Hr,h_beam];
    end
end

interval=k;
index=sort(index,'ascend');



for k=interval+1:max_K
    [~,I]=max(H_mod(:,k));
    if ismember(I,index)
        Hr=[Hr,H_beam(:,k)];
    end
end

num_serve=size(Hr,2);

if num_serve>=K
    Hr=Hr(:,1:K);
else 
    while num_serve<K
        h_beam = U*beamspace_channel(n,1,L,lambda,d,sigma_square_LOS,sigma_square_NLOS);
        [~,max_index]=max(abs(h_beam).^2);
        if ismember(max_index,index)
            Hr=[Hr,h_beam];
            num_serve=num_serve+1;
        end
    end
end
H_se=Hr; 
Hr=Hr(index,:);

cluster=zeros(rf_num,K); % This cluster records the users in every beam in descending order of channel gain 
H_reduce=zeros(rf_num,rf_num); % This matrix is for zero-forcing

Hr_mod=abs(Hr).^2;
set = Hr_mod*0;

for k = 1:K
   [~,I]=max(Hr_mod(:,k));
   set(I,k)=1;
end 

for i=1:rf_num
   %find each user in a certain beam 
   temp=Hr(:,set(i,:)>0) ;
   
   
   temp_sum=sum(abs(temp).^2,1);
   [~,order]=sort(temp_sum,'descend');
   temp1=find(set(i,:)>0);
   temp1=temp1(order);
   
   for j=1:length(temp1)
     
       cluster(i,j)=temp1(j);    
   end
   
   H_reduce(:,i)=Hr(:,temp1(1));
    
end
