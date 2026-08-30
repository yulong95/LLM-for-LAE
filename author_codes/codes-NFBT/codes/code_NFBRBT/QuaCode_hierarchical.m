function [dict, theta_record_list_hierarchical_1,r_record_list_hierarchical_1] = QuaCode_hierarchical(N, d, lambda, P,sampling_interval)
    c = 3e8;
    theta_max = P(1);theta_min = P(2);r_max = P(3);r_min = P(4);
    theta_interval = sampling_interval(1);r_interval = sampling_interval(2);
    theta_list = theta_min:theta_interval:theta_max;r_list = r_min:r_interval:r_max;
    dict = zeros(N,length(theta_list)*length(r_list));
    theta_record_list_hierarchical_1 = zeros(1,length(theta_list)*length(r_list));
    r_record_list_hierarchical_1 = zeros(1,length(theta_list)*length(r_list));
    cnt = 1;
    for i_theta = 1:length(theta_list)
      for i_r = 1:length(r_list)
        theta = theta_list(i_theta);
        r = r_list(i_r);
        at = polar_domain_manifold( N, d, c/lambda, r, asin(theta) );
        dict(:,cnt) = at;
        theta_record_list_hierarchical_1(cnt) = theta;
        r_record_list_hierarchical_1(cnt) = r;
        cnt = cnt + 1;
      end
    end
    dict = dict(:,1:cnt-1);
    theta_record_list_hierarchical_1 = theta_record_list_hierarchical_1(:,1:cnt-1);
    r_record_list_hierarchical_1 = r_record_list_hierarchical_1(:,1:cnt-1);
end