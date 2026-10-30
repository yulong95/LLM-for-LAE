function [w] = Code3D_uniform(Nt1, Nt2, d, lambda, x_list, y_list, z_list)

c = 3e8;
fc = c/lambda;

w = zeros(Nt1*Nt2, length(x_list)*length(y_list)*length(z_list));
cnt = 1;
for i_x = 1:length(x_list)
    x_grid = x_list(i_x);
    for i_y = 1:length(y_list)
        y_grid = y_list(i_y);
        for i_z = 1:length(z_list)
            z_grid = z_list(i_z);
            [phi,theta,r] = cart2sph(x_grid,y_grid,z_grid);
            [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, fc, r, pi/2-theta, phi);
            w(:, cnt) = at_wo;
            cnt = cnt+1;
        end
    end
end
w = w(:,1:cnt-1);

end

