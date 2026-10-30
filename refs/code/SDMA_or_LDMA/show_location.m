function [] = show_location(r0,theta_list)
%SHOW_LOCATION 此处显示有关此函数的摘要
%   此处显示详细说明
    x_list = r0.*cos(theta_list);
    y_list = r0.*sin(theta_list);
    sz = 25;
    c = linspace(1,10,length(x_list));
    scatter(x_list,y_list,sz,c,'filled')
%     scatter(x_list, y_list);
    axis equal;
    box on;
    grid on;
end

