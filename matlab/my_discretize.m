function [ F_disc ] = my_discretize( F, d_dist, d_angle )
   
   F_disc = [F(1)-mod(F(1),d_dist);...
             F(2:4)-mod(F(2:4),d_angle)];
%   F_disc = [round(F(1)/d_dist)*d_dist; round(F(2:4)/d_angle)*d_angle];
%  F_disc = [quant(F(1),d_dist); quant(F(2:4),d_angle)];


end

