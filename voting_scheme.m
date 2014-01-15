function [ max_tot, transform ] = voting_scheme( model_map, model_points, ...
                                          model_normals, scene_points, ...
                                          scene_normals )
%voting scheme
%   Detailed explanation goes here

d_dist = 0.05 * size(scene_points,1);
n_angle = 30;
d_angle = 2*pi / n_angle;

indeces = 1:size(scene_points,1);
[p,q] = meshgrid(indeces, indeces);
index_pairs = [p(:) q(:)];

Opt.Format = 'hex';
Opt.Method = 'SHA-1';

accumulator = zeros(size(model_points,1), n_angle);
transforms = zeros(size(model_points,1), n_angle, 4, 4);

for ii = 1:size(index_pairs,1)
  
  if mod(ii, 1000) == 0
    fprintf('On point %d of %d\n', ii, size(index_pairs,1));
  end
  % Handle case of identical point in pair
  if index_pairs(ii,1) == index_pairs(ii,2)
    continue
  end
  
  s_r = scene_points(index_pairs(ii,1),:);
  s_i = scene_points(index_pairs(ii,2),:);
  n_r_s = scene_normals(index_pairs(ii,1),:);
  
  F = point_pair_feature(s_r, n_r_s, s_i, scene_normals(index_pairs(ii,2),:));
  F_disc = floor([round(F(1)); F(2:4)*2*pi/d_angle]); % BIG COMMENT: Fix 
  % the discretization
  
  hash = DataHash(F_disc, Opt);
  key = hex2num(hash(1:16));
  
  matched_model_points = NaN;
  if isKey(model_map, key)
    % Model map returns N by 2 array where every row is a set of point pair
    % indeces within model_map
    matched_model_points = model_map(key);
    
    for jj = 1:size(matched_model_points,1)
      
      [T_m_g,T_s_g,alpha] = trans_model_scene( ...
                              model_points(matched_model_points(jj,1),:), ...
                              model_normals(matched_model_points(jj,1),:), ...
                              model_points(matched_model_points(jj,2),:), ...
                              s_r, n_r_s, s_i);
                            
      alpha_disc = round((n_angle-1)*alpha / (2*pi)) + 1;
      
      accumulator(matched_model_points(jj,1),alpha_disc) = ...
                    accumulator(matched_model_points(jj,1),alpha_disc) + 1;
      transforms(matched_model_points(jj,1),alpha_disc,:,:) = invht(T_s_g)*rotx(alpha)*T_m_g;
                        
    end
  end
end

[Y_rows, I_row] = max(accumulator);
[max_tot, I_col] = max(Y_rows);

transform = squeeze(transforms(I_row(I_col), I_col, :, :));

end