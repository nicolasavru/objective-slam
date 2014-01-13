function [ output_args ] = voting_scheme( model_map, scene_points )
%voting scheme
%   Detailed explanation goes here

d_dist = 0.05 * size(scene_points,1);
d_angle = 2*pi / 30;

indeces = 1:size(scene_points,1);
[p,q] = meshgrid(indeces, indeces);
index_pairs = [p(:) q(:)];

for ii = 1:size(index_pairs,1)
  
  % Handle case of identical point in pair
  if (index_pairs(ii,1) == index_pairs(ii,2)
    continue
  end
  
  F = point_pair_feature(scene_points(index_pairs(ii,1),:), ...
                         scene_points(index_pairs(ii,2),:));
  F_disc = floor([round(F(1); F(2:4)*2*pi/d_angle]); % BIG COMMENT: Fix 
  % the discretization
  
  hash = DataHash(F_disc, Opt);
  key = hex2num(hash(1:16));
  
  matched_model_points = NaN;
  if isKey(model_map, key)
    matched_model_points = model_map(key);
  end
  
  % INCOMPLETE LEFT OFF AT VOTING SCHEME SECTION ON PAGE 1001 OF DROST
  
end

end