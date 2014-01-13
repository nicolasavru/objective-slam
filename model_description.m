function [ mapObj ] = model_description( model_points )
%model_description 
%   Detailed explanation goes here

d_dist = 0.05 * size(model_points,1);
d_angle = 2*pi / 30;

indeces = 1:size(model_points,1);
[p,q] = meshgrid(indeces, indeces);
index_pairs = [p(:) q(:)];

mapObj = containers.Map('KeyType', 'double');
Opt.Format = 'hex';
Opt.Method = 'SHA-1';

for ii = 1:size(index_pairs,1)
  
  % Handle case of identical point in pair
  if (index_pairs(ii,1) == index_pairs(ii,2)
    continue
  end
  
  F = point_pair_feature(model_points(index_pairs(ii,1),:), ...
                         model_points(index_pairs(ii,2),:));
  F_disc = floor([round(F(1)); F(2:4)*2*pi/d_angle]); % BIG COMMENT: Fix 
  % the discretization
  
  hash = DataHash(F_disc, Opt);
  key = hex2num(hash(1:16));
  
  if isKey(mapObj, key)
    entry = mapObj(key);
    entry(end+1,:) = [model_points(index_pairs(ii,1),:), ...
                      model_points(index_pairs(ii,2),:)];
  else
    mapObj(key) = [model_points(index_pairs(ii,1),:), ...
                   model_points(index_pairs(ii,2),:)];
  end
  
end

end

