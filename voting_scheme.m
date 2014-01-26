function [ max_tot, transform1, transform2, transform3, max_tots, accumulator ] = voting_scheme( model_map, model_points, ...
                                          model_normals, scene_points, ...
                                          scene_normals, d_dist, ...
                                                      d_angle)
%voting scheme
%   Detailed explanation goes here

n_angle = round(2*pi / d_angle);
accum_thresh = 0.9;
skip = 5;

indices = 1:size(scene_points,1);
r_indices = 1:skip:size(scene_points,1);
[p,q] = meshgrid(r_indices, indices);
index_pairs = [p(:) q(:)];

Opt.Format = 'hex';
Opt.Method = 'SHA-1';

accumulator = zeros(size(model_points,1), n_angle, indices(end));
% transforms = zeros(size(model_points,1), n_angle, 4, 4);
transforms_Tmg = zeros(size(model_points,1), n_angle, 4, 4, indices(end));
transforms_Tsg = zeros(size(model_points,1), n_angle, 4, 4, indices(end));
transforms_alpha = zeros(size(model_points,1), n_angle, 4, 4, indices(end));

I_rows = zeros(indices(end), 1);
I_cols = zeros(indices(end), 1);
max_tots = zeros(indices(end), 1);
size(max_tots)

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

  F = real(point_pair_feature(s_r, n_r_s, s_i, scene_normals(index_pairs(ii,2),:)));
%   F_disc = [F(1)-mod(F(1),d_dist); F(2:4)-mod(F(2:4),d_angle)];
  F_disc = [quant(F(1),d_dist); quant(F(2:4),d_angle)];

  hash = DataHash(F_disc, Opt);
  key = hex2num(hash(1:16));

  if isKey(model_map, key)
    % Model map returns N by 2 array where every row is a set of point pair
    % indices within model_map
    matched_model_points = model_map(key);

    for jj = 1:size(matched_model_points,1)

      [T_m_g,T_s_g,alpha] = trans_model_scene( ...
                              model_points(matched_model_points(jj,1),:), ...
                              model_normals(matched_model_points(jj,1),:), ...
                              model_points(matched_model_points(jj,2),:), ...
                              s_r, n_r_s, s_i);
                            
%       alpha_disc = alpha-mod(alpha,d_angle);
      alpha = real(alpha);
      % alpha_disc = quant(alpha+pi,d_angle)+1;
      alpha_disc = alpha+pi-mod(alpha+pi,d_angle);
      alpha_ind = min(round(alpha_disc/d_angle)+1, n_angle); % rounding error?

      accumulator(matched_model_points(jj,1),alpha_ind, index_pairs(ii,1)) = ...
          accumulator(matched_model_points(jj, 1),alpha_ind, index_pairs(ii,1)) + 1;
%       transforms(matched_model_points(jj,1),alpha_ind,:,:) = invht(T_s_g)*rotx(alpha)*T_m_g;
      transforms_Tmg(matched_model_points(jj,1), ...
                     alpha_ind,:,:, index_pairs(ii,1)) = T_m_g;
      transforms_Tsg(matched_model_points(jj,1), ...
                     alpha_ind,:,:, index_pairs(ii,1)) = T_s_g;
      transforms_alpha(matched_model_points(jj, ...
                                            1),alpha_ind,:,:, index_pairs(ii,1)) = rotx(alpha);

    end
    [Y_rows, I_row] = max(accumulator(:,:,index_pairs(ii,1)));
    [max_tot, I_col] = max(Y_rows);

    I_rows(index_pairs(ii,1)) = I_row(I_col);
    I_cols(index_pairs(ii,1)) = I_col;
    max_tots(index_pairs(ii,1)) = max_tot;
  end
end

max_tot = max(max_tots);
max_tots = max_tots ./ max_tot;
max_ind = find(max_tots > accum_thresh);

ret_rows = I_rows(max_ind);
ret_cols = I_cols(max_ind);

max_tots
max_ind
ret_rows
ret_cols

% I_row = I_rows(max_ind);
% I_col = I_cols(max_ind);

% max_tots
% max_tot
% max_ind
% I_row
% I_col

% size(transforms_Tsg)
% size(transforms_alpha)
% size(transforms_Tmg)

% for ii = 1:size(transforms_Tmg, 3)
%     for jj = 1:size(transforms_Tmg, 1)
%         for kk = 1:size(transforms_Tmg, 2)
%             squeeze(transforms_Tmg(jj,kk, :, :, ii)
%         end
%     end
% end

% transform = squeeze(transforms(I_row(I_col), I_col, :, :));

transform1 = zeros(4,4,length(max_ind));
transform2 = zeros(4,4,length(max_ind));
transform3 = zeros(4,4,length(max_ind));

for ii = 1:length(max_ind)
    ret_row = ret_rows(ii);
    ret_col = ret_cols(ii);
    ind = max_ind(ii);

    transform1(:,:,ii) = squeeze(transforms_Tmg(ret_row, ret_col, :, :, ind));
    transform2(:,:,ii) = squeeze(transforms_Tsg(ret_row, ret_col, :, :, ind));
    transform3(:,:,ii) = squeeze(transforms_alpha(ret_row, ret_col, :, :, ind));

% transform1 = squeeze(transforms_Tmg(ret_rows, ret_cols, :, :, max_ind));
% transform2 = squeeze(transforms_Tsg(ret_rows, ret_cols, :, :, max_ind));
% transform3 = squeeze(transforms_alpha(ret_rows, ret_cols, :, :, max_ind));

end

% size(transform1)
% size(transform2)
% size(transform3)

end
