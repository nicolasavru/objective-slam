function [ max_tot, transform1, transform2, transform3 ] = voting_scheme( model_map, model_points, ...
                                          model_normals, scene_points, ...
                                          scene_normals, d_dist, d_angle )
%voting scheme
%   Detailed explanation goes here

n_angle = round(2*pi / d_angle);

indeces = 1:size(scene_points,1);
[p,q] = meshgrid(indeces, indeces);
index_pairs = [p(:) q(:)];

Opt.Format = 'hex';
Opt.Method = 'SHA-1';

accumulator = zeros(size(model_points,1), n_angle, indeces(end));
% transforms = zeros(size(model_points,1), n_angle, 4, 4);
transforms_Tmg = zeros(size(model_points,1), n_angle, 4, 4, indeces(end));
transforms_Tsg = zeros(size(model_points,1), n_angle, 4, 4, indeces(end));
transforms_alpha = zeros(size(model_points,1), n_angle, 4, 4, indeces(end));

I_rows = zeros(indeces(end), 1);
I_cols = zeros(indeces(end), 1);
max_tots = zeros(indeces(end), 1);

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
    % indeces within model_map
    matched_model_points = model_map(key);

    for jj = 1:size(matched_model_points,1)

      [T_m_g,T_s_g,alpha] = trans_model_scene( ...
                              model_points(matched_model_points(jj,1),:), ...
                              model_normals(matched_model_points(jj,1),:), ...
                              model_points(matched_model_points(jj,2),:), ...
                              s_r, n_r_s, s_i);
                            
%       alpha_disc = alpha-mod(alpha,d_angle);
      alpha = real(alpha);
      alpha_disc = quant(alpha,d_angle)+1;
      alpha_ind = round(alpha_disc/d_angle);

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

    % I_row
    % I_col
    % size(I_rows)
    % size(I_row(I_col))

    I_rows(ii) = I_row(I_col);
    I_cols(ii) = I_col;
    max_tots(ii) = max_tot;
  end
end

[max_tot, max_ind] = max(max_tots);
I_row = I_rows(max_ind)
I_col = I_cols(max_ind)

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
transform1 = squeeze(transforms_Tmg(I_row, I_col, :, :, :));
transform2 = squeeze(transforms_Tsg(I_row, I_col, :, :, :));
transform3 = squeeze(transforms_alpha(I_row, I_col, :, :, :));

% size(transform1)
% size(transform2)
% size(transform3)

end
