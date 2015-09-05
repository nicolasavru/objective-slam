function [ output_args ] = compare_data_with_pcl(matlab_data, pcl_filename, size_of_feature)
% size_of_feature is the number of numbers in the feature. For example, it
% is 1 for floats, 4 for point-pair features (float4s), etc.
D = matlab_data(:);
num_features = sqrt(length(D) / size_of_feature);

fid = fopen(pcl_filename,'rb');
A = fread(fid,[size_of_feature*num_features, num_features],'float');
fclose(fid);
B = A(:);

figure, plot(abs(B-D))
end

