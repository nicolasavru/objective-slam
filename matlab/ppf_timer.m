N = 256;
F = zeros(N*N, 4);

[x y] = cuda_float3_c_write('/tmp', N);

tStart=tic;
for ii=1:N
    for jj =1:N
    F((ii-1)*N+jj,:) = point_pair_feature(x(:,ii),y(:,ii),x(:,jj),y(:,jj)).';
    %F((ii-1)*N+jj,:)
    end
end
tElapsed=toc(tStart)