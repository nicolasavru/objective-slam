function [ F ] = point_pair_feature( m_1, n_1, m_2, n_2 )
%point_pair_feature Computes feature vector for point pair model
    d = m_2 - m_1;
    F1 = norm(d);
    F2 = acos(dot(n_1,d) / (norm(n_1)*norm(d)));
    F3 = acos(dot(n_2,d) / (norm(n_2)*norm(d)));
    F4 = acos(dot(n_1,n_2) / (norm(n_1)*norm(n_2)));

    F = [F1; F2; F3; F4];

end

