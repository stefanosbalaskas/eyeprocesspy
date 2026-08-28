data {
  int<lower=1> N;
  int<lower=2> P;
  int<lower=2> I;
  array[N] int<lower=1,upper=P> person;
  array[N] int<lower=1,upper=I> item;
  array[N] int<lower=0,upper=1> y;
  vector[N] log_rt;
  array[N] int<lower=0> gaze;
}
parameters {
  matrix[P,3] z_person;
  vector<lower=0>[3] sd_person;
  cholesky_factor_corr[3] L_person;

  matrix[I,3] z_item;
  vector[3] mu_item;
  vector<lower=0>[3] sd_item;
  cholesky_factor_corr[3] L_item;

  real<lower=0> sigma_rt;
  real<lower=0> gaze_size;
}
transformed parameters {
  matrix[P,3] person_eff =
    z_person * diag_pre_multiply(sd_person, L_person)';
  matrix[I,3] item_eff =
    rep_matrix(mu_item', I) +
    z_item * diag_pre_multiply(sd_item, L_item)';
}
model {
  to_vector(z_person) ~ std_normal();
  sd_person ~ normal(0, 1);
  L_person ~ lkj_corr_cholesky(2);

  to_vector(z_item) ~ std_normal();
  mu_item ~ normal(0, 1);
  sd_item ~ normal(0, 1);
  L_item ~ lkj_corr_cholesky(2);

  sigma_rt ~ normal(0, .5);
  gaze_size ~ gamma(2, .1);

  for (n in 1:N) {
    int p = person[n];
    int i = item[n];
    // Rasch response: theta_p - difficulty_i
    y[n] ~ bernoulli_logit(person_eff[p,1] - item_eff[i,1]);

    // Lognormal RT location: time-intensity_i - speed_p
    log_rt[n] ~ normal(item_eff[i,2] - person_eff[p,2], sigma_rt);

    // Negative-binomial gaze count with log mean intensity_i + process_p.
    gaze[n] ~ neg_binomial_2_log(item_eff[i,3] + person_eff[p,3], gaze_size);
  }
}
generated quantities {
  corr_matrix[3] person_cor = multiply_lower_tri_self_transpose(L_person);
  corr_matrix[3] item_cor = multiply_lower_tri_self_transpose(L_item);
  real mean_y_rep;
  real mean_log_rt_rep;
  real mean_gaze_rep;
  {
    vector[N] y_r;
    vector[N] rt_r;
    vector[N] g_r;
    for (n in 1:N) {
      int p = person[n];
      int i = item[n];
      y_r[n] = bernoulli_logit_rng(person_eff[p,1] - item_eff[i,1]);
      rt_r[n] = normal_rng(item_eff[i,2] - person_eff[p,2], sigma_rt);
      g_r[n] = neg_binomial_2_log_rng(item_eff[i,3] + person_eff[p,3], gaze_size);
    }
    mean_y_rep = mean(y_r);
    mean_log_rt_rep = mean(rt_r);
    mean_gaze_rep = mean(g_r);
  }
}
