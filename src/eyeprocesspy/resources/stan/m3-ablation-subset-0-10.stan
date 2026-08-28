data {
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1, upper=4> K;

  int<lower=0, upper=1> use_rt;
  int<lower=0, upper=1> use_gaze;
  int<lower=0, upper=1> use_pupil;
  int<lower=0, upper=K> idx_rt;
  int<lower=0, upper=K> idx_gaze;
  int<lower=0, upper=K> idx_pupil;

  int<lower=1> N_response;
  array[N_response] int<lower=1, upper=J> person_response;
  array[N_response] int<lower=1, upper=I> item_response;
  array[N_response] int<lower=0, upper=1> y_response;

  int<lower=0> N_rt;
  array[N_rt] int<lower=1, upper=J> person_rt;
  array[N_rt] int<lower=1, upper=I> item_rt;
  vector[N_rt] log_rt;

  int<lower=0> N_gaze;
  array[N_gaze] int<lower=1, upper=J> person_gaze;
  array[N_gaze] int<lower=1, upper=I> item_gaze;
  array[N_gaze] int<lower=0> gaze;

  int<lower=0> N_pupil;
  array[N_pupil] int<lower=1, upper=J> person_pupil;
  array[N_pupil] int<lower=1, upper=I> item_pupil;
  vector[N_pupil] pupil;
  matrix[N_pupil, 8] X_pupil;
  array[8] int<lower=0, upper=1> use_pupil_covariate;

  int<lower=1, upper=2> prior_profile;
}

parameters {
  matrix[K, J] z_person;
  vector<lower=0>[K] sigma_person;
  cholesky_factor_corr[K] L_person;

  matrix[K, I] z_item;
  vector[K] mu_item;
  vector<lower=0>[K] sigma_item;
  cholesky_factor_corr[K] L_item;

  vector<lower=0>[I] nu;
  vector<lower=0>[I] s;
  real<lower=0> sigma_pupil;
  vector[8] gamma_pupil;
}

transformed parameters {
  matrix[K, J] person_eff =
    diag_pre_multiply(sigma_person, L_person) * z_person;
  matrix[K, I] item_eff =
    rep_matrix(mu_item, I) +
    diag_pre_multiply(sigma_item, L_item) * z_item;

  vector[J] theta = (person_eff[1])';
  vector[I] b = (item_eff[1])';
  matrix[K, K] corr_person = multiply_lower_tri_self_transpose(L_person);
  matrix[K, K] corr_item = multiply_lower_tri_self_transpose(L_item);
}

model {
  to_vector(z_person) ~ std_normal();
  to_vector(z_item) ~ std_normal();

  if (prior_profile == 1) {
    sigma_person ~ normal(0, 1);
    sigma_item ~ normal(0, 1);
    L_person ~ lkj_corr_cholesky(2);
    L_item ~ lkj_corr_cholesky(2);
    mu_item[1] ~ normal(0, 1);
    if (use_rt == 1) mu_item[idx_rt] ~ normal(4, 1);
    if (use_gaze == 1) mu_item[idx_gaze] ~ normal(3.5, 1.5);
    if (use_pupil == 1) mu_item[idx_pupil] ~ normal(0, 0.75);
    nu ~ lognormal(log(1.5), 0.5);
    s ~ lognormal(log(6), 0.7);
    sigma_pupil ~ normal(0, 0.75);
    gamma_pupil ~ normal(0, 0.5);
  } else {
    sigma_person ~ normal(0, 2);
    sigma_item ~ normal(0, 2);
    L_person ~ lkj_corr_cholesky(1);
    L_item ~ lkj_corr_cholesky(1);
    mu_item[1] ~ normal(0, 0.5);
    if (use_rt == 1) mu_item[idx_rt] ~ normal(4, 0.5);
    if (use_gaze == 1) mu_item[idx_gaze] ~ normal(3.5, 1);
    if (use_pupil == 1) mu_item[idx_pupil] ~ normal(0, 1);
    nu ~ gamma(1, 1);
    s ~ inv_gamma(1, 1);
    sigma_pupil ~ normal(0, 1);
    gamma_pupil ~ normal(0, 1);
  }

  for (n in 1:N_response) {
    y_response[n] ~ bernoulli_logit(theta[person_response[n]] - b[item_response[n]]);
  }

  if (use_rt == 1) {
    for (n in 1:N_rt) {
      log_rt[n] ~ normal(
        item_eff[idx_rt, item_rt[n]] - person_eff[idx_rt, person_rt[n]],
        1 / nu[item_rt[n]]
      );
    }
  }

  if (use_gaze == 1) {
    for (n in 1:N_gaze) {
      gaze[n] ~ neg_binomial_2_log(
        item_eff[idx_gaze, item_gaze[n]] + person_eff[idx_gaze, person_gaze[n]],
        s[item_gaze[n]]
      );
    }
  }

  if (use_pupil == 1) {
    for (n in 1:N_pupil) {
      real nuisance = 0;
      for (k in 1:8) {
        nuisance += use_pupil_covariate[k] * gamma_pupil[k] * X_pupil[n, k];
      }
      pupil[n] ~ normal(
        item_eff[idx_pupil, item_pupil[n]] + person_eff[idx_pupil, person_pupil[n]] + nuisance,
        sigma_pupil
      );
    }
  }
}

generated quantities {
  vector[N_response] log_lik_response;
  array[N_response] int y_rep;

  for (n in 1:N_response) {
    real eta = theta[person_response[n]] - b[item_response[n]];
    log_lik_response[n] = bernoulli_logit_lpmf(y_response[n] | eta);
    y_rep[n] = bernoulli_logit_rng(eta);
  }
}
