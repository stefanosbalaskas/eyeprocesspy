data {
  int<lower=1> J;
  int<lower=1> I;

  int<lower=1> N_response;
  array[N_response] int<lower=1, upper=J> person_response;
  array[N_response] int<lower=1, upper=I> item_response;
  array[N_response] int<lower=0, upper=1> y_response;

  int<lower=1> N_rt;
  array[N_rt] int<lower=1, upper=J> person_rt;
  array[N_rt] int<lower=1, upper=I> item_rt;
  vector[N_rt] log_rt;

  int<lower=1, upper=2> prior_profile;
}

parameters {
  matrix[2, J] z_person;
  vector<lower=0>[2] sigma_person;
  cholesky_factor_corr[2] L_person;

  matrix[2, I] z_item;
  vector[2] mu_item;
  vector<lower=0>[2] sigma_item;
  cholesky_factor_corr[2] L_item;

  vector<lower=0>[I] nu;
}

transformed parameters {
  matrix[2, J] person_eff =
    diag_pre_multiply(sigma_person, L_person) * z_person;

  matrix[2, I] item_eff =
    rep_matrix(mu_item, I) +
    diag_pre_multiply(sigma_item, L_item) * z_item;

  vector[J] theta = (person_eff[1])';
  vector[J] tau = (person_eff[2])';

  vector[I] b = (item_eff[1])';
  vector[I] beta = (item_eff[2])';

  matrix[2, 2] corr_person =
    multiply_lower_tri_self_transpose(L_person);

  matrix[2, 2] corr_item =
    multiply_lower_tri_self_transpose(L_item);
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
    mu_item[2] ~ normal(4, 1);

    nu ~ lognormal(log(1.5), 0.5);
  } else {
    sigma_person ~ normal(0, 2);
    sigma_item ~ normal(0, 2);
    L_person ~ lkj_corr_cholesky(1);
    L_item ~ lkj_corr_cholesky(1);

    mu_item[1] ~ normal(0, 0.5);
    mu_item[2] ~ normal(4, 0.5);

    nu ~ gamma(1, 1);
  }

  for (n in 1:N_response) {
    y_response[n] ~ bernoulli_logit(
      theta[person_response[n]] - b[item_response[n]]
    );
  }

  for (n in 1:N_rt) {
    log_rt[n] ~ normal(
      beta[item_rt[n]] - tau[person_rt[n]],
      1 / nu[item_rt[n]]
    );
  }
}

generated quantities {
  vector[N_response] log_lik_response;
  vector[N_rt] log_lik_rt;

  array[N_response] int y_rep;
  vector[N_rt] log_rt_rep;

  vector[I] W_obs = rep_vector(0.0, I);
  vector[I] W_rep = rep_vector(0.0, I);
  vector[I] L_obs = rep_vector(0.0, I);
  vector[I] L_rep = rep_vector(0.0, I);

  for (n in 1:N_response) {
    int j = person_response[n];
    int i = item_response[n];
    real eta = theta[j] - b[i];
    real p = inv_logit(eta);
    real den = p * (1 - p) + 1e-9;

    log_lik_response[n] = bernoulli_logit_lpmf(y_response[n] | eta);
    y_rep[n] = bernoulli_rng(p);

    W_obs[i] += square(y_response[n] - p) / den;
    W_rep[i] += square(y_rep[n] - p) / den;
  }

  for (n in 1:N_rt) {
    int j = person_rt[n];
    int i = item_rt[n];
    real mu = beta[i] - tau[j];
    real sd = 1 / nu[i];

    log_lik_rt[n] = normal_lpdf(log_rt[n] | mu, sd);
    log_rt_rep[n] = normal_rng(mu, sd);

    L_obs[i] += square(log_rt[n] - mu) / square(sd);
    L_rep[i] += square(log_rt_rep[n] - mu) / square(sd);
  }
}
