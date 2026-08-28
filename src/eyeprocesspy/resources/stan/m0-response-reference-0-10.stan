data {
  int<lower=1> J;
  int<lower=1> I;

  int<lower=1> N_response;
  array[N_response] int<lower=1, upper=J> person_response;
  array[N_response] int<lower=1, upper=I> item_response;
  array[N_response] int<lower=0, upper=1> y_response;

  int<lower=1, upper=2> prior_profile;
}

parameters {
  vector[J] theta_raw;
  real<lower=0> sigma_theta;

  vector[I] b_raw;
  real mu_b;
  real<lower=0> sigma_b;
}

transformed parameters {
  vector[J] theta = sigma_theta * theta_raw;
  vector[I] b = mu_b + sigma_b * b_raw;
}

model {
  theta_raw ~ std_normal();
  b_raw ~ std_normal();

  if (prior_profile == 1) {
    sigma_theta ~ normal(0, 1);
    mu_b ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
  } else {
    sigma_theta ~ normal(0, 2);
    mu_b ~ normal(0, 0.5);
    sigma_b ~ normal(0, 2);
  }

  for (n in 1:N_response) {
    y_response[n] ~ bernoulli_logit(
      theta[person_response[n]] - b[item_response[n]]
    );
  }
}

generated quantities {
  vector[N_response] log_lik_response;
  array[N_response] int y_rep;
  vector[I] W_obs = rep_vector(0.0, I);
  vector[I] W_rep = rep_vector(0.0, I);

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
}
