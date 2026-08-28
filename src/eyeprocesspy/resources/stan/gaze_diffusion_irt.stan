functions {
  real selected_wiener_lpdf(real rt, int response, real boundary, real nondecision, real starting, real drift) {
    if (response == 1) return wiener_lpdf(rt | boundary, nondecision, starting, drift);
    return wiener_lpdf(rt | boundary, nondecision, 1 - starting, -drift);
  }
  real selected_wiener_lcdf(real rt, int response, real boundary, real nondecision, real starting, real drift) {
    if (response == 1) return wiener_lcdf_unnorm(rt, boundary, nondecision, starting, drift);
    return wiener_lcdf_unnorm(rt, boundary, nondecision, 1 - starting, -drift);
  }
  real selected_wiener_lccdf(real rt, int response, real boundary, real nondecision, real starting, real drift) {
    if (response == 1) return wiener_lccdf_unnorm(rt, boundary, nondecision, starting, drift);
    return wiener_lccdf_unnorm(rt, boundary, nondecision, 1 - starting, -drift);
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> J;
  int<lower=0> Fd;
  int<lower=0> Fb;
  int<lower=0> Fn;
  int<lower=0> Fs;
  matrix[N, Fd] Xd;
  matrix[N, Fb] Xb;
  matrix[N, Fn] Xn;
  matrix[N, Fs] Xs;
  array[N] int<lower=0, upper=1> y;
  vector<lower=0>[N] rt;
  array[N] int<lower=0, upper=2> censor;
  array[N] int<lower=1, upper=P> person;
  array[N] int<lower=1, upper=J> item;
  real<lower=0> min_rt;
  real<lower=0> rt_lower;
  real<lower=rt_lower> rt_upper;
  int<lower=0, upper=1> use_contaminant;
}
parameters {
  vector[Fd] beta_drift;
  vector[Fb] beta_boundary;
  vector[Fn] beta_nondecision;
  vector[Fs] beta_starting;
  vector[P] person_drift_raw;
  vector[J] item_difficulty_raw;
  vector[P] person_boundary_raw;
  vector[J] item_boundary_raw;
  real<lower=0> person_drift_sd;
  real<lower=0> item_difficulty_sd;
  real<lower=0> person_boundary_sd;
  real<lower=0> item_boundary_sd;
  real drift_intercept;
  real boundary_intercept;
  real nondecision_intercept;
  real starting_intercept;
  real contaminant_logit;
}
transformed parameters {
  vector[P] person_drift = person_drift_sd * (person_drift_raw - mean(person_drift_raw));
  vector[J] item_difficulty = item_difficulty_sd * (item_difficulty_raw - mean(item_difficulty_raw));
  vector[P] person_boundary = person_boundary_sd * (person_boundary_raw - mean(person_boundary_raw));
  vector[J] item_boundary = item_boundary_sd * (item_boundary_raw - mean(item_boundary_raw));
  real<lower=0, upper=0.2> contaminant_probability = 0.2 * inv_logit(contaminant_logit) * use_contaminant;
}
model {
  beta_drift ~ normal(0, 0.5);
  beta_boundary ~ normal(0, 0.3);
  beta_nondecision ~ normal(0, 0.3);
  beta_starting ~ normal(0, 0.3);
  person_drift_raw ~ std_normal();
  item_difficulty_raw ~ std_normal();
  person_boundary_raw ~ std_normal();
  item_boundary_raw ~ std_normal();
  person_drift_sd ~ normal(0, 1);
  item_difficulty_sd ~ normal(0, 1);
  person_boundary_sd ~ normal(0, 0.5);
  item_boundary_sd ~ normal(0, 0.5);
  drift_intercept ~ normal(0, 1);
  boundary_intercept ~ normal(0, 0.7);
  nondecision_intercept ~ normal(-2, 0.7);
  starting_intercept ~ normal(0, 0.7);
  contaminant_logit ~ normal(-3, 1);
  for (n in 1:N) {
    real drift = drift_intercept + person_drift[person[n]] - item_difficulty[item[n]] + Xd[n] * beta_drift;
    real boundary = exp(boundary_intercept + person_boundary[person[n]] + item_boundary[item[n]] + Xb[n] * beta_boundary);
    real nondecision = 0.95 * min_rt * inv_logit(nondecision_intercept + Xn[n] * beta_nondecision);
    real starting = 0.02 + 0.96 * inv_logit(starting_intercept + Xs[n] * beta_starting);
    real diffusion_lp;
    real contaminant_lp;
    if (censor[n] == 0) {
      diffusion_lp = selected_wiener_lpdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = -log(rt_upper - rt_lower) - log(2);
    } else if (censor[n] == 1) {
      diffusion_lp = selected_wiener_lccdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = log(fmax(rt_upper - rt[n], 1e-12) / (rt_upper - rt_lower)) - log(2);
    } else {
      diffusion_lp = selected_wiener_lcdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = log(fmax(rt[n] - rt_lower, 1e-12) / (rt_upper - rt_lower)) - log(2);
    }
    if (use_contaminant == 1) target += log_mix(contaminant_probability, contaminant_lp, diffusion_lp);
    else target += diffusion_lp;
  }
}
generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  vector[N] rt_rep;
  for (n in 1:N) {
    real drift = drift_intercept + person_drift[person[n]] - item_difficulty[item[n]] + Xd[n] * beta_drift;
    real boundary = exp(boundary_intercept + person_boundary[person[n]] + item_boundary[item[n]] + Xb[n] * beta_boundary);
    real nondecision = 0.95 * min_rt * inv_logit(nondecision_intercept + Xn[n] * beta_nondecision);
    real starting = 0.02 + 0.96 * inv_logit(starting_intercept + Xs[n] * beta_starting);
    real diffusion_lp;
    real contaminant_lp;
    if (censor[n] == 0) {
      diffusion_lp = selected_wiener_lpdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = -log(rt_upper - rt_lower) - log(2);
    } else if (censor[n] == 1) {
      diffusion_lp = selected_wiener_lccdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = log(fmax(rt_upper - rt[n], 1e-12) / (rt_upper - rt_lower)) - log(2);
    } else {
      diffusion_lp = selected_wiener_lcdf(rt[n] | y[n], boundary, nondecision, starting, drift);
      contaminant_lp = log(fmax(rt[n] - rt_lower, 1e-12) / (rt_upper - rt_lower)) - log(2);
    }
    log_lik[n] = use_contaminant == 1 ? log_mix(contaminant_probability, contaminant_lp, diffusion_lp) : diffusion_lp;
    y_rep[n] = bernoulli_rng(inv_logit(2 * drift * boundary));
    rt_rep[n] = nondecision + lognormal_rng(log(fmax(boundary / fmax(abs(drift), 0.25), 0.05)), 0.35);
  }
}
