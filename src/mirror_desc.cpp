/* mirror_desc.cpp
 * To conduct the mirror descent algorithm in C++
 */

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Raw function for Mirror Descent (Exponentiated Gradient Descent).
//
// @title Raw function for Mirror Descent (Exponentiated Gradient Descent)
// @param v The vector for which we are optimizing the objective
// function. In the synthetic control example, this is the region weights
// @param alpha The step-size
// @param grad The gradient of the object function
arma::mat f_mirror_desc(arma::mat v, double alpha, arma::mat grad) {

  //Exponentiated Gradient Descent (or mirror-descent for entropy regularizer)
  arma::mat h = v % exp(-alpha * grad);
  double sum_h = as_scalar(sum(h));
  h = h / sum_h;

  return h;

}

//' Mirror Descent (Exponentiated Gradient Descent) for Synthetic Control.
//'
//' This algorithm is from Aethy et al. (2017; Matrix Completion). This
//' function will minimize with respect to w
//'
//' (X0 * w - X1)'(X0 * w - X1)
//'
//' The matrix X where X0 is the controls and X1 is the treated vector.
//' Before passing X1 and X0 to this function, X needs to be scaled
//' to have zero rowmeans, variance 1, and normalized by the largest value.
//'
//' @title Mirror Descent (Exponentiated Gradient Descent) for Synthetic Control
//' @param X0_scaled The scaled matrix of controls
//' @param X1_scaled The scaled matrix (column vector) for the treated values
//' @param V The diagonal matrix for the weights for each variable
//' @param niter The number of interations for the algorithm. Defauls to
//' 10,000
//' @param rel_tol The stopping tolerance. Defaults to 1e-8
//' @export
// [[Rcpp::export]]
arma::mat mirror_desc(arma::mat X0_scaled, arma::mat X1_scaled, arma::mat V,
		      int niter = 10000, double rel_tol = 1e-8) {

  // The number of treated units
  double n = X0_scaled.n_cols;
  // The weights on the treated units
  arma::mat w = (1 / n) * arma::ones<arma::mat>(n, 1);

  // Set up the optimization
  arma::mat J = X0_scaled.t() * V * X0_scaled;
  arma::mat g = X0_scaled.t() * V * X1_scaled;
  arma::mat h = X1_scaled.t() * V * X1_scaled;

  // The objective value for the weights
  double obj_val = as_scalar(w.t() * J * w - 2 * w.t() * g + h);

  // The loop
  double alpha = 1;
  for (int itr = 1; itr < niter; itr++) {
    double step_size = alpha;
    arma::mat grad = 2 * (J * w - g);
    arma::mat w_np = f_mirror_desc(w, step_size, grad);
    double obj_val_n = as_scalar(w_np.t() * J * w_np - 2 * w_np.t() * g + h);
    double rel_imp = (obj_val - obj_val_n) / obj_val;
    if(obj_val_n < 1e-14) {
      w = w_np;
      break;
    }
    if (rel_imp < 0) {
      alpha = 0.95 * alpha;
    } else {
      w = w_np;
      obj_val = obj_val_n;
    }
    if ((rel_imp > 0) & (rel_imp < rel_tol) ){
      w = w_np;
      break;
    }
  }
  return w;
}
