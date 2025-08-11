# Optimization-for-Regression-CMPUT-267-Assignment-4-
This notebook demonstrates linear regression optimization on the scikit-learn California Housing dataset. It implements and compares multiple learners and step-size schedules, with interactive visualizations to inspect model fit and training dynamics.

What it does
- Loads & explores data; builds design matrices with a bias column.

- Closed-form OLS (closed_form_learner) via the normal equation.

- Batch Gradient Descent (batch_gradient_descent_learner) optimizing MSE.

- Mini-Batch Gradient Descent (minibatch_gradient_descent_learner) with configurable batch_size.

- Polynomial Regression (degree 2) via a custom feature map phi_2 and normalization (transform_and_normalize).

- Step-size schedules: constant, inverse decay, exponential decay, and a normalized-gradient step size.

- Visualizations: widgets to toggle plots for data, fitted predictors, loss vs. epochs, and schedule comparisons.

Key functions
- closed_form_learner(X, Y) -> (predictor, w_hat)

- batch_gradient_descent_learner(X, Y, step_size, epochs, random_seed) -> (predictor, w)

- minibatch_gradient_descent_learner(X, Y, step_size, epochs, batch_size, random_seed) -> (predictor, w)

- phi_2(x) (degree-2 feature map), transform_and_normalize(X), calculate_loss(X, Y, w)

- Step sizes: constant_step_size, inverse_decaying_step_size, exponential_decaying_step_size, normalized_gradient_step_size

Requirements
numpy, pandas, matplotlib, scikit-learn, ipywidgets, otter-grader (optional for autograding).

How to run (just download and open in google colab)
- pip install -r requirements.txt (or install the packages above)
- jupyter notebook "Yusuf_ass4.ipynb"
- Run cells top-to-bottom and use the checkboxes/sliders to reveal plots and compare learners.
