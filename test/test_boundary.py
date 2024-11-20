import numpy as np

from main import X_train, predicted_sales_for_input


def test_boundary_case(model, feature_columns):
    """
    Test the model with boundary values and handle results with if-else.
    """
    # Create extreme feature values (e.g., maximum possible values)
    boundary_input = [np.max(X_train[col]) for col in feature_columns]

    try:
        # Try predicting sales for boundary input
        result = predicted_sales_for_input(model, boundary_input)

        # Use if-else to validate the result
        if result is not None and result > 0:
            print(f"Boundary Test Case: PASSED. Result: {result}")
        else:
            print("Boundary Test Case: FAILED. Result is invalid or not positive.")
    except Exception as e:
        # Exception block considered as test failure
        print(f"Boundary Test Case: FAILED. Exception occurred: {e}")
