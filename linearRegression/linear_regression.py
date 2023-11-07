from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import sqrt
import numpy.typing as npt


'''
If there is a test set the return will be only the trained model. Otherwise, the return 
will consist in the trained model, the test set variables (X) and labels (y)
'''
def linear_regression(
    X: npt.NDArray, 
    y: npt.NDArray, 
    test_size: float,
    standardize: bool = False
):
    model = LinearRegression()
    # Split data into training and testing set
    if test_size == 0: 
        X_train, y_train = X, y
        # Train the model
        model.fit(X_train, y_train)
        return model
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=4
        )
        if standardize:
            # Standard Scaler
            sc_x = StandardScaler().fit(X)
            sc_y = StandardScaler().fit(y)
            # Transform the train and test set with the scaler
            X_train = sc_x.transform(X_train)
            X_test = sc_x.transform(X_test)
            y_train = sc_y.transform(y_train)
            y_test = sc_y.transform(y_test)

        # Train the model
        model.fit(X_train, y_train)
        return model, X_test, y_test


'''
A test set labels (y) is obligatory, so as a prediction with them
'''
def evaluation_metric(
    y_test: npt.NDArray,
    y_pred: npt.NDArray,
    metric: str
) -> float:
    if metric == 'MAE': return metrics.mean_absolute_error(y_test, y_pred).round(4)
    elif metric == 'MSE': return metrics.mean_squared_error(y_test, y_pred).round(4)
    elif metric == 'RMSE': return sqrt(metrics.mean_squared_error(y_test, y_pred)).round(4)
    elif metric == 'r2': return metrics.r2_score(y_test, y_pred).round(4)
    elif metric == 'all': 
        return {'MAE': metrics.mean_absolute_error(y_test, y_pred).round(4),
                'MSE': metrics.mean_squared_error(y_test, y_pred).round(4),
                'RMSE': sqrt(metrics.mean_squared_error(y_test, y_pred)).round(4),
                'r2_score': metrics.r2_score(y_test, y_pred).round(4)
                }